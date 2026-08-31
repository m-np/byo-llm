"""
OpenAI chat.completions <-> SageMaker DJL/LMI (vLLM) translation, plus the
Lambda entrypoint that wires it to one specific SageMaker endpoint.

Deliberately split: the translation functions below (openai_request_to_lmi_payload,
lmi_response_to_openai, sagemaker_error_to_openai_error) are pure -- no boto3,
no AWS calls -- so tests/test_handler.py can exercise them directly and
exhaustively without touching real infrastructure. Only lambda_handler()
itself calls boto3, and that's mocked in tests too (see
test_lambda_handler_* in tests/test_handler.py).

v1 is non-streaming only. A `stream: true` request gets a clean 400
OpenAI-style error rather than being silently ignored -- see
streaming_handler.py for the planned Function URL streaming path.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "512"))

# OpenAI request fields forwarded as-is to the DJL/LMI vLLM chat schema,
# which accepts a near-identical generation-params vocabulary when invoked
# with a "messages" payload (see deploy.py's build_environment() /
# OPTION_ROLLING_BATCH=vllm). Anything not in this set is dropped rather
# than forwarded, since the container rejects unrecognized fields.
_FORWARDED_PARAMS = (
    "temperature",
    "top_p",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "logprobs",
    "top_logprobs",
)

_sagemaker_runtime_client = None  # lazy singleton; created on first real invocation


def _get_client():
    global _sagemaker_runtime_client
    if _sagemaker_runtime_client is None:
        _sagemaker_runtime_client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    return _sagemaker_runtime_client


class OpenAIError(Exception):
    """Raised for request-shape problems we want to report back as a proper
    OpenAI-style {"error": {...}} JSON body instead of an unhandled 500."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param

    def to_openai_error_body(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": None,
            }
        }


def openai_request_to_lmi_payload(body: dict) -> dict:
    """Translate an OpenAI POST /v1/chat/completions request body into the
    payload the DJL/LMI vLLM container expects."""
    if "messages" not in body or not isinstance(body["messages"], list) or not body["messages"]:
        raise OpenAIError("'messages' is required and must be a non-empty array.", param="messages")

    if body.get("stream"):
        raise OpenAIError(
            "Streaming is not supported by this endpoint (v1 is non-streaming only). "
            "See lambda/streaming_handler.py for the planned Function URL streaming path.",
            param="stream",
        )

    payload: dict[str, Any] = {"messages": body["messages"]}
    payload["max_tokens"] = body.get("max_tokens", DEFAULT_MAX_TOKENS)

    for field in _FORWARDED_PARAMS:
        if field in body and body[field] is not None:
            payload[field] = body[field]

    return payload


def lmi_response_to_openai(sm_body: dict, *, request_id: str, model: str, created: int) -> dict:
    """Translate a DJL/LMI (vLLM, chat schema) response into an OpenAI
    ChatCompletion response object.

    Handles two response shapes:
      1. The expected shape when invoked with a "messages" payload under
         OPTION_ROLLING_BATCH=vllm: already close to OpenAI's own
         `choices[].message` structure -- we normalize the envelope fields
         (id/object/created/model) LMI doesn't set the OpenAI way.
      2. A defensive fallback for older/non-chat LMI response shapes
         ({"generated_text": "..."} or a list of those), wrapped into a
         synthetic single choice so this degrades gracefully instead of
         crashing if the container version changes underneath it.
    """
    if "choices" in sm_body:
        choices = []
        for i, c in enumerate(sm_body["choices"]):
            message = c.get("message") or {"role": "assistant", "content": c.get("text", "")}
            choices.append(
                {
                    "index": c.get("index", i),
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                    "finish_reason": c.get("finish_reason", "stop"),
                }
            )
        usage = sm_body.get(
            "usage", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        )
    else:
        generations = sm_body if isinstance(sm_body, list) else [sm_body]
        choices = [
            {
                "index": i,
                "message": {"role": "assistant", "content": g.get("generated_text", "")},
                "finish_reason": "stop",
            }
            for i, g in enumerate(generations)
        ]
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
    }


def sagemaker_error_to_openai_error(err: ClientError) -> tuple[int, dict]:
    """Map a boto3 SageMaker Runtime ClientError to (http_status, openai_error_body)."""
    code = err.response.get("Error", {}).get("Code", "")
    message = err.response.get("Error", {}).get("Message", str(err))

    if code == "ModelError":
        # The container itself returned a non-2xx (bad payload, engine crash, OOM, etc.)
        return 502, {"error": {"message": message, "type": "api_error", "param": None, "code": "model_error"}}
    if code == "ValidationException":
        return 400, {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}}
    if code in ("ThrottlingException", "ServiceUnavailable"):
        return 429, {"error": {"message": message, "type": "rate_limit_error", "param": None, "code": None}}
    return 500, {"error": {"message": message, "type": "api_error", "param": None, "code": code or None}}


def _proxy_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    """API Gateway HTTP API (payload format 2.0) proxy integration entrypoint."""
    if not SAGEMAKER_ENDPOINT_NAME:
        return _proxy_response(
            500,
            {
                "error": {
                    "message": "SAGEMAKER_ENDPOINT_NAME is not configured on this Lambda.",
                    "type": "api_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    try:
        raw_body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        openai_request = json.loads(raw_body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _proxy_response(400, OpenAIError("Request body is not valid JSON.").to_openai_error_body())

    try:
        lmi_payload = openai_request_to_lmi_payload(openai_request)
    except OpenAIError as e:
        return _proxy_response(e.status_code, e.to_openai_error_body())

    model_name = openai_request.get("model", SAGEMAKER_ENDPOINT_NAME)
    request_id = uuid.uuid4().hex
    created = int(time.time())

    client = _get_client()
    try:
        sm_response = client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(lmi_payload),
        )
        sm_body = json.loads(sm_response["Body"].read())
    except ClientError as e:
        status_code, error_body = sagemaker_error_to_openai_error(e)
        return _proxy_response(status_code, error_body)

    openai_response = lmi_response_to_openai(
        sm_body, request_id=request_id, model=model_name, created=created
    )
    return _proxy_response(200, openai_response)
