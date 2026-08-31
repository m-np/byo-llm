"""
Chat Completions-style request/response translation, plus the Lambda
entrypoint that wires it to one specific SageMaker endpoint.

"Chat Completions-style" means: a `{"messages": [{"role": ..., "content": ...}], ...}`
request in, a `{"choices": [{"message": {...}, "finish_reason": ...}], "usage": {...}}`
response out. That shape isn't invented here -- it's the same one used by
most hosted chat-model APIs, which is what makes this adapter a drop-in
`base_url` swap for apps already built against one of them (see the
comment above each translation call below for exactly what that swap looks
like).

Deliberately split: the translation functions below (chat_request_to_lmi_payload,
lmi_response_to_chat_completion, sagemaker_error_to_chat_error) are pure --
no boto3, no AWS calls -- so tests/test_handler.py can exercise them
directly and exhaustively without touching real infrastructure. Only
lambda_handler() itself calls boto3, and that's mocked in tests too (see
test_lambda_handler_* in tests/test_handler.py).

v1 is non-streaming only. A `stream: true` request gets a clean 400 error
rather than being silently ignored -- see streaming_handler.py for the
planned Function URL streaming path.
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

# Chat Completions-style request fields forwarded as-is to the DJL/LMI vLLM
# chat schema, which accepts a near-identical generation-params vocabulary
# when invoked with a "messages" payload (see deploy.py's build_environment()
# / OPTION_ROLLING_BATCH=vllm). Anything not in this set is dropped rather
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


class ChatAPIError(Exception):
    """Raised for request-shape problems we want to report back as a
    structured `{"error": {...}} JSON body instead of an unhandled 500."""

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

    def to_error_body(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": None,
            }
        }


def chat_request_to_lmi_payload(body: dict) -> dict:
    """Translate a Chat Completions-style request body into the payload the
    DJL/LMI vLLM container expects."""
    if "messages" not in body or not isinstance(body["messages"], list) or not body["messages"]:
        raise ChatAPIError("'messages' is required and must be a non-empty array.", param="messages")

    if body.get("stream"):
        raise ChatAPIError(
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


def lmi_response_to_chat_completion(sm_body: dict, *, request_id: str, model: str, created: int) -> dict:
    """Translate a DJL/LMI (vLLM, chat schema) response into a Chat
    Completions-style response object.

    Handles two response shapes:
      1. The expected shape when invoked with a "messages" payload under
         OPTION_ROLLING_BATCH=vllm: already close to the target
         `choices[].message` structure -- we normalize the envelope fields
         (id/object/created/model) LMI sets differently.
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


def sagemaker_error_to_chat_error(err: ClientError) -> tuple[int, dict]:
    """Map a boto3 SageMaker Runtime ClientError to (http_status, error_body)."""
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
        chat_request = json.loads(raw_body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _proxy_response(400, ChatAPIError("Request body is not valid JSON.").to_error_body())

    # ---- Equivalent client-side call, if you're using the OpenAI SDK against
    # this Lambda's URL (any client that speaks Chat Completions works the
    # same way -- this isn't OpenAI-specific plumbing, just the most common
    # client people already have installed):
    #
    #   from openai import OpenAI
    #   client = OpenAI(base_url="<this API's base URL>/v1", api_key="unused")
    #   client.chat.completions.create(model="...", messages=[...])
    #
    # `chat_request` here is exactly the JSON body that call sends.
    try:
        lmi_payload = chat_request_to_lmi_payload(chat_request)
    except ChatAPIError as e:
        return _proxy_response(e.status_code, e.to_error_body())

    model_name = chat_request.get("model", SAGEMAKER_ENDPOINT_NAME)
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
        # ---- If you're using the OpenAI SDK client-side, a non-2xx response
        # here surfaces as an `openai.APIError` (or a subclass, e.g.
        # `openai.RateLimitError` for our 429s) raised from
        # client.chat.completions.create(...) -- same as it would for a real
        # OpenAI outage/error, so existing try/except error handling in
        # calling code doesn't need to change either.
        status_code, error_body = sagemaker_error_to_chat_error(e)
        return _proxy_response(status_code, error_body)

    # ---- Equivalent client-side read, if you're using the OpenAI SDK:
    #
    #   resp = client.chat.completions.create(...)
    #   resp.choices[0].message.content
    #
    # `chat_response` below is exactly the JSON body that .create(...) call
    # parses into that `resp` object.
    chat_response = lmi_response_to_chat_completion(
        sm_body, request_id=request_id, model=model_name, created=created
    )
    return _proxy_response(200, chat_response)
