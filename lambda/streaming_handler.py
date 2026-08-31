"""
STUB / architecture note -- not wired up, not deployed by infra/setup_lambda_api.py.

v1 (handler.py) is intentionally non-streaming: classic API Gateway (both
REST and the HTTP API used here) buffers the full Lambda response before
returning it to the client -- there is no way to forward tokens as they're
generated through that path, regardless of what the Lambda itself does.

To get real token-by-token streaming, the path is:

    Chat Completions-style client (stream=True)
        -> Lambda Function URL (InvokeMode=RESPONSE_STREAM)   <-- NOT API Gateway
        -> this Lambda, writing SSE chunks as they arrive
        -> sagemaker-runtime InvokeEndpointWithResponseStream  <-- NOT InvokeEndpoint
        -> DJL/LMI container (OPTION_ROLLING_BATCH=vllm streams tokens as
           they're generated when invoked via the streaming API)

If the calling app happens to be using the OpenAI SDK, this is what it's
sending/expecting at each end of that path -- shown here since it's the
most common client people already have installed, not because the
protocol is OpenAI-specific:

    for chunk in client.chat.completions.create(model="...", messages=[...], stream=True):
        print(chunk.choices[0].delta.content or "", end="")
    # each `chunk` on the wire is exactly one line from
    # lmi_chunk_to_chat_stream_chunk() below, and the loop ends on the
    # `data: [DONE]` line from sse_done_line().

Two things worth flagging honestly rather than guessing at exact current
syntax, since both are runtime/version-sensitive:

1. Python Lambda response streaming. As of this writing, AWS's
   `awslambda.streamifyResponse()` wrapper is documented for the Node.js
   managed runtime. Python doesn't have a first-party equivalent in the
   zip-based managed runtime -- the two AWS-documented ways to get response
   streaming out of a Python Lambda are:
     a) Package as a **container image** implementing the Lambda Runtime
        API directly (which natively supports chunked/streaming responses
        regardless of language), or
     b) Use the `aws-lambda-web-adapter` layer (github.com/awslabs/aws-lambda-web-adapter)
        in front of a normal ASGI app (e.g. FastAPI) that returns a
        StreamingResponse -- this is the path sketched below, since it lets
        the translation logic stay in plain Python/FastAPI instead of
        hand-rolling the Runtime API.
   VERIFY the current state of Python response streaming support against
   AWS's docs before building on this -- it has changed over time and may
   have changed again since.

2. The exact JSON shape DJL/LMI emits per streamed chunk under
   OPTION_ROLLING_BATCH=vllm (token-by-token JSON lines vs. SSE-framed
   `data: {...}` vs. something else) is container-version-dependent.
   `lmi_chunk_to_chat_stream_chunk()` below assumes a delta-shaped
   per-token JSON object (`{"choices": [{"delta": {"content": "..."}, ...}]}`)
   because that's what recent LMI chat-schema streaming responses use --
   confirm against a real invoke_endpoint_with_response_stream() response
   for your pinned container version before relying on it.

What IS implemented and unit-tested below: the pure translation of one
already-parsed LMI stream chunk into a Chat Completions-style
`chat.completion.chunk` SSE line, since that logic doesn't depend on any
of the above and is worth having ready. See tests/test_streaming_handler.py.
"""
from __future__ import annotations

import json
from typing import Optional


def lmi_chunk_to_chat_stream_chunk(
    lmi_chunk: dict,
    *,
    request_id: str,
    model: str,
    created: int,
    finish_reason: Optional[str] = None,
) -> str:
    """Translate one streamed LMI/vLLM chunk into a single Chat
    Completions-style Server-Sent Event line: `data: {...}\\n\\n`.

    lmi_chunk is expected to look like:
        {"choices": [{"delta": {"content": "some token(s)"}, "index": 0}]}
    (see module docstring point 2 -- verify against your container version).
    """
    delta_content = ""
    choices = lmi_chunk.get("choices") or []
    if choices:
        delta_content = (choices[0].get("delta") or {}).get("content", "")

    payload = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def sse_done_line() -> str:
    """The terminal line the Chat Completions streaming protocol expects
    after the last content chunk."""
    return "data: [DONE]\n\n"


# --- Sketch of the FastAPI + aws-lambda-web-adapter entrypoint -------------
#
# Deliberately not imported/executed anywhere -- this repo does not depend
# on fastapi or the web adapter layer (see requirements.txt). Uncomment and
# flesh out if/when streaming becomes a real requirement:
#
# from fastapi import FastAPI, Request
# from fastapi.responses import StreamingResponse
# import boto3, os, time, uuid
#
# app = FastAPI()
# sm_runtime = boto3.client("sagemaker-runtime", region_name=os.environ["AWS_REGION"])
#
# @app.post("/v1/chat/completions")
# async def chat_completions(request: Request):
#     body = await request.json()
#     lmi_payload = {...}  # reuse chat_request_to_lmi_payload() from handler.py,
#                           # minus the `stream: true` rejection
#     request_id, created = uuid.uuid4().hex, int(time.time())
#
#     def event_stream():
#         resp = sm_runtime.invoke_endpoint_with_response_stream(
#             EndpointName=os.environ["SAGEMAKER_ENDPOINT_NAME"],
#             ContentType="application/json",
#             Body=json.dumps(lmi_payload),
#         )
#         for event in resp["Body"]:
#             chunk = json.loads(event["PayloadPart"]["Bytes"])
#             yield lmi_chunk_to_chat_stream_chunk(
#                 chunk, request_id=request_id, model=body.get("model", ""), created=created
#             )
#         yield sse_done_line()
#
#     return StreamingResponse(event_stream(), media_type="text/event-stream")
#
# Deployed as a Lambda container image (or zip + aws-lambda-web-adapter
# layer) behind a Function URL with InvokeMode=RESPONSE_STREAM -- not
# behind API Gateway.
