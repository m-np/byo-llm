#!/usr/bin/env python3
"""
Hit the SageMaker endpoint DIRECTLY via sagemaker-runtime InvokeEndpoint,
bypassing Lambda and API Gateway entirely.

Use this to isolate where a failure is: if this script works but
test_endpoint_via_api.py doesn't, the bug is in the Lambda translation
layer or API Gateway wiring, not the model/container. If this script
itself fails, the problem is the endpoint/model/vLLM config, and the
adapter layer isn't even reachable yet.

Usage:
    python scripts/test_endpoint_direct.py                        # uses MODEL_KEY below
    python scripts/test_endpoint_direct.py llama-3.1-8b-instruct   # or pass a different model key

Edit the constants below to change the prompt or generation params --
no flags to look up.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AWS_REGION, get_model  # noqa: E402

# --- Edit these ---
MODEL_KEY = "qwen2.5-14b-awq"  # overridden by an optional command-line argument
PROMPT = "In one sentence, what is vLLM?"
MAX_TOKENS = 128
TEMPERATURE = 0.7
REGION = AWS_REGION
# ---


def main() -> int:
    model_key = sys.argv[1] if len(sys.argv) > 1 else MODEL_KEY
    try:
        endpoint_name = get_model(model_key).endpoint_name
    except KeyError as e:
        print(f"Error: {e.args[0]}")
        return 1

    # Raw payload the DJL/LMI (vLLM chat schema) container expects directly
    # -- this is what handler.py's chat_request_to_lmi_payload() produces
    # from a Chat Completions-style request. Sending it here bypasses that
    # translation on purpose.
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    print(f"Endpoint: {endpoint_name}  (region {REGION})")
    print(f"Payload:  {json.dumps(payload)}\n")

    client = boto3.client("sagemaker-runtime", region_name=REGION)
    start = time.time()
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
    except Exception as e:
        print(f"❌ InvokeEndpoint failed: {e}")
        return 1
    elapsed = time.time() - start

    body = json.loads(response["Body"].read())
    print(f"✅ Response received in {elapsed:.2f}s:\n")
    print(json.dumps(body, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
