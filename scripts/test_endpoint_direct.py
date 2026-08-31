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
    python scripts/test_endpoint_direct.py --model qwen2.5-14b-awq
    python scripts/test_endpoint_direct.py --endpoint-name my-endpoint --prompt "Explain vLLM in one sentence."
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AWS_REGION, get_model  # noqa: E402

DEFAULT_PROMPT = "In one sentence, what is vLLM?"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model key from models.yaml")
    group.add_argument("--endpoint-name", help="Explicit SageMaker endpoint name")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--region", default=AWS_REGION)
    args = parser.parse_args(argv)

    endpoint_name = args.endpoint_name or get_model(args.model).endpoint_name

    # Raw payload the DJL/LMI (vLLM chat schema) container expects directly
    # -- this is what handler.py's chat_request_to_lmi_payload() produces
    # from a Chat Completions-style request. Sending it here bypasses that
    # translation on purpose.
    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    print(f"Endpoint: {endpoint_name}  (region {args.region})")
    print(f"Payload:  {json.dumps(payload)}\n")

    client = boto3.client("sagemaker-runtime", region_name=args.region)
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
