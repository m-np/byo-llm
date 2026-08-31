#!/usr/bin/env python3
"""
Hit the deployed adapter through the FULL path: API Gateway -> Lambda ->
SageMaker. Sends a standard Chat Completions-style request body over plain
HTTP (via `requests`, already in requirements.txt) so this script doesn't
depend on any particular client library.

Use this alongside test_endpoint_direct.py to isolate failures: if the
direct script works but this one doesn't, the problem is in
lambda/handler.py's translation logic or the API Gateway route/permissions
-- not the model itself.

Usage:
    python scripts/test_endpoint_via_api.py --url https://abc123.execute-api.us-east-1.amazonaws.com/v1/chat/completions
    python scripts/test_endpoint_via_api.py  # reads API_GATEWAY_URL from .env

---
If your app happens to already be built on the OpenAI SDK, this is the
real point of this repo -- point `base_url` at your deployment and nothing
else in the app needs to change. Not run by this script (keeps `openai`
out of requirements.txt for this experiment), but this is literally it:

    pip install openai
    from openai import OpenAI
    client = OpenAI(
        base_url="https://abc123.execute-api.us-east-1.amazonaws.com/v1",  # note: no /chat/completions -- the SDK appends that
        api_key="unused",  # this adapter doesn't check it; add API Gateway auth before exposing this beyond localhost
    )
    resp = client.chat.completions.create(
        model="qwen2.5-14b-awq",
        messages=[{"role": "user", "content": "In one sentence, what is vLLM?"}],
    )
    print(resp.choices[0].message.content)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT  # noqa: E402  (also triggers load_dotenv() via config's import)

DEFAULT_PROMPT = "In one sentence, what is vLLM?"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("API_GATEWAY_URL"),
        help="Full route URL, e.g. https://<api-id>.execute-api.<region>.amazonaws.com/v1/chat/completions "
        "(default: API_GATEWAY_URL from .env)",
    )
    parser.add_argument("--model", default="qwen2.5-14b-awq", help="Value sent in the 'model' field (cosmetic -- the adapter always calls the one endpoint it's configured for)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP timeout in seconds (Lambda itself times out at 60s -- see infra/setup_lambda_api.py)")
    args = parser.parse_args(argv)

    if not args.url:
        print(
            "Error: no --url given and API_GATEWAY_URL is not set in .env.\n"
            f"Set it after running infra/setup_lambda_api.py, or check {REPO_ROOT / '.env.example'}."
        )
        return 1

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
    }

    print(f"POST {args.url}")
    print(f"Body: {json.dumps(body)}\n")

    start = time.time()
    try:
        resp = requests.post(args.url, json=body, timeout=args.timeout)
    except requests.RequestException as e:
        print(f"❌ Request failed before getting an HTTP response: {e}")
        print("(This usually means the URL, API Gateway route, or network path is wrong --")
        print(" a reachable-but-erroring adapter would still return an HTTP response.)")
        return 1
    elapsed = time.time() - start

    print(f"Status: {resp.status_code}  ({elapsed:.2f}s)\n")
    try:
        print(json.dumps(resp.json(), indent=2))
    except ValueError:
        print(resp.text)

    if resp.status_code != 200:
        return 1

    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content")
    if content:
        print(f"\n✅ Assistant said: {content}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
