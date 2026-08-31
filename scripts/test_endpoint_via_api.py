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
    python scripts/test_endpoint_via_api.py                        # reads API_GATEWAY_URL from .env
    python scripts/test_endpoint_via_api.py https://abc123.execute-api.us-east-1.amazonaws.com/v1/chat/completions

Edit the constants below to change the prompt or generation params -- no
flags to look up.

---
Same request, called directly with curl instead of this script -- no
client library, Python, or dependency of any kind needed, since this is
just plain JSON over HTTP:

    curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
            "model": "qwen2.5-14b-awq",
            "messages": [{"role": "user", "content": "In one sentence, what is vLLM?"}]
          }'
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPO_ROOT  # noqa: E402  (also triggers load_dotenv() via config's import)

# --- Edit these ---
MODEL = "qwen2.5-14b-awq"  # cosmetic -- the adapter always calls the one endpoint it's configured for
PROMPT = "In one sentence, what is vLLM?"
MAX_TOKENS = 128
TIMEOUT_S = 90.0  # Lambda itself times out at 60s -- see infra/setup_lambda_api.py
# ---


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("API_GATEWAY_URL")
    if not url:
        print(
            "Error: no URL given and API_GATEWAY_URL is not set in .env.\n"
            f"Pass one as an argument, or set it after running infra/setup_lambda_api.py "
            f"(see {REPO_ROOT / '.env.example'})."
        )
        return 1

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }

    print(f"POST {url}")
    print(f"Body: {json.dumps(body)}\n")

    start = time.time()
    try:
        resp = requests.post(url, json=body, timeout=TIMEOUT_S)
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
