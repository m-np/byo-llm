"""
Unit tests for the one piece of lambda/streaming_handler.py that's actually
implemented: translating a single already-parsed LMI stream chunk into an
OpenAI-compatible SSE line. See streaming_handler.py's module docstring for
what's stubbed vs. real.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lambda"))

import streaming_handler  # noqa: E402


def test_chunk_translation_wraps_content_delta():
    lmi_chunk = {"choices": [{"delta": {"content": "Hel"}, "index": 0}]}
    line = streaming_handler.lmi_chunk_to_openai_sse_chunk(
        lmi_chunk, request_id="abc", model="qwen2.5-14b-awq", created=123
    )
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    payload = json.loads(line[len("data: ") :])
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "Hel"
    assert payload["id"] == "chatcmpl-abc"


def test_chunk_translation_empty_delta_on_no_choices():
    line = streaming_handler.lmi_chunk_to_openai_sse_chunk(
        {"choices": []}, request_id="abc", model="m", created=0
    )
    payload = json.loads(line[len("data: ") :])
    assert payload["choices"][0]["delta"] == {}


def test_chunk_translation_carries_finish_reason():
    lmi_chunk = {"choices": [{"delta": {}, "index": 0}]}
    line = streaming_handler.lmi_chunk_to_openai_sse_chunk(
        lmi_chunk, request_id="abc", model="m", created=0, finish_reason="stop"
    )
    payload = json.loads(line[len("data: ") :])
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_done_line():
    assert streaming_handler.openai_sse_done_line() == "data: [DONE]\n\n"
