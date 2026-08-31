"""
Unit tests for lambda/handler.py, independent of real AWS. boto3 is mocked
throughout -- no network calls, no real credentials needed.

Run: pytest tests/test_handler.py -v
"""
import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lambda"))

import handler  # noqa: E402


# ---- openai_request_to_lmi_payload ----------------------------------------


def test_request_translation_basic_passthrough():
    body = {
        "model": "qwen2.5-14b-awq",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    payload = handler.openai_request_to_lmi_payload(body)
    assert payload["messages"] == body["messages"]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 100


def test_request_translation_applies_default_max_tokens_when_absent():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    payload = handler.openai_request_to_lmi_payload(body)
    assert payload["max_tokens"] == handler.DEFAULT_MAX_TOKENS


def test_request_translation_drops_unsupported_fields():
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "user": "some-user-id",  # OpenAI field, not forwarded
        "logit_bias": {"123": 1},  # not in _FORWARDED_PARAMS
    }
    payload = handler.openai_request_to_lmi_payload(body)
    assert "user" not in payload
    assert "logit_bias" not in payload


def test_request_translation_forwards_known_generation_params():
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 0.9,
        "n": 2,
        "stop": ["\n"],
        "seed": 42,
    }
    payload = handler.openai_request_to_lmi_payload(body)
    assert payload["top_p"] == 0.9
    assert payload["n"] == 2
    assert payload["stop"] == ["\n"]
    assert payload["seed"] == 42


def test_request_translation_raises_on_missing_messages():
    with pytest.raises(handler.OpenAIError) as exc_info:
        handler.openai_request_to_lmi_payload({})
    assert exc_info.value.param == "messages"
    assert exc_info.value.status_code == 400


def test_request_translation_raises_on_empty_messages_array():
    with pytest.raises(handler.OpenAIError):
        handler.openai_request_to_lmi_payload({"messages": []})


def test_request_translation_raises_on_stream_true():
    with pytest.raises(handler.OpenAIError) as exc_info:
        handler.openai_request_to_lmi_payload(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
    assert exc_info.value.param == "stream"


def test_request_translation_allows_stream_false():
    payload = handler.openai_request_to_lmi_payload(
        {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    )
    assert "stream" not in payload  # not forwarded either way -- non-streaming container call


# ---- lmi_response_to_openai -------------------------------------------------


def test_response_translation_openai_shaped_choices():
    sm_body = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = handler.lmi_response_to_openai(
        sm_body, request_id="abc123", model="qwen2.5-14b-awq", created=1700000000
    )
    assert result["id"] == "chatcmpl-abc123"
    assert result["object"] == "chat.completion"
    assert result["created"] == 1700000000
    assert result["model"] == "qwen2.5-14b-awq"
    assert result["choices"][0]["message"]["content"] == "Hello there!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 15


def test_response_translation_defaults_finish_reason_and_role_if_missing():
    sm_body = {"choices": [{"message": {"content": "hi"}}]}
    result = handler.lmi_response_to_openai(sm_body, request_id="x", model="m", created=0)
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][0]["message"]["role"] == "assistant"


def test_response_translation_falls_back_on_legacy_generated_text_dict():
    sm_body = {"generated_text": "legacy style response"}
    result = handler.lmi_response_to_openai(sm_body, request_id="x", model="m", created=0)
    assert result["choices"][0]["message"]["content"] == "legacy style response"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] is None


def test_response_translation_falls_back_on_legacy_generated_text_list():
    sm_body = [{"generated_text": "first"}, {"generated_text": "second"}]
    result = handler.lmi_response_to_openai(sm_body, request_id="x", model="m", created=0)
    assert len(result["choices"]) == 2
    assert result["choices"][0]["message"]["content"] == "first"
    assert result["choices"][1]["message"]["content"] == "second"


# ---- sagemaker_error_to_openai_error ---------------------------------------


def _client_error(code: str, message: str = "boom") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "InvokeEndpoint")


def test_error_translation_model_error_maps_to_502():
    status, body = handler.sagemaker_error_to_openai_error(_client_error("ModelError"))
    assert status == 502
    assert body["error"]["code"] == "model_error"


def test_error_translation_validation_exception_maps_to_400():
    status, body = handler.sagemaker_error_to_openai_error(_client_error("ValidationException"))
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"


def test_error_translation_throttling_maps_to_429():
    status, body = handler.sagemaker_error_to_openai_error(_client_error("ThrottlingException"))
    assert status == 429
    assert body["error"]["type"] == "rate_limit_error"


def test_error_translation_unknown_code_maps_to_500():
    status, body = handler.sagemaker_error_to_openai_error(_client_error("SomethingElse"))
    assert status == 500
    assert body["error"]["code"] == "SomethingElse"


# ---- lambda_handler (boto3 mocked) -----------------------------------------


def _api_gateway_event(body: dict, is_base64: bool = False) -> dict:
    raw = json.dumps(body)
    if is_base64:
        return {"body": base64.b64encode(raw.encode()).decode(), "isBase64Encoded": True}
    return {"body": raw, "isBase64Encoded": False}


@pytest.fixture
def mock_sagemaker_runtime(monkeypatch):
    monkeypatch.setattr(handler, "SAGEMAKER_ENDPOINT_NAME", "test-endpoint")
    monkeypatch.setattr(handler, "_sagemaker_runtime_client", None)

    fake_client = MagicMock()
    monkeypatch.setattr(handler, "_get_client", lambda: fake_client)
    return fake_client


def _sm_body_stream(payload: dict):
    m = MagicMock()
    m.read.return_value = json.dumps(payload).encode()
    return m


def test_lambda_handler_happy_path(mock_sagemaker_runtime):
    mock_sagemaker_runtime.invoke_endpoint.return_value = {
        "Body": _sm_body_stream(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )
    }
    event = _api_gateway_event(
        {"model": "qwen2.5-14b-awq", "messages": [{"role": "user", "content": "hello"}]}
    )

    response = handler.lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["choices"][0]["message"]["content"] == "Hi!"
    assert body["object"] == "chat.completion"

    # Confirm the endpoint was invoked with a translated (not raw OpenAI) payload
    call_kwargs = mock_sagemaker_runtime.invoke_endpoint.call_args.kwargs
    assert call_kwargs["EndpointName"] == "test-endpoint"
    sent_payload = json.loads(call_kwargs["Body"])
    assert sent_payload["messages"] == [{"role": "user", "content": "hello"}]


def test_lambda_handler_decodes_base64_body(mock_sagemaker_runtime):
    mock_sagemaker_runtime.invoke_endpoint.return_value = {
        "Body": _sm_body_stream({"choices": [{"message": {"content": "ok"}}]})
    }
    event = _api_gateway_event(
        {"messages": [{"role": "user", "content": "hello"}]}, is_base64=True
    )
    response = handler.lambda_handler(event, None)
    assert response["statusCode"] == 200


def test_lambda_handler_rejects_invalid_json_body(mock_sagemaker_runtime):
    event = {"body": "{not valid json", "isBase64Encoded": False}
    response = handler.lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert "not valid JSON" in json.loads(response["body"])["error"]["message"]
    mock_sagemaker_runtime.invoke_endpoint.assert_not_called()


def test_lambda_handler_rejects_missing_messages(mock_sagemaker_runtime):
    event = _api_gateway_event({"model": "qwen2.5-14b-awq"})
    response = handler.lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"]["param"] == "messages"
    mock_sagemaker_runtime.invoke_endpoint.assert_not_called()


def test_lambda_handler_rejects_stream_true(mock_sagemaker_runtime):
    event = _api_gateway_event(
        {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    response = handler.lambda_handler(event, None)
    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"]["param"] == "stream"
    mock_sagemaker_runtime.invoke_endpoint.assert_not_called()


def test_lambda_handler_translates_sagemaker_client_error(mock_sagemaker_runtime):
    mock_sagemaker_runtime.invoke_endpoint.side_effect = _client_error(
        "ModelError", "container crashed"
    )
    event = _api_gateway_event({"messages": [{"role": "user", "content": "hi"}]})
    response = handler.lambda_handler(event, None)
    assert response["statusCode"] == 502
    assert "container crashed" in json.loads(response["body"])["error"]["message"]


def test_lambda_handler_returns_500_if_endpoint_name_unconfigured(monkeypatch):
    monkeypatch.setattr(handler, "SAGEMAKER_ENDPOINT_NAME", "")
    event = _api_gateway_event({"messages": [{"role": "user", "content": "hi"}]})
    response = handler.lambda_handler(event, None)
    assert response["statusCode"] == 500
