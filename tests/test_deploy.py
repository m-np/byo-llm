"""
Unit tests for deploy.py, independent of real AWS. Uses a fake SageMaker
client (same pattern as tests/test_teardown.py) plus monkeypatching of
deploy.HF_TOKEN to exercise the gated-model path without touching .env.

Run: pytest tests/test_deploy.py -v
"""
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deploy  # noqa: E402
from config import ModelConfig  # noqa: E402


def _not_found_error(op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": f'Could not find x because {op} failed.'}},
        op,
    )


def _model(**overrides) -> ModelConfig:
    defaults = dict(
        key="test-model",
        hf_model_id="org/test-model",
        instance_type="ml.g5.2xlarge",
        tensor_parallel_degree=1,
        max_model_len=8192,
        quantize=None,
        dtype=None,
        requires_hf_token=False,
        note="",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class FakeSageMakerClient:
    """Enough surface area to drive deploy.check_no_existing_resources() and
    deploy.deploy()."""

    def __init__(self, model_present=False, config_present=False, endpoint_present=False):
        self.model_present = model_present
        self.config_present = config_present
        self.endpoint_present = endpoint_present
        self.calls: list[tuple] = []

    def describe_model(self, ModelName):
        self.calls.append(("describe_model", ModelName))
        if not self.model_present:
            raise _not_found_error("DescribeModel")
        return {"ModelName": ModelName}

    def describe_endpoint_config(self, EndpointConfigName):
        self.calls.append(("describe_endpoint_config", EndpointConfigName))
        if not self.config_present:
            raise _not_found_error("DescribeEndpointConfig")
        return {"EndpointConfigName": EndpointConfigName}

    def describe_endpoint(self, EndpointName):
        self.calls.append(("describe_endpoint", EndpointName))
        if not self.endpoint_present:
            raise _not_found_error("DescribeEndpoint")
        return {"EndpointName": EndpointName, "EndpointStatus": "InService"}

    def create_model(self, **kwargs):
        self.calls.append(("create_model", kwargs))
        self.model_present = True
        return {}

    def create_endpoint_config(self, **kwargs):
        self.calls.append(("create_endpoint_config", kwargs))
        self.config_present = True
        return {}

    def create_endpoint(self, **kwargs):
        self.calls.append(("create_endpoint", kwargs))
        self.endpoint_present = True
        return {"EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/test"}


# ---- build_environment ----------------------------------------------------


def test_build_environment_sets_core_vllm_engine_vars(monkeypatch):
    monkeypatch.setattr(deploy, "HF_TOKEN", None)
    model = _model()
    env = deploy.build_environment(model)

    assert env["HF_MODEL_ID"] == "org/test-model"
    assert env["OPTION_ENGINE"] == "vLLM"
    assert env["OPTION_ROLLING_BATCH"] == "vllm"
    assert env["OPTION_TENSOR_PARALLEL_DEGREE"] == "1"
    assert env["OPTION_MAX_MODEL_LEN"] == "8192"
    assert "HF_TOKEN" not in env


def test_build_environment_includes_quantize_when_set(monkeypatch):
    monkeypatch.setattr(deploy, "HF_TOKEN", None)
    model = _model(quantize="awq")
    env = deploy.build_environment(model)
    assert env["OPTION_QUANTIZE"] == "awq"


def test_build_environment_omits_quantize_when_unset(monkeypatch):
    monkeypatch.setattr(deploy, "HF_TOKEN", None)
    env = deploy.build_environment(_model(quantize=None))
    assert "OPTION_QUANTIZE" not in env


def test_build_environment_raises_if_gated_model_missing_hf_token(monkeypatch):
    monkeypatch.setattr(deploy, "HF_TOKEN", None)
    model = _model(requires_hf_token=True)
    with pytest.raises(RuntimeError, match="gated"):
        deploy.build_environment(model)


def test_build_environment_includes_hf_token_for_gated_model_when_set(monkeypatch):
    monkeypatch.setattr(deploy, "HF_TOKEN", "hf_faketoken")
    model = _model(requires_hf_token=True)
    env = deploy.build_environment(model)
    assert env["HF_TOKEN"] == "hf_faketoken"


def test_build_environment_includes_hf_token_even_for_ungated_model_if_set(monkeypatch):
    """Harmless to pass HF_TOKEN along even when not required -- some
    ungated repos still benefit from higher HF rate limits with a token."""
    monkeypatch.setattr(deploy, "HF_TOKEN", "hf_faketoken")
    env = deploy.build_environment(_model(requires_hf_token=False))
    assert env["HF_TOKEN"] == "hf_faketoken"


# ---- check_no_existing_resources ------------------------------------------


def test_check_no_existing_resources_empty_when_nothing_exists():
    client = FakeSageMakerClient()
    model = _model()
    assert deploy.check_no_existing_resources(client, model) == []


def test_check_no_existing_resources_flags_all_three():
    client = FakeSageMakerClient(model_present=True, config_present=True, endpoint_present=True)
    model = _model()
    existing = deploy.check_no_existing_resources(client, model)
    assert len(existing) == 3
    assert any("model" in e for e in existing)
    assert any("endpoint config" in e for e in existing)
    assert any("endpoint '" in e for e in existing)


def test_check_no_existing_resources_flags_only_what_exists():
    client = FakeSageMakerClient(model_present=True, config_present=False, endpoint_present=False)
    existing = deploy.check_no_existing_resources(client, _model())
    assert len(existing) == 1
    assert "model" in existing[0]


# ---- deploy() ---------------------------------------------------------------


def test_deploy_creates_resources_in_order_model_config_endpoint(monkeypatch):
    monkeypatch.setattr(deploy, "HF_TOKEN", None)
    client = FakeSageMakerClient()
    model = _model()

    deploy.deploy(client, model, "123.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.28.0", "arn:aws:iam::123:role/sm")

    op_order = [c[0] for c in client.calls]
    assert op_order == ["create_model", "create_endpoint_config", "create_endpoint"]


def test_deploy_wires_model_name_through_endpoint_config(monkeypatch):
    monkeypatch.setattr(deploy, "HF_TOKEN", None)
    client = FakeSageMakerClient()
    model = _model()

    deploy.deploy(client, model, "image-uri", "role-arn")

    create_model_call = next(c for c in client.calls if c[0] == "create_model")
    create_config_call = next(c for c in client.calls if c[0] == "create_endpoint_config")

    assert create_model_call[1]["ModelName"] == model.model_name
    assert create_model_call[1]["ExecutionRoleArn"] == "role-arn"
    assert create_model_call[1]["PrimaryContainer"]["Image"] == "image-uri"

    variant = create_config_call[1]["ProductionVariants"][0]
    assert variant["ModelName"] == model.model_name
    assert variant["InstanceType"] == model.instance_type
    assert variant["InitialInstanceCount"] == 1


def test_deploy_endpoint_uses_matching_config_name(monkeypatch):
    monkeypatch.setattr(deploy, "HF_TOKEN", None)
    client = FakeSageMakerClient()
    model = _model()

    deploy.deploy(client, model, "image-uri", "role-arn")

    create_endpoint_call = next(c for c in client.calls if c[0] == "create_endpoint")
    assert create_endpoint_call[1]["EndpointName"] == model.endpoint_name
    assert create_endpoint_call[1]["EndpointConfigName"] == model.endpoint_config_name


# ---- wait_for_in_service ----------------------------------------------------


class _StatusSequenceClient:
    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.calls = 0

    def describe_endpoint(self, EndpointName):
        self.calls += 1
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return {"EndpointStatus": status, "FailureReason": "boom"}


def test_wait_for_in_service_returns_on_in_service():
    client = _StatusSequenceClient(["Creating", "Creating", "InService"])
    status = deploy.wait_for_in_service(client, "ep", poll_s=0)
    assert status == "InService"


def test_wait_for_in_service_raises_on_failed():
    client = _StatusSequenceClient(["Creating", "Failed"])
    with pytest.raises(RuntimeError, match="boom"):
        deploy.wait_for_in_service(client, "ep", poll_s=0)
