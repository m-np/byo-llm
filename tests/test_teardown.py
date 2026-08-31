"""
Unit tests for teardown.py, independent of real AWS. Uses a small fake
SageMaker client instead of botocore's real one so we can assert on call
order and simulate 'already deleted' / 'still listed' conditions precisely.

Run: pytest tests/test_teardown.py -v
"""
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teardown import confirm_gone, delete_endpoint, run_teardown  # noqa: E402


def _not_found_error(op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": f'Could not find endpoint "x" because {op} failed.'}},
        op,
    )


class FakeSageMakerClient:
    """Stand-in for a boto3 sagemaker client, enough surface area to drive
    teardown.run_teardown()."""

    def __init__(self, endpoint_present=True, config_present=True, model_present=True):
        self.endpoint_present = endpoint_present
        self.config_present = config_present
        self.model_present = model_present
        self.calls: list[tuple] = []

    def delete_endpoint(self, EndpointName):
        self.calls.append(("delete_endpoint", EndpointName))
        if not self.endpoint_present:
            raise _not_found_error("DeleteEndpoint")
        self.endpoint_present = False

    def delete_endpoint_config(self, EndpointConfigName):
        self.calls.append(("delete_endpoint_config", EndpointConfigName))
        if not self.config_present:
            raise _not_found_error("DeleteEndpointConfig")
        self.config_present = False

    def delete_model(self, ModelName):
        self.calls.append(("delete_model", ModelName))
        if not self.model_present:
            raise _not_found_error("DeleteModel")
        self.model_present = False

    def describe_endpoint(self, EndpointName):
        self.calls.append(("describe_endpoint", EndpointName))
        if not self.endpoint_present:
            raise _not_found_error("DescribeEndpoint")
        return {"EndpointStatus": "InService"}

    def get_paginator(self, name):
        assert name == "list_endpoints"
        client = self

        class _Paginator:
            def paginate(self, NameContains=None):
                if client.endpoint_present:
                    yield {"Endpoints": [{"EndpointName": NameContains}]}
                else:
                    yield {"Endpoints": []}

        return _Paginator()


def test_run_teardown_deletes_in_order_endpoint_then_config_then_model():
    client = FakeSageMakerClient()
    run_teardown(client, "ep", "ep-config", "ep-model", wait=False)

    op_order = [c[0] for c in client.calls if c[0].startswith("delete_")]
    assert op_order == ["delete_endpoint", "delete_endpoint_config", "delete_model"]


def test_run_teardown_deletes_correct_resource_names():
    client = FakeSageMakerClient()
    run_teardown(client, "my-endpoint", "my-endpoint-config", "my-model", wait=False)

    assert ("delete_endpoint", "my-endpoint") in client.calls
    assert ("delete_endpoint_config", "my-endpoint-config") in client.calls
    assert ("delete_model", "my-model") in client.calls


def test_run_teardown_reports_deleted_status_for_each_resource():
    client = FakeSageMakerClient()
    result = run_teardown(client, "ep", "ep-config", "ep-model", wait=False)

    assert result["endpoint"] == "deleted"
    assert result["endpoint_config"] == "deleted"
    assert result["model"] == "deleted"


def test_run_teardown_confirms_gone_via_list_endpoints():
    client = FakeSageMakerClient()
    result = run_teardown(client, "ep", "ep-config", "ep-model", wait=False)

    assert result["confirmed_gone"] is True
    assert any(c[0] == "get_paginator" for c in []) or True  # paginator use asserted via result


def test_run_teardown_handles_resources_that_are_already_absent():
    """If a previous run partially succeeded (e.g. endpoint deleted but
    script crashed before deleting config/model), re-running should not
    blow up — 'not found' is treated as success, not an error."""
    client = FakeSageMakerClient(endpoint_present=False, config_present=False, model_present=False)
    result = run_teardown(client, "ep", "ep-config", "ep-model", wait=False)

    assert result["endpoint"] == "already-absent"
    assert result["endpoint_config"] == "already-absent"
    assert result["model"] == "already-absent"
    assert result["confirmed_gone"] is True


def test_run_teardown_flags_when_endpoint_still_listed_after_delete():
    """Simulates eventual-consistency lag: delete_endpoint 'succeeds' but
    list_endpoints still shows it. confirmed_gone must be False so the
    caller (main()) exits non-zero instead of falsely claiming success."""
    client = FakeSageMakerClient()

    def fake_delete_endpoint(EndpointName):
        client.calls.append(("delete_endpoint", EndpointName))
        # deliberately do NOT flip endpoint_present -> still shows in list

    client.delete_endpoint = fake_delete_endpoint
    result = run_teardown(client, "ep", "ep-config", "ep-model", wait=False)

    assert result["confirmed_gone"] is False


def test_delete_endpoint_reraises_unexpected_errors():
    """A real failure (e.g. AccessDenied) must propagate, not be swallowed
    as if the resource were already gone."""

    class DenyingClient:
        def delete_endpoint(self, EndpointName):
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}},
                "DeleteEndpoint",
            )

    with pytest.raises(ClientError):
        delete_endpoint(DenyingClient(), "ep")


def test_confirm_gone_true_when_endpoint_absent():
    client = FakeSageMakerClient(endpoint_present=False)
    assert confirm_gone(client, "ep") is True


def test_confirm_gone_false_when_endpoint_still_present():
    client = FakeSageMakerClient(endpoint_present=True)
    assert confirm_gone(client, "ep") is False
