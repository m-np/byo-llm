#!/usr/bin/env python3
"""
Deploy a model from models.yaml to a SageMaker real-time inference endpoint,
using the DJL/LMI container with the vLLM backend.

Usage:
    python deploy.py --model qwen2.5-14b-awq
    python deploy.py                        # uses the model marked `default: true`

This creates three SageMaker resources: a Model, an EndpointConfig, and an
Endpoint. If any of the three already exist under the derived names, deploy
refuses to proceed — run `python teardown.py --model <key> --confirm` first.
Model swaps in this repo are "tear down, deploy fresh", not in-place
updates. That's simpler and safer for a short-lived experiment, and it
doesn't cost anything extra: a bare Model/EndpointConfig with no live
Endpoint behind it does NOT bill (only a running Endpoint instance does),
so a failed/interrupted deploy leaves at most inert metadata, never a
silent charge.

The endpoint takes several minutes to reach InService (container pull +
weight download from Hugging Face + vLLM engine init — expect 5-15+
minutes, longer for the 70B-class models). This script polls and prints
status; Ctrl-C is safe, the endpoint keeps creating in the background.
Check on it separately with:
    aws sagemaker describe-endpoint --endpoint-name <name> --region <region>

Remember: every second this endpoint exists, it bills. Run teardown.py when
you're done for the session.
"""
from __future__ import annotations

import argparse
import sys
import time

import boto3
from botocore.exceptions import ClientError
from sagemaker import image_uris

from config import AWS_REGION, HF_TOKEN, ModelConfig, default_model_key, get_model, require_role_arn
from pricing import format_cost_estimate

# The djl-lmi container bundles DJL Serving + vLLM (among other engines).
# Pinned to a version confirmed available in this repo's pinned `sagemaker`
# SDK (requirements.txt). To bump: pip install a newer `sagemaker`, then
#   python -c "from sagemaker import image_uris; print(image_uris.retrieve('djl-lmi', region='us-east-1', version='<new-version>'))"
# and update LMI_VERSION below once that resolves.
LMI_VERSION = "0.28.0"

DEFAULT_MAX_ROLLING_BATCH_SIZE = "8"  # max concurrent requests vLLM will batch together
CONTAINER_STARTUP_HEALTH_CHECK_TIMEOUT_S = 900  # generous: weight download can be slow


def resolve_image_uri(region: str) -> str:
    try:
        return image_uris.retrieve(framework="djl-lmi", region=region, version=LMI_VERSION)
    except Exception as e:
        raise RuntimeError(
            "Could not resolve the DJL LMI container image URI via the sagemaker SDK "
            f"(framework='djl-lmi', version={LMI_VERSION!r}, region={region!r}).\n"
            f"Underlying error: {e}\n\n"
            "This usually means LMI_VERSION in deploy.py is stale for your installed "
            "`sagemaker` package, or that version isn't published in this region. "
            "List what your installed SDK knows about with:\n"
            "  python -c \"import json,os,sagemaker; "
            "d=os.path.join(os.path.dirname(sagemaker.__file__),'image_uri_config'); "
            "print(json.load(open(os.path.join(d,'djl-lmi.json')))['versions'].keys())\"\n"
            "or see https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-container-docs.html"
        ) from e


def build_environment(model: ModelConfig) -> dict:
    """Translate a ModelConfig into DJL/LMI container environment variables."""
    if model.requires_hf_token and not HF_TOKEN:
        raise RuntimeError(
            f"Model '{model.key}' is gated on Hugging Face (requires_hf_token: true in "
            "models.yaml) but HF_TOKEN is not set. Accept the model's license on its HF "
            "model page, generate a token with read access, and set HF_TOKEN in .env."
        )

    env = {
        "HF_MODEL_ID": model.hf_model_id,
        "OPTION_ENGINE": "vLLM",
        "OPTION_ROLLING_BATCH": "vllm",  # belt-and-suspenders: selects the vLLM rolling-batch path explicitly
        "OPTION_TENSOR_PARALLEL_DEGREE": str(model.tensor_parallel_degree),
        "OPTION_MAX_MODEL_LEN": str(model.max_model_len),
        "OPTION_MAX_ROLLING_BATCH_SIZE": DEFAULT_MAX_ROLLING_BATCH_SIZE,
    }
    if model.quantize:
        env["OPTION_QUANTIZE"] = model.quantize
    if model.dtype:
        env["OPTION_DTYPE"] = model.dtype
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
    return env


def _resource_exists(describe_call) -> bool:
    try:
        describe_call()
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = e.response.get("Error", {}).get("Message", "")
        if code in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException") and (
            "Could not find" in msg or "does not exist" in msg
        ):
            return False
        raise


def check_no_existing_resources(sm_client, model: ModelConfig) -> list[str]:
    """Return names of any of the three resources that already exist and
    would block a fresh deploy."""
    existing = []
    if _resource_exists(lambda: sm_client.describe_model(ModelName=model.model_name)):
        existing.append(f"model '{model.model_name}'")
    if _resource_exists(
        lambda: sm_client.describe_endpoint_config(EndpointConfigName=model.endpoint_config_name)
    ):
        existing.append(f"endpoint config '{model.endpoint_config_name}'")
    if _resource_exists(lambda: sm_client.describe_endpoint(EndpointName=model.endpoint_name)):
        existing.append(f"endpoint '{model.endpoint_name}'")
    return existing


def deploy(sm_client, model: ModelConfig, image_uri: str, role_arn: str) -> dict:
    """Create Model -> EndpointConfig -> Endpoint, in that order. Caller
    must have already checked check_no_existing_resources()."""
    env = build_environment(model)

    sm_client.create_model(
        ModelName=model.model_name,
        PrimaryContainer={"Image": image_uri, "Environment": env},
        ExecutionRoleArn=role_arn,
    )

    sm_client.create_endpoint_config(
        EndpointConfigName=model.endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model.model_name,
                "InstanceType": model.instance_type,
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": CONTAINER_STARTUP_HEALTH_CHECK_TIMEOUT_S,
            }
        ],
    )

    return sm_client.create_endpoint(
        EndpointName=model.endpoint_name,
        EndpointConfigName=model.endpoint_config_name,
    )


def wait_for_in_service(sm_client, endpoint_name: str, poll_s: int = 20, timeout_s: int = 1800) -> str:
    """Poll describe_endpoint, printing status changes, until InService,
    Failed, or timeout."""
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        if status != last_status:
            print(f"  [{time.strftime('%H:%M:%S')}] endpoint status: {status}")
            last_status = status
        if status == "InService":
            return status
        if status == "Failed":
            reason = desc.get("FailureReason", "unknown")
            raise RuntimeError(f"Endpoint creation failed: {reason}")
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for '{endpoint_name}' to reach InService")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", help="Model key from models.yaml (default: the entry marked `default: true`)"
    )
    parser.add_argument("--region", default=AWS_REGION)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return immediately after create_endpoint instead of polling until InService",
    )
    args = parser.parse_args(argv)

    model_key = args.model or default_model_key()
    if not model_key:
        print("Error: no --model given and no model in models.yaml is marked `default: true`.")
        return 1

    try:
        model = get_model(model_key)
        role_arn = require_role_arn()
    except (KeyError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    print(f"Model:          {model.key}  ({model.hf_model_id})")
    print(f"Instance type:  {model.instance_type}")
    print(f"Region:         {args.region}")
    print(format_cost_estimate(model.instance_type, args.region))
    if model.note:
        print(f"Note: {model.note}")

    sm_client = boto3.client("sagemaker", region_name=args.region)

    existing = check_no_existing_resources(sm_client, model)
    if existing:
        print("\nError: the following resources already exist:")
        for name in existing:
            print(f"  - {name}")
        print(
            f"\nRun 'python teardown.py --model {model.key} --confirm' first, "
            "or deploy under a different model key."
        )
        return 1

    print("\nResolving DJL/LMI container image URI...")
    try:
        image_uri = resolve_image_uri(args.region)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    print(f"  {image_uri}")

    print("\nCreating model, endpoint config, and endpoint...")
    try:
        deploy(sm_client, model, image_uri, role_arn)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    print(f"  model:           {model.model_name}")
    print(f"  endpoint config: {model.endpoint_config_name}")
    print(f"  endpoint:        {model.endpoint_name}")

    if args.no_wait:
        print("\n--no-wait passed; not polling for InService. Check status with:")
        print(f"  aws sagemaker describe-endpoint --endpoint-name {model.endpoint_name} --region {args.region}")
        return 0

    print(
        "\nWaiting for endpoint to reach InService (this can take 5-15+ minutes "
        "for a fresh weight download)..."
    )
    try:
        wait_for_in_service(sm_client, model.endpoint_name)
    except (RuntimeError, TimeoutError) as e:
        print(f"\nError: {e}")
        print(f"Remember to run 'python teardown.py --model {model.key} --confirm' to stop billing.")
        return 1

    print(f"\n✅ Endpoint '{model.endpoint_name}' is InService.")
    print(f"Test it directly with:        python scripts/test_endpoint_direct.py --model {model.key}")
    print(f"Tear it down when done with:  python teardown.py --model {model.key} --confirm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
