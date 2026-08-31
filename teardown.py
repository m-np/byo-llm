#!/usr/bin/env python3
"""
Delete a SageMaker endpoint, its endpoint config, and the underlying model.

    RUN THIS AT THE END OF EVERY SESSION.
    SageMaker real-time endpoints bill per second for as long as they exist
    -- there is no "stopped" state, only "deleted" and "still billing".

Usage:
    # Using a model key from models.yaml (recommended — derives all three
    # resource names the same way deploy.py created them):
    python teardown.py --model qwen2.5-14b-awq --confirm

    # Or an explicit endpoint name (e.g. if you deployed by hand):
    python teardown.py --endpoint-name my-endpoint --confirm

Without --confirm, this only PRINTS what it would delete and exits — nothing
is touched. With --confirm, you'll additionally be asked to type the exact
endpoint name back before anything is deleted.

Deletion order is: endpoint -> endpoint config -> model. After deleting the
endpoint, this script polls describe_endpoint until it 404s (or times out),
then deletes the config and model, then does a final list_endpoints check
and reports whether the endpoint is actually gone.
"""
from __future__ import annotations

import argparse
import sys
import time

import boto3
from botocore.exceptions import ClientError

from config import AWS_REGION, get_model


def _not_found(err: ClientError) -> bool:
    """True if a ClientError means 'this resource is already gone', as
    opposed to a real failure (permissions, throttling, etc.) that should
    propagate."""
    code = err.response.get("Error", {}).get("Code", "")
    msg = err.response.get("Error", {}).get("Message", "")
    return code in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException") and (
        "Could not find" in msg or "does not exist" in msg or "not found" in msg.lower()
    )


def delete_endpoint(sm_client, endpoint_name: str) -> str:
    """Delete the endpoint if it exists. Returns 'deleted' or 'already-absent'."""
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        return "deleted"
    except ClientError as e:
        if _not_found(e):
            return "already-absent"
        raise


def delete_endpoint_config(sm_client, endpoint_config_name: str) -> str:
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        return "deleted"
    except ClientError as e:
        if _not_found(e):
            return "already-absent"
        raise


def delete_model(sm_client, model_name: str) -> str:
    try:
        sm_client.delete_model(ModelName=model_name)
        return "deleted"
    except ClientError as e:
        if _not_found(e):
            return "already-absent"
        raise


def wait_for_endpoint_gone(sm_client, endpoint_name: str, timeout_s: int = 180, poll_s: int = 5) -> bool:
    """Poll describe_endpoint until it 404s (endpoint fully deleted) or we
    time out. SageMaker endpoint deletion is asynchronous — the DELETE call
    returns immediately but the endpoint stays in 'Deleting' status for a
    bit. We wait so config/model deletion (and the final confirmation
    check) happen after the endpoint is truly gone, not mid-flight."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            sm_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            if _not_found(e):
                return True
            raise
        time.sleep(poll_s)
    return False


def confirm_gone(sm_client, endpoint_name: str) -> bool:
    """Belt-and-suspenders check per requirement: confirm deletion by
    checking list_endpoints afterward, independent of whatever
    delete_endpoint() itself reported."""
    paginator = sm_client.get_paginator("list_endpoints")
    for page in paginator.paginate(NameContains=endpoint_name):
        for ep in page.get("Endpoints", []):
            if ep["EndpointName"] == endpoint_name:
                return False
    return True


def run_teardown(
    sm_client,
    endpoint_name: str,
    endpoint_config_name: str,
    model_name: str,
    wait: bool = True,
) -> dict:
    """Delete endpoint -> endpoint config -> model, in that order, then
    confirm via list_endpoints. Pure function of (client, names) so it's
    testable with a fake/mock client, independent of real AWS."""
    result = {"endpoint_name": endpoint_name}

    result["endpoint"] = delete_endpoint(sm_client, endpoint_name)
    if wait and result["endpoint"] == "deleted":
        result["endpoint_wait_completed"] = wait_for_endpoint_gone(sm_client, endpoint_name)

    result["endpoint_config"] = delete_endpoint_config(sm_client, endpoint_config_name)
    result["model"] = delete_model(sm_client, model_name)
    result["confirmed_gone"] = confirm_gone(sm_client, endpoint_name)

    return result


def _resolve_names(args) -> tuple[str, str, str]:
    if args.model:
        mc = get_model(args.model)
        endpoint_name = args.endpoint_name or mc.endpoint_name
        endpoint_config_name = args.endpoint_config_name or mc.endpoint_config_name
        sm_model_name = args.model_name or mc.model_name
    else:
        endpoint_name = args.endpoint_name
        endpoint_config_name = args.endpoint_config_name or f"{endpoint_name}-config"
        sm_model_name = args.model_name or f"{endpoint_name.rsplit('-endpoint', 1)[0]}-model"
    return endpoint_name, endpoint_config_name, sm_model_name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model key from models.yaml (derives all resource names)")
    group.add_argument("--endpoint-name", help="Explicit SageMaker endpoint name")
    parser.add_argument(
        "--endpoint-config-name",
        help="Override the endpoint config name (default: derived from --model/--endpoint-name)",
    )
    parser.add_argument(
        "--model-name",
        help="Override the SageMaker model name (default: derived from --model/--endpoint-name)",
    )
    parser.add_argument("--region", default=AWS_REGION)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without this flag, nothing is deleted (dry run only).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for the endpoint to finish deleting before deleting its config/model.",
    )
    args = parser.parse_args(argv)

    try:
        endpoint_name, endpoint_config_name, sm_model_name = _resolve_names(args)
    except KeyError as e:
        print(f"Error: {e.args[0] if e.args else e}")
        return 1

    print("Will delete, in this order:")
    print(f"  1. endpoint:        {endpoint_name}")
    print(f"  2. endpoint config: {endpoint_config_name}")
    print(f"  3. model:           {sm_model_name}")
    print(f"  region:             {args.region}")

    if not args.confirm:
        print("\nDry run — no --confirm flag passed. Nothing was deleted.")
        print("Re-run with --confirm to actually delete these resources.")
        return 0

    typed = input(f"\nType the endpoint name ('{endpoint_name}') to confirm deletion: ")
    if typed.strip() != endpoint_name:
        print("Confirmation text did not match the endpoint name. Aborting — nothing deleted.")
        return 1

    sm_client = boto3.client("sagemaker", region_name=args.region)
    result = run_teardown(
        sm_client, endpoint_name, endpoint_config_name, sm_model_name, wait=not args.no_wait
    )

    print("\nResult:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result["confirmed_gone"]:
        print(
            f"\n✅ Confirmed via list_endpoints: '{endpoint_name}' no longer exists. "
            "Billing for it has stopped."
        )
        return 0
    else:
        print(
            f"\n⚠️  '{endpoint_name}' still appears in list_endpoints. It may still be "
            "billing. Check the AWS console, or re-run this script."
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
