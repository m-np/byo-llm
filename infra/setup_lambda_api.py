#!/usr/bin/env python3
"""
Minimal, plain-boto3 infra setup for the OpenAI-compatible adapter: an IAM
role scoped to one SageMaker endpoint, the Lambda function (lambda/handler.py),
and an HTTP API (API Gateway v2) route `POST /v1/chat/completions` in front
of it.

Deliberately NOT CDK -- this is a 1-week experiment; a few idempotent boto3
calls are faster to iterate on than a CDK stack, and model swaps
(deploy.py / teardown.py) are meant to be independent of this infra layer
anyway. If this repo outlives the experiment, promoting this script to a
CDK app is a reasonable next step (see README "Growing past a 1-week
experiment").

Usage:
    python infra/setup_lambda_api.py --model qwen2.5-14b-awq

The SageMaker endpoint for --model must already exist (run deploy.py
first) -- the IAM policy this script writes is scoped to that specific
endpoint's ARN, so it needs to know the name up front.

Safe to re-run: the IAM role / Lambda function / HTTP API are updated in
place rather than duplicated, keyed off fixed names. This script does not
delete anything; there is no infra teardown counterpart yet (a 1-week
experiment's Lambda + HTTP API don't meaningfully bill on their own, unlike
the SageMaker endpoint -- see README cost notes).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AWS_REGION, get_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
HANDLER_FILE = REPO_ROOT / "lambda" / "handler.py"

FUNCTION_NAME = "openai-adapter-chat-completions"
ROLE_NAME = "openai-adapter-lambda-role"
API_NAME = "openai-adapter-http-api"
ROUTE = "POST /v1/chat/completions"
ROUTE_PATH = ROUTE.split(" ", 1)[1]

LAMBDA_RUNTIME = "python3.12"
LAMBDA_HANDLER = "handler.lambda_handler"
LAMBDA_TIMEOUT_S = 60  # generous: a 14B model generating max_tokens can take a while
LAMBDA_MEMORY_MB = 256


def _account_id(session: boto3.Session) -> str:
    return session.client("sts").get_caller_identity()["Account"]


def _zip_handler() -> bytes:
    """Package lambda/handler.py alone -- it only imports boto3, which is
    preinstalled in the Lambda Python runtime, so no dependency layer is
    needed."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(HANDLER_FILE, arcname="handler.py")
    return buf.getvalue()


def create_or_update_role(iam_client, account_id: str, region: str, endpoint_name: str) -> str:
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
        ],
    }
    endpoint_arn = f"arn:aws:sagemaker:{region}:{account_id}:endpoint/{endpoint_name}"
    log_group_arn = f"arn:aws:logs:{region}:{account_id}:log-group:/aws/lambda/{FUNCTION_NAME}:*"

    permissions_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "sagemaker:InvokeEndpoint", "Resource": endpoint_arn},
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": log_group_arn,
            },
        ],
    }

    freshly_created = False
    try:
        iam_client.get_role(RoleName=ROLE_NAME)
        print(f"IAM role '{ROLE_NAME}' already exists, updating trust/permissions policy...")
        iam_client.update_assume_role_policy(RoleName=ROLE_NAME, PolicyDocument=json.dumps(trust_policy))
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        print(f"Creating IAM role '{ROLE_NAME}'...")
        iam_client.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust_policy))
        freshly_created = True

    iam_client.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="openai-adapter-scoped-permissions",
        PolicyDocument=json.dumps(permissions_policy),
    )

    role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    print(f"  role ARN: {role_arn}")
    print(f"  scoped to invoke: {endpoint_arn}")

    if freshly_created:
        # Newly created IAM roles can take a few seconds to propagate; a
        # create_function call too soon after fails with "role cannot be
        # assumed". _create_with_retry() below also retries as a backstop.
        print("  waiting ~8s for IAM role propagation...")
        time.sleep(8)

    return role_arn


def _wait_for_update_complete(lambda_client, timeout_s: int = 60, poll_s: int = 2) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = lambda_client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]
        if state.get("LastUpdateStatus", "Successful") != "InProgress":
            return
        time.sleep(poll_s)


def _create_with_retry(lambda_client, role_arn: str, env: dict, zip_bytes: bytes, attempts: int = 6, delay_s: int = 5):
    last_err = None
    for attempt in range(attempts):
        try:
            lambda_client.create_function(
                FunctionName=FUNCTION_NAME,
                Runtime=LAMBDA_RUNTIME,
                Role=role_arn,
                Handler=LAMBDA_HANDLER,
                Code={"ZipFile": zip_bytes},
                Timeout=LAMBDA_TIMEOUT_S,
                MemorySize=LAMBDA_MEMORY_MB,
                Environment=env,
                Description="OpenAI chat.completions <-> SageMaker adapter (byo-llm repo)",
            )
            return
        except ClientError as e:
            if e.response["Error"]["Code"] == "InvalidParameterValueException" and attempt < attempts - 1:
                print(f"  role not yet assumable, retrying in {delay_s}s ({attempt + 1}/{attempts})...")
                last_err = e
                time.sleep(delay_s)
                continue
            raise
    raise last_err


def create_or_update_function(lambda_client, role_arn: str, endpoint_name: str, region: str) -> str:
    zip_bytes = _zip_handler()
    env = {"Variables": {"SAGEMAKER_ENDPOINT_NAME": endpoint_name, "AWS_REGION": region}}

    try:
        lambda_client.get_function(FunctionName=FUNCTION_NAME)
        print(f"Lambda function '{FUNCTION_NAME}' already exists, updating code + config...")
        lambda_client.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=zip_bytes)
        _wait_for_update_complete(lambda_client)
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Role=role_arn,
            Environment=env,
            Timeout=LAMBDA_TIMEOUT_S,
            MemorySize=LAMBDA_MEMORY_MB,
        )
        _wait_for_update_complete(lambda_client)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        print(f"Creating Lambda function '{FUNCTION_NAME}'...")
        _create_with_retry(lambda_client, role_arn, env, zip_bytes)

    function_arn = lambda_client.get_function(FunctionName=FUNCTION_NAME)["Configuration"]["FunctionArn"]
    print(f"  function ARN: {function_arn}")
    return function_arn


def create_or_update_http_api(apigw_client, lambda_client, function_arn: str, region: str, account_id: str) -> str:
    api = next((a for a in apigw_client.get_apis()["Items"] if a["Name"] == API_NAME), None)
    if api:
        api_id = api["ApiId"]
        print(f"HTTP API '{API_NAME}' already exists (id={api_id})")
    else:
        print(f"Creating HTTP API '{API_NAME}'...")
        api_id = apigw_client.create_api(Name=API_NAME, ProtocolType="HTTP")["ApiId"]

    integration = next(
        (i for i in apigw_client.get_integrations(ApiId=api_id)["Items"] if i.get("IntegrationUri") == function_arn),
        None,
    )
    if integration:
        integration_id = integration["IntegrationId"]
    else:
        integration_id = apigw_client.create_integration(
            ApiId=api_id,
            IntegrationType="AWS_PROXY",
            IntegrationUri=function_arn,
            PayloadFormatVersion="2.0",
            IntegrationMethod="POST",
        )["IntegrationId"]

    target = f"integrations/{integration_id}"
    existing_route = next((r for r in apigw_client.get_routes(ApiId=api_id)["Items"] if r["RouteKey"] == ROUTE), None)
    if existing_route:
        apigw_client.update_route(ApiId=api_id, RouteId=existing_route["RouteId"], Target=target)
    else:
        apigw_client.create_route(ApiId=api_id, RouteKey=ROUTE, Target=target)

    if not any(s["StageName"] == "$default" for s in apigw_client.get_stages(ApiId=api_id)["Items"]):
        apigw_client.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True)

    source_arn = f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*{ROUTE_PATH}"
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="AllowInvokeFromHttpApi",
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=source_arn,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceConflictException":  # already granted
            raise

    return apigw_client.get_api(ApiId=api_id)["ApiEndpoint"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", required=True, help="Model key from models.yaml -- must already be deployed (see deploy.py)"
    )
    parser.add_argument("--region", default=AWS_REGION)
    args = parser.parse_args(argv)

    model = get_model(args.model)
    session = boto3.Session(region_name=args.region)
    account_id = _account_id(session)

    sm_client = session.client("sagemaker")
    try:
        sm_client.describe_endpoint(EndpointName=model.endpoint_name)
    except ClientError:
        print(f"Error: endpoint '{model.endpoint_name}' does not exist. Run deploy.py --model {args.model} first.")
        return 1

    iam_client = session.client("iam")
    lambda_client = session.client("lambda")
    apigw_client = session.client("apigatewayv2")

    role_arn = create_or_update_role(iam_client, account_id, args.region, model.endpoint_name)
    function_arn = create_or_update_function(lambda_client, role_arn, model.endpoint_name, args.region)
    api_endpoint = create_or_update_http_api(apigw_client, lambda_client, function_arn, args.region, account_id)

    print("\n✅ Done.")
    print(f"OpenAI SDK base_url:  {api_endpoint}/v1")
    print(f"Full route:           {api_endpoint}{ROUTE_PATH}")
    print(f"\nTest it with: python scripts/test_endpoint_via_api.py --url {api_endpoint}{ROUTE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
