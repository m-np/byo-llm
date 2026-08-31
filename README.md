# byo-llm

Deploy an open-source LLM to AWS SageMaker behind an OpenAI-compatible
`/v1/chat/completions` endpoint, so any app using the OpenAI SDK can point
`base_url` at it with no other code changes.

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<api-id>.execute-api.<region>.amazonaws.com/v1",
    api_key="unused",  # adapter doesn't check it -- see "Security" below before exposing this
)
resp = client.chat.completions.create(
    model="qwen2.5-14b-awq",
    messages=[{"role": "user", "content": "hi"}],
)
```

Built as a **1-week, cost-conscious experiment**, not production infra. That
shapes several decisions below -- fixed single-instance endpoints (no
autoscaling), plain boto3 for the Lambda/API Gateway wiring instead of a
full CDK stack, and `teardown.py` was built and tested *before* anything
else, because a SageMaker real-time endpoint has no "stopped" state -- it
bills per second from `InService` until you delete it.

> **The one command to remember:**
> ```
> python teardown.py --model qwen2.5-14b-awq --confirm
> ```
> Run it at the end of every session. See "Cost notes" below.

## How it fits together

```
 models.yaml ──► deploy.py ──► SageMaker endpoint (DJL/LMI container, vLLM backend)
                                        ▲
                                        │ sagemaker-runtime InvokeEndpoint
                                        │
 OpenAI SDK ──► API Gateway (HTTP API) ─► Lambda (lambda/handler.py) ──┘
   base_url        POST /v1/chat/completions   translates OpenAI <-> LMI JSON

 infra/setup_lambda_api.py wires the Lambda + API Gateway (plain boto3, not CDK).
 teardown.py deletes the SageMaker endpoint/config/model. Run it when you're done.
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # requirements.txt + pytest

cp .env.example .env
# edit .env: set AWS_REGION and SAGEMAKER_ROLE_ARN (see below)
```

Run the unit tests (no AWS needed -- everything's mocked):

```bash
pytest tests/ -v
```

### IAM role setup

`SAGEMAKER_ROLE_ARN` in `.env` is the role SageMaker itself assumes to pull
the container image and download model weights -- it is **not** your own
AWS credentials, and this repo does not create it for you (see "Ambiguity
this repo resolved" below for why).

Trust policy (who can assume the role):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Quickest path: attach the AWS-managed `AmazonSageMakerFullAccess` policy.
Tighter, minimal policy if you'd rather scope it down:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"], "Resource": "*" },
    { "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*" },
    { "Effect": "Allow", "Action": ["cloudwatch:PutMetricData"], "Resource": "*" }
  ]
}
```
(No S3 permissions needed for the default setup -- the container downloads
weights directly from Hugging Face, not from S3.)

Create it via the console (IAM -> Roles -> Create role -> Custom trust
policy) or CLI:

```bash
aws iam create-role --role-name sagemaker-llm-execution-role \
  --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name sagemaker-llm-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

> **If you're using root account credentials** (check with
> `aws sts get-caller-identity` -- the ARN will end in `:root`): AWS
> strongly discourages this for day-to-day API calls. Root credentials
> can't be scoped down, can't be individually revoked/rotated like an IAM
> user's, and a leak means full account compromise. Create an IAM user (or
> role, if you're on SSO) with the specific permissions this repo needs
> (`sagemaker:*`, `lambda:*`, `apigateway:*`, `iam:*Role*` for
> `infra/setup_lambda_api.py`) instead, and use that day to day.

## Usage

### 1. Deploy a model

```bash
python deploy.py --model qwen2.5-14b-awq   # or omit --model to use the `default: true` entry
```

Prints the estimated hourly cost before doing anything, then creates a
SageMaker Model, EndpointConfig, and Endpoint, and polls until `InService`
(5-15+ minutes -- container pull + weight download + vLLM init). Refuses to
run if any of the three resources already exist under that model's derived
names; run `teardown.py` first if you're redeploying.

### 2. Test it directly (bypasses Lambda/API Gateway)

```bash
python scripts/test_endpoint_direct.py --model qwen2.5-14b-awq
```

### 3. Wire up the OpenAI-compatible adapter

```bash
python infra/setup_lambda_api.py --model qwen2.5-14b-awq
```

Creates a narrowly-scoped IAM role (only `sagemaker:InvokeEndpoint` on
*this* endpoint's ARN, plus its own CloudWatch Logs group), the Lambda
(`lambda/handler.py`), and an HTTP API route `POST /v1/chat/completions`.
Prints the URL to use as your OpenAI SDK `base_url`. Safe to re-run --
updates resources in place rather than duplicating them.

### 4. Test the full path

```bash
python scripts/test_endpoint_via_api.py --url https://<api-id>.execute-api.<region>.amazonaws.com/v1/chat/completions
```

If step 2 works but this doesn't, the bug is in `lambda/handler.py`'s
translation logic or the API Gateway wiring -- not the model.

### 5. Tear down

```bash
python teardown.py --model qwen2.5-14b-awq --confirm
```

Deletes the SageMaker endpoint, then its config, then the model, in that
order, then re-checks `list_endpoints` to confirm it's actually gone.
Without `--confirm` it only prints what it *would* delete. With
`--confirm`, you'll additionally be asked to type the endpoint name back
before anything is deleted. (`infra/setup_lambda_api.py`'s Lambda/API
Gateway resources aren't deleted by this -- they don't meaningfully bill
on their own; see cost notes.)

## Picking a model

| Key | HF model | Instance | VRAM fit | Gated? | When to use |
|---|---|---|---|---|---|
| `qwen2.5-14b-awq` (default) | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `ml.g5.2xlarge` | 4-bit AWQ, ~8GB weights on a 24GB A10G, comfortable headroom for KV cache | No | Best quality-per-dollar single-GPU option here. Default for this experiment. |
| `mistral-7b-instruct` | `mistralai/Mistral-7B-Instruct-v0.3` | `ml.g5.2xlarge` | bf16, ~15GB weights on 24GB A10G | No | Cheapest capable baseline; largest available context window (32k) of the three; good pipeline sanity-check model. |
| `llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | `ml.g5.2xlarge` | bf16, ~16GB weights on 24GB A10G | **Yes** -- accept Meta's license on the HF model page and set `HF_TOKEN` in `.env` | Quality comparison against Qwen/Mistral; skip if you don't want to deal with HF gating. |

Add more models by adding an entry to `models.yaml` -- no code changes
needed. See the comments at the top of that file for what each field means.

## Cost notes

Estimates only, `us-east-1`, on-demand, recorded 2026-08 (see `pricing.py`
-- verify against https://aws.amazon.com/sagemaker/pricing/, this drifts):

| Instance | $/hr | GPU | VRAM | Notes |
|---|---|---|---|---|
| `ml.g4dn.xlarge` | $0.736 | 1x T4 | 16GB | Cheapest GPU option; too small for the 7B+ models in this repo at reasonable context length |
| `ml.g5.xlarge` | $1.408 | 1x A10G | 24GB | Slightly less vCPU/memory headroom than g5.2xlarge for the same GPU |
| `ml.g5.2xlarge` | $1.515 | 1x A10G | 24GB | **What every model in `models.yaml` uses by default** |
| `ml.g5.4xlarge` | $2.030 | 1x A10G | 24GB | More vCPU/RAM, same GPU -- rarely worth it for inference alone |
| `ml.g5.12xlarge` | $7.090 | 4x A10G | 24GB each | For 70B-class models needing `tensor_parallel_degree: 4` |
| `ml.g5.48xlarge` | $20.360 | 8x A10G | 24GB each | Largest g5; 70B+ at higher throughput |
| `ml.p4d.24xlarge` | $37.688 | 8x A100 | 40GB each | Overkill for anything in this repo's default configs |

At the default `ml.g5.2xlarge`, an endpoint left running costs roughly
**$36/day** or **$255/week**. `deploy.py` prints this estimate every time
it runs so it's never a surprise. This is exactly why `teardown.py` exists
and was built first.

## Streaming

v1 is **non-streaming only** (`lambda/handler.py` returns a clean 400 if
you send `stream: true`). Classic API Gateway (including the HTTP API used
here) buffers the entire Lambda response -- there's no way to forward
tokens as they're generated through that path no matter what the Lambda
does.

`lambda/streaming_handler.py` documents the real path (Lambda Function URL
with `InvokeMode=RESPONSE_STREAM` + `sagemaker-runtime
InvokeEndpointWithResponseStream`) and implements + unit-tests the one
piece that isn't AWS-runtime-version-dependent: translating a single
streamed LMI chunk into an OpenAI `chat.completion.chunk` SSE line. The
Lambda response-streaming plumbing itself is sketched but **not wired up or
deployed** -- see that file's module docstring for exactly what's stubbed
and why, before building on it.

## Repo structure

```
models.yaml              Model registry -- add a model here, no code changes needed
config.py                Shared .env + models.yaml loading
pricing.py                Approximate per-instance-type hourly cost estimates
deploy.py                 Creates Model + EndpointConfig + Endpoint from models.yaml
teardown.py                Deletes them, in order, with confirmation + verification
lambda/
  handler.py                OpenAI <-> LMI translation + Lambda entrypoint (non-streaming)
  streaming_handler.py       Streaming architecture note/stub (not deployed)
infra/
  setup_lambda_api.py        Plain-boto3 Lambda + HTTP API Gateway setup (no CDK)
scripts/
  test_endpoint_direct.py    Hit SageMaker directly, bypassing the adapter
  test_endpoint_via_api.py   Hit the full API Gateway -> Lambda -> SageMaker path
tests/
  test_teardown.py, test_deploy.py, test_handler.py, test_streaming_handler.py
  -- all mock boto3, no AWS credentials needed to run
```

## Ambiguity this repo resolved

Per the original request, these were flagged and confirmed rather than
guessed at:

- **Region**: defaults to `us-east-1`.
- **IAM role**: `deploy.py` expects an existing role ARN via
  `SAGEMAKER_ROLE_ARN` in `.env` (see "IAM role setup" above), rather than
  creating one itself -- keeps blast radius of what this repo can touch in
  your AWS account smaller for a short-lived experiment.
- **Default model**: `qwen2.5-14b-awq`, ungated on Hugging Face, best
  quality-per-dollar single-GPU option of the three configured.

## Growing past a 1-week experiment

Things deliberately left out of v1 that a longer-lived deployment would
want:

- **Autoscaling** (`sagemaker.client.register_scalable_target` +
  `put_scaling_policy` with `SageMakerVariantInvocationsPerInstance`
  target tracking) -- fixed single-instance for now.
- **CDK** for the Lambda/API Gateway layer, instead of
  `infra/setup_lambda_api.py`'s plain boto3 calls -- worth it once this
  needs to be reproducible across environments or reviewed as a diff.
- **CloudWatch alarms** on endpoint latency/error rate.
- **Auth** on the API Gateway route -- right now anything with the URL can
  call it; `api_key="unused"` in the OpenAI SDK example above is literal.
  Add an API Gateway authorizer (API key, JWT, IAM) before exposing this
  beyond your own machine.
- **Streaming**, properly wired (see above).
