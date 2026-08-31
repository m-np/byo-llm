"""
Shared config loading for deploy.py / teardown.py / autoscale.py / infra scripts.

Reads AWS/region/role settings from the environment (populated from .env via
python-dotenv if present) and model definitions from models.yaml.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional at runtime if env vars are already exported
    # (e.g. in CI or a shell profile) — see requirements.txt.
    pass

REPO_ROOT = Path(__file__).resolve().parent
MODELS_FILE = REPO_ROOT / "models.yaml"

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")
HF_TOKEN = os.environ.get("HF_TOKEN")

MAX_SAGEMAKER_NAME_LEN = 63  # hard limit enforced by the SageMaker API


def _truncate_name(name: str) -> str:
    if len(name) <= MAX_SAGEMAKER_NAME_LEN:
        return name
    return name[:MAX_SAGEMAKER_NAME_LEN]


def _sanitize_for_sagemaker_name(key: str) -> str:
    """SageMaker resource names must match [a-zA-Z0-9]([-a-zA-Z0-9]*[a-zA-Z0-9])?
    -- notably no '.'. Model keys in models.yaml (e.g. "qwen2.5-14b-awq",
    "llama-3.1-8b-instruct") are chosen for readability, not for that regex,
    so sanitize when deriving resource names. The yaml key itself (used for
    --model lookups) is untouched by this."""
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", key)
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized or "model"


@dataclass
class ModelConfig:
    key: str
    hf_model_id: str
    instance_type: str
    tensor_parallel_degree: int
    max_model_len: int
    quantize: Optional[str] = None
    dtype: Optional[str] = None
    requires_hf_token: bool = False
    note: str = ""

    @property
    def _sanitized_key(self) -> str:
        return _sanitize_for_sagemaker_name(self.key)

    @property
    def endpoint_name(self) -> str:
        return _truncate_name(f"{self._sanitized_key}-endpoint")

    @property
    def model_name(self) -> str:
        return _truncate_name(f"{self._sanitized_key}-model")

    @property
    def endpoint_config_name(self) -> str:
        return _truncate_name(f"{self._sanitized_key}-endpoint-config")


def load_models(path: Path = MODELS_FILE) -> dict[str, ModelConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "models" not in raw:
        raise ValueError(f"{path} has no top-level 'models:' key")

    models: dict[str, ModelConfig] = {}
    for key, data in raw["models"].items():
        try:
            models[key] = ModelConfig(
                key=key,
                hf_model_id=data["hf_model_id"],
                instance_type=data["instance_type"],
                tensor_parallel_degree=int(data.get("tensor_parallel_degree", 1)),
                max_model_len=int(data.get("max_model_len", 4096)),
                quantize=data.get("quantize"),
                dtype=data.get("dtype"),
                requires_hf_token=bool(data.get("requires_hf_token", False)),
                note=(data.get("note") or "").strip(),
            )
        except KeyError as e:
            raise ValueError(f"models.yaml entry '{key}' is missing required field {e}") from e
    return models


def get_model(key: str, path: Path = MODELS_FILE) -> ModelConfig:
    models = load_models(path)
    if key not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"Unknown model '{key}'. Available: {available}")
    return models[key]


def default_model_key(path: Path = MODELS_FILE) -> Optional[str]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for key, data in (raw or {}).get("models", {}).items():
        if data.get("default"):
            return key
    return None


def require_role_arn() -> str:
    if not SAGEMAKER_ROLE_ARN:
        raise RuntimeError(
            "SAGEMAKER_ROLE_ARN is not set. Copy .env.example to .env and set it to "
            "an IAM role ARN with a SageMaker trust policy (see README 'IAM role setup')."
        )
    return SAGEMAKER_ROLE_ARN
