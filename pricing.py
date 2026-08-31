"""
Approximate SageMaker real-time inference on-demand pricing, USD/hour.

These numbers are a point-in-time snapshot (recorded 2026-08) for us-east-1
only and WILL drift — AWS changes SageMaker pricing periodically and it
varies by region. Treat this as a rough sanity check, not a bill forecast.
Verify current numbers at https://aws.amazon.com/sagemaker/pricing/ before
relying on it for a real budget decision.
"""
from __future__ import annotations

from typing import NamedTuple, Optional


class InstancePricing(NamedTuple):
    hourly_usd: float
    gpus: int
    gpu_type: str
    vram_gb_per_gpu: int


# instance_type -> pricing (us-east-1, on-demand, real-time inference)
INSTANCE_PRICING_US_EAST_1: dict[str, InstancePricing] = {
    "ml.g4dn.xlarge":  InstancePricing(0.736, 1, "T4", 16),
    "ml.g4dn.2xlarge": InstancePricing(1.052, 1, "T4", 16),
    "ml.g5.xlarge":    InstancePricing(1.408, 1, "A10G", 24),
    "ml.g5.2xlarge":   InstancePricing(1.515, 1, "A10G", 24),
    "ml.g5.4xlarge":   InstancePricing(2.030, 1, "A10G", 24),
    "ml.g5.12xlarge":  InstancePricing(7.090, 4, "A10G", 24),
    "ml.g5.48xlarge":  InstancePricing(20.360, 8, "A10G", 24),
    "ml.p4d.24xlarge": InstancePricing(37.688, 8, "A100", 40),
}


def estimate_hourly_cost(instance_type: str, region: str = "us-east-1") -> Optional[float]:
    """Best-effort hourly cost estimate. Returns None if unknown (unpriced
    instance type, or a region we don't have numbers pinned for)."""
    if region != "us-east-1":
        return None
    entry = INSTANCE_PRICING_US_EAST_1.get(instance_type)
    return entry.hourly_usd if entry else None


def format_cost_estimate(instance_type: str, region: str = "us-east-1") -> str:
    """Human-readable line for deploy.py to print on every run."""
    cost = estimate_hourly_cost(instance_type, region)
    if cost is None:
        return (
            f"⚠️  No pinned price for {instance_type} in {region} — check "
            f"https://aws.amazon.com/sagemaker/pricing/ manually before deploying."
        )
    daily = cost * 24
    week = daily * 7
    return (
        f"💵 Estimated cost for {instance_type} in {region}: "
        f"${cost:.3f}/hr  (~${daily:.2f}/day, ~${week:.2f}/week if left running). "
        f"This bills continuously until you run teardown.py — see README."
    )
