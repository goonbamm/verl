"""Custom reward function for DeepMath-style datasets.

This module allows using existing `math_reward` logic without modifying
core verl reward routing.
"""

from __future__ import annotations

from verl.utils.reward_score.math_reward import compute_score as base_math_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Compute DeepMath reward using verl's built-in math reward.

    Args:
        data_source: Dataset identifier from parquet (kept for compatibility).
        solution_str: Model response string.
        ground_truth: Target final answer string.
        extra_info: Optional sample metadata.
        **kwargs: Forward-compatible extra kwargs from reward manager.

    Returns:
        float: 1.0 when parsed final boxed answer matches ground truth else 0.0.
    """
    del data_source, extra_info, kwargs
    return float(base_math_reward(solution_str=solution_str, ground_truth=ground_truth))
