"""Custom reward function for OpenR1-Math dataset.

This reward intentionally reuses verl's built-in `math_reward` because OpenR1
samples are math QA entries with boxed final-answer conventions.
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
    """Compute OpenR1-Math reward using verl's built-in math reward.

    Args:
        data_source: Dataset identifier from parquet (kept for compatibility).
        solution_str: Model response string.
        ground_truth: Target final answer string.
        extra_info: Optional sample metadata.
        **kwargs: Forward-compatible extra kwargs from reward manager.

    Returns:
        float: Reward score from `math_reward` (0.0 or 1.0).
    """
    del data_source, extra_info, kwargs
    return float(base_math_reward(solution_str=solution_str, ground_truth=ground_truth))
