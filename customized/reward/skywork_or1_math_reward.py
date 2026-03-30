"""Custom reward function for Skywork-OR1-Math dataset.

Based on local benchmark runs on existing 100-sample math benchmark sets,
`math_dapo` with `strict_box_verify=True` matched the top accuracy while being
the fastest implementation among top-scoring candidates.
"""

from __future__ import annotations

from verl.utils.reward_score.math_dapo import compute_score as math_dapo_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Compute Skywork-OR1-Math reward using strict boxed-answer matching.

    Args:
        data_source: Dataset identifier from parquet (kept for compatibility).
        solution_str: Model response string.
        ground_truth: Target final answer string.
        extra_info: Optional sample metadata.
        **kwargs: Forward-compatible extra kwargs from reward manager.

    Returns:
        float: Reward score from `math_dapo` strict box verification.
    """
    del data_source, extra_info, kwargs
    return float(math_dapo_reward(solution_str=solution_str, ground_truth=ground_truth, strict_box_verify=True))
