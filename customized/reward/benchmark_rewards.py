"""Benchmark reward scoring implementations on a fixed sample dataset."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from customized.reward.deepmath_reward import compute_score as deepmath_reward
from verl.utils.reward_score.math_dapo import compute_score as math_dapo_reward
from verl.utils.reward_score.math_reward import compute_score as math_reward


@dataclass
class RewardResult:
    name: str
    mean_score: float
    pass_rate: float
    elapsed_sec: float
    per_sample_ms: float
    error_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark reward functions with the same sample set")
    parser.add_argument(
        "--sample_path",
        default="customized/data_preprocess/samples/deepmath_sample_100.jsonl",
        help="Path to sample JSONL",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of full repeated benchmark runs")
    parser.add_argument(
        "--result_markdown",
        default="customized/reward/benchmark_results.md",
        help="Markdown file to save benchmark table",
    )
    return parser.parse_args()


def maybe_load_math_verify() -> Callable[..., Any] | None:
    try:
        module = importlib.import_module("verl.utils.reward_score.math_verify")
        return getattr(module, "compute_score", None)
    except Exception:
        return None


def load_samples(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_reward(
    name: str,
    fn: Callable[[dict[str, Any]], Any],
    samples: list[dict[str, Any]],
    repeat: int,
) -> RewardResult:
    scores: list[float] = []
    error_count = 0

    start = time.perf_counter()
    for _ in range(repeat):
        for sample in samples:
            try:
                raw = fn(sample)
                if isinstance(raw, dict):
                    val = float(raw.get("score", 0.0))
                elif isinstance(raw, (list, tuple)):
                    val = float(raw[0])
                else:
                    val = float(raw)
            except Exception:
                val = 0.0
                error_count += 1
            scores.append(val)

    elapsed = time.perf_counter() - start
    total = len(scores)
    mean_score = sum(scores) / max(total, 1)
    pass_rate = sum(1 for x in scores if x >= 0.999) / max(total, 1)
    return RewardResult(
        name=name,
        mean_score=mean_score,
        pass_rate=pass_rate,
        elapsed_sec=elapsed,
        per_sample_ms=(elapsed * 1000.0 / max(total, 1)),
        error_count=error_count,
    )


def to_markdown(results: list[RewardResult], sample_count: int, repeat: int) -> str:
    lines = [
        "# Reward Benchmark Results",
        "",
        f"- sample_count: {sample_count}",
        f"- repeat: {repeat}",
        "",
        "| method | mean_score | pass_rate | elapsed_sec | per_sample_ms | errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.name} | {row.mean_score:.4f} | {row.pass_rate:.2%} | {row.elapsed_sec:.4f} | {row.per_sample_ms:.3f} | {row.error_count} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    samples = load_samples(args.sample_path)

    reward_methods: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
        (
            "deepmath_reward",
            lambda sample: deepmath_reward(
                sample.get("data_source", "deepmath-103k"),
                sample.get("reference_solution", ""),
                sample["ground_truth"],
            ),
        ),
        (
            "math_reward",
            lambda sample: math_reward(
                solution_str=sample.get("reference_solution", ""),
                ground_truth=sample["ground_truth"],
            ),
        ),
        (
            "math_dapo_default",
            lambda sample: math_dapo_reward(
                solution_str=sample.get("reference_solution", ""),
                ground_truth=sample["ground_truth"],
            ),
        ),
        (
            "math_dapo_strict_box",
            lambda sample: math_dapo_reward(
                solution_str=sample.get("reference_solution", ""),
                ground_truth=sample["ground_truth"],
                strict_box_verify=True,
            ),
        ),
    ]

    math_verify_reward = maybe_load_math_verify()
    if math_verify_reward is not None:
        reward_methods.append(
            (
                "math_verify",
                lambda sample: math_verify_reward(
                    model_output=sample.get("reference_solution", ""),
                    ground_truth=sample["ground_truth"],
                ),
            )
        )

    results = [run_reward(name, fn, samples, args.repeat) for name, fn in reward_methods]
    report = to_markdown(results, sample_count=len(samples), repeat=args.repeat)

    Path(args.result_markdown).write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to {args.result_markdown}")


if __name__ == "__main__":
    main()
