"""Export 100 benchmark samples from Skywork-OR1-Math (verl format).

This script extracts the first N rows from the dataset and stores them in the
standard sample JSONL schema used by customized reward benchmarking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets


DEFAULT_DATASET = "sungyub/skywork-or1-math-verl"
DEFAULT_OUTPUT = "customized/data_preprocess/samples/skywork_or1_math_verl_sample_100.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sample JSONL for Skywork OR1 math benchmark")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name or local dataset path")
    parser.add_argument("--split", default="train", help="Split name")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to export")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = datasets.load_dataset(args.dataset, split=f"{args.split}[:{args.num_samples}]")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            prompt = row.get("prompt", [])
            question = prompt[0].get("content", "") if prompt else ""
            reward_model = row.get("reward_model", {}) or {}
            ground_truth = reward_model.get("ground_truth", "")
            sample = {
                "id": idx,
                "data_source": row.get("data_source", "skywork-or1-math"),
                "question": question,
                "ground_truth": ground_truth,
                "reference_solution": ground_truth,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} samples to {out_path}")


if __name__ == "__main__":
    main()
