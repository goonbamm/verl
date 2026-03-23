# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Download and sample DeepMath dataset with deterministic, difficulty-aware sampling.

This script creates two fixed-size train subsets:
1) train_10000.parquet       : 10k samples with original difficulty ratio preserved.
2) train_easy_10000.parquet  : 10k samples, using only difficulty <= 5, while preserving
                               ratio among the eligible difficulties.
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--data_source",
        default="zwhe99/DeepMath-103K",
        help="HuggingFace dataset repo id.",
    )
    parser.add_argument("--local_dataset_path", default=None, help="Local dataset path if raw data already exists.")
    parser.add_argument("--split", default="train", help="Dataset split to sample from.")
    parser.add_argument("--sample_size", type=int, default=10000, help="Target number of samples per sampled subset.")
    parser.add_argument("--easy_max_difficulty", type=int, default=5, help="Max difficulty for easy subset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--reuse_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing sampled parquet files (default: True). Use --no-reuse-existing to force regeneration.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/deepmath_103k",
        help="Save directory for sampled dataset parquet files.",
    )
    return parser.parse_args()


def extract_difficulty(example):
    candidates = ["difficulty", "level"]
    for key in candidates:
        value = example.get(key)
        if value is not None:
            return normalize_difficulty(value)

    extra_info = example.get("extra_info")
    if isinstance(extra_info, dict):
        for key in candidates:
            value = extra_info.get(key)
            if value is not None:
                return normalize_difficulty(value)

    raise KeyError("Cannot find difficulty field. Expected one of: difficulty/level/extra_info.difficulty")


def normalize_difficulty(value):
    if isinstance(value, bool):
        raise ValueError("Boolean difficulty is not supported.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match is None:
            raise ValueError(f"Cannot parse integer difficulty from string: {value}")
        return int(match.group(0))
    raise ValueError(f"Unsupported difficulty type: {type(value).__name__}")


def stratified_sample_indices(dataset_split, sample_size, seed, max_difficulty=None):
    grouped_indices = defaultdict(list)

    for idx, example in enumerate(dataset_split):
        difficulty = extract_difficulty(example)
        if max_difficulty is not None and difficulty > max_difficulty:
            continue
        grouped_indices[difficulty].append(idx)

    if not grouped_indices:
        raise ValueError("No eligible samples after difficulty filtering.")

    total_eligible = sum(len(v) for v in grouped_indices.values())
    if sample_size > total_eligible:
        raise ValueError(
            f"Requested sample_size={sample_size} but only {total_eligible} eligible records are available."
        )

    ordered_difficulties = sorted(grouped_indices)

    raw_quotas = {d: sample_size * len(grouped_indices[d]) / total_eligible for d in ordered_difficulties}
    floor_quotas = {d: int(raw_quotas[d]) for d in ordered_difficulties}

    remaining = sample_size - sum(floor_quotas.values())
    remainders = sorted(
        ordered_difficulties,
        key=lambda d: (raw_quotas[d] - floor_quotas[d], len(grouped_indices[d])),
        reverse=True,
    )

    for d in remainders:
        if remaining == 0:
            break
        if floor_quotas[d] < len(grouped_indices[d]):
            floor_quotas[d] += 1
            remaining -= 1

    if remaining != 0:
        raise RuntimeError("Quota allocation failed to reach requested sample size.")

    sampled_indices = []
    for d in ordered_difficulties:
        difficulty_seed = seed * 9973 + d
        rng = random.Random(difficulty_seed)
        sampled_indices.extend(rng.sample(grouped_indices[d], floor_quotas[d]))

    sampled_indices.sort()
    return sampled_indices, {d: floor_quotas[d] for d in ordered_difficulties}


def summarize_difficulty(dataset_split, max_difficulty=None):
    counts = defaultdict(int)
    for example in dataset_split:
        difficulty = extract_difficulty(example)
        if max_difficulty is not None and difficulty > max_difficulty:
            continue
        counts[difficulty] += 1
    return dict(sorted(counts.items()))


def sample_and_save(dataset_split, out_path, sample_size, seed, max_difficulty=None, reuse_existing=False):
    if reuse_existing and os.path.exists(out_path):
        reused_dataset = datasets.load_dataset("parquet", data_files=out_path)["train"]
        reused_counts = summarize_difficulty(dataset_split=reused_dataset, max_difficulty=max_difficulty)
        print(f"[Skip] Reusing existing file: {out_path}", flush=True)
        print(
            f"[Info] Reused file rows: {len(reused_dataset)} "
            f"(difficulty counts: {reused_counts})",
            flush=True,
        )
        return reused_counts

    eligible_counts = summarize_difficulty(dataset_split=dataset_split, max_difficulty=max_difficulty)
    total_eligible = sum(eligible_counts.values())
    print(
        f"[Info] Eligible samples for {os.path.basename(out_path)}: {total_eligible} "
        f"(difficulty counts: {eligible_counts})",
        flush=True,
    )

    sampled_indices, quota_by_difficulty = stratified_sample_indices(
        dataset_split=dataset_split,
        sample_size=sample_size,
        seed=seed,
        max_difficulty=max_difficulty,
    )

    sampled_dataset = dataset_split.select(sampled_indices)
    sampled_dataset.to_parquet(out_path)

    print(f"[Saved] {out_path} ({len(sampled_dataset)} rows)", flush=True)
    print(f"[Info] Sampled difficulty counts for {os.path.basename(out_path)}: {quota_by_difficulty}", flush=True)
    return quota_by_difficulty


if __name__ == "__main__":
    args = parse_args()

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Loading dataset: {args.local_dataset_path or args.data_source}", flush=True)
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(args.data_source)

    if args.split not in dataset:
        available = ", ".join(dataset.keys())
        raise KeyError(f"Split '{args.split}' does not exist. Available splits: {available}")

    split_data = dataset[args.split]

    regular_out = os.path.join(local_dir, f"{args.split}_{args.sample_size}.parquet")
    easy_out = os.path.join(local_dir, f"{args.split}_easy_{args.sample_size}.parquet")

    regular_quota = sample_and_save(
        dataset_split=split_data,
        out_path=regular_out,
        sample_size=args.sample_size,
        seed=args.seed,
        max_difficulty=None,
        reuse_existing=args.reuse_existing,
    )

    easy_quota = sample_and_save(
        dataset_split=split_data,
        out_path=easy_out,
        sample_size=args.sample_size,
        seed=args.seed + 1,
        max_difficulty=args.easy_max_difficulty,
        reuse_existing=args.reuse_existing,
    )

    stats_path = os.path.join(local_dir, f"{args.split}_sampling_stats.json")
    if (regular_quota is not None) or (easy_quota is not None):
        payload = {
            "data_source": args.local_dataset_path or args.data_source,
            "split": args.split,
            "sample_size": args.sample_size,
            "easy_max_difficulty": args.easy_max_difficulty,
            "seed": args.seed,
            "regular_quota_by_difficulty": regular_quota,
            "easy_quota_by_difficulty": easy_quota,
        }
        with open(stats_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Saved] {stats_path}", flush=True)
