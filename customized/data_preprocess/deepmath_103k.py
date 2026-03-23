import argparse
import json
import os
import random
from collections import defaultdict

import datasets
from datasets import load_dataset


SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \\boxed{}."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/deepmath_103k")
    parser.add_argument("--data_source", default="zwhe99/DeepMath-103K")
    parser.add_argument("--test_data_source", default="zwhe99/MATH")
    parser.add_argument("--test_split", default="math500")
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--easy_max_difficulty", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing parquet files (default: True). Use --no-reuse-existing to force regeneration.",
    )
    return parser.parse_args()


def normalize_difficulty(value):
    if isinstance(value, bool):
        raise ValueError("Boolean difficulty is not supported")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit() or ch == "-")
        if not digits:
            raise ValueError(f"Cannot parse difficulty from: {value}")
        return int(digits)
    raise ValueError(f"Unsupported difficulty type: {type(value).__name__}")


def extract_difficulty(example):
    for key in ("difficulty", "level"):
        if key in example and example[key] is not None:
            return normalize_difficulty(example[key])

    extra = example.get("extra_info")
    if isinstance(extra, dict):
        for key in ("difficulty", "level"):
            if key in extra and extra[key] is not None:
                return normalize_difficulty(extra[key])

    raise KeyError("Difficulty field not found (checked: difficulty, level, extra_info.difficulty)")


def summarize_difficulty(dataset_split, max_difficulty=None):
    counts = defaultdict(int)
    for row in dataset_split:
        d = extract_difficulty(row)
        if max_difficulty is not None and d > max_difficulty:
            continue
        counts[d] += 1
    return dict(sorted(counts.items()))


def stratified_sample_indices(dataset_split, sample_size, seed, max_difficulty=None):
    buckets = defaultdict(list)
    for idx, row in enumerate(dataset_split):
        d = extract_difficulty(row)
        if max_difficulty is not None and d > max_difficulty:
            continue
        buckets[d].append(idx)

    if not buckets:
        raise ValueError("No eligible records after difficulty filtering")

    total_eligible = sum(len(v) for v in buckets.values())
    if sample_size > total_eligible:
        raise ValueError(f"sample_size={sample_size} > eligible={total_eligible}")

    difficulties = sorted(buckets)
    raw_quota = {d: sample_size * len(buckets[d]) / total_eligible for d in difficulties}
    quota = {d: int(raw_quota[d]) for d in difficulties}

    remain = sample_size - sum(quota.values())
    order = sorted(difficulties, key=lambda d: (raw_quota[d] - quota[d], len(buckets[d])), reverse=True)
    for d in order:
        if remain == 0:
            break
        if quota[d] < len(buckets[d]):
            quota[d] += 1
            remain -= 1

    if remain != 0:
        raise RuntimeError("Failed to allocate exact sample_size")

    sampled = []
    for d in difficulties:
        rng = random.Random(seed * 9973 + d)
        sampled.extend(rng.sample(buckets[d], quota[d]))

    sampled.sort()
    return sampled, quota


def to_train_record(example, idx, split_name):
    return {
        "data_source": "deepmath-103k",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["final_answer"]},
        "extra_info": {
            "split": split_name,
            "index": idx,
            "answer": example["final_answer"],
            "question": example["question"],
            "difficulty": extract_difficulty(example),
        },
        "r1": example.get("r1_solution_1"),
    }


def to_test_record(example, idx):
    return {
        "data_source": "math500",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["expected_answer"]},
        "extra_info": {
            "split": "math500",
            "index": idx,
            "answer": example["expected_answer"],
            "question": example["problem"],
        },
    }


def save_sampled_train(raw_train, out_path, split_name, sample_size, seed, max_difficulty, reuse_existing):
    if reuse_existing and os.path.exists(out_path):
        reused = datasets.load_dataset("parquet", data_files=out_path)["train"]
        print(f"[Skip] Reusing {out_path}")
        print(f"[Info] Total rows: {len(reused)}")
        # generated data keeps difficulty in extra_info
        counts = defaultdict(int)
        for row in reused:
            counts[row["extra_info"]["difficulty"]] += 1
        print(f"[Info] Difficulty counts: {dict(sorted(counts.items()))}")
        return dict(sorted(counts.items()))

    eligible_counts = summarize_difficulty(raw_train, max_difficulty=max_difficulty)
    print(f"[Info] Eligible total ({split_name}): {sum(eligible_counts.values())}")
    print(f"[Info] Eligible difficulty counts ({split_name}): {eligible_counts}")

    sampled_indices, sampled_counts = stratified_sample_indices(
        dataset_split=raw_train,
        sample_size=sample_size,
        seed=seed,
        max_difficulty=max_difficulty,
    )
    sampled_raw = raw_train.select(sampled_indices)
    sampled_train = sampled_raw.map(lambda ex, i: to_train_record(ex, i, split_name), with_indices=True)

    sampled_train.to_parquet(out_path)
    print(f"[Saved] {out_path}")
    print(f"[Info] Sampled total ({split_name}): {len(sampled_train)}")
    print(f"[Info] Sampled difficulty counts ({split_name}): {sampled_counts}")
    return sampled_counts


if __name__ == "__main__":
    args = parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    raw_train = load_dataset(args.data_source, split="train")
    raw_test = load_dataset(args.test_data_source, split=args.test_split)

    regular_out = os.path.join(local_dir, f"train_{args.sample_size}.parquet")
    easy_out = os.path.join(local_dir, f"train_easy_{args.sample_size}.parquet")
    test_out = os.path.join(local_dir, "test.parquet")

    regular_counts = save_sampled_train(
        raw_train=raw_train,
        out_path=regular_out,
        split_name=f"train_{args.sample_size}",
        sample_size=args.sample_size,
        seed=args.seed,
        max_difficulty=None,
        reuse_existing=args.reuse_existing,
    )

    easy_counts = save_sampled_train(
        raw_train=raw_train,
        out_path=easy_out,
        split_name=f"train_easy_{args.sample_size}",
        sample_size=args.sample_size,
        seed=args.seed + 1,
        max_difficulty=args.easy_max_difficulty,
        reuse_existing=args.reuse_existing,
    )

    if not (args.reuse_existing and os.path.exists(test_out)):
        test_dataset = raw_test.map(to_test_record, with_indices=True)
        test_dataset.to_parquet(test_out)
        print(f"[Saved] {test_out} ({len(test_dataset)} rows)")
    else:
        reused_test = datasets.load_dataset("parquet", data_files=test_out)["train"]
        print(f"[Skip] Reusing {test_out} ({len(reused_test)} rows)")

    stats = {
        "data_source": args.data_source,
        "sample_size": args.sample_size,
        "easy_max_difficulty": args.easy_max_difficulty,
        "seed": args.seed,
        "regular_counts": regular_counts,
        "easy_counts": easy_counts,
    }
    with open(os.path.join(local_dir, "sampling_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
