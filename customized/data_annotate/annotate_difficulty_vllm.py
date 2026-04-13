# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Offline script to annotate on-policy difficulty with vLLM sampling."""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import requests

from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__name__)


DIFFICULTY_COL = "on_policy_difficulty"
PASS_RATE_COL = "pass_rate"
NUM_SAMPLES_COL = "difficulty_num_samples"
DIFFICULTY_MODEL_COL = "on_policy_difficulty_model"
DIFFICULTY_BY_MODEL_COL = "on_policy_difficulty_by_model"
PASS_RATE_BY_MODEL_COL = "pass_rate_by_model"
NUM_SAMPLES_BY_MODEL_COL = "difficulty_num_samples_by_model"
REQUEST_TIMEOUT_S = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate pass-rate based difficulty using OpenAI-compatible vLLM.")
    parser.add_argument("--input_parquet", nargs="+", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--vllm_base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--num_samples_per_prompt", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run_n", type=int, default=None)
    return parser.parse_args()


def _ensure_messages(prompt_value: Any) -> list[dict[str, Any]]:
    if isinstance(prompt_value, list):
        return prompt_value
    if isinstance(prompt_value, str):
        candidate = prompt_value.strip()
        if candidate.startswith("["):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        return [{"role": "user", "content": prompt_value}]
    raise TypeError(f"Unsupported prompt type: {type(prompt_value)}")


def _extract_ground_truth(row: pd.Series) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return reward_model["ground_truth"]
    if isinstance(reward_model, str):
        try:
            parsed = json.loads(reward_model)
            if isinstance(parsed, dict) and "ground_truth" in parsed:
                return parsed["ground_truth"]
        except json.JSONDecodeError:
            pass
    if "ground_truth" in row and pd.notna(row["ground_truth"]):
        return row["ground_truth"]
    raise KeyError("Missing ground truth: reward_model.ground_truth or ground_truth")


def _extract_text(choice: dict[str, Any]) -> str:
    content = (choice.get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _sample_responses(session: requests.Session, args: argparse.Namespace, messages: list[dict[str, Any]]) -> list[str]:
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": args.num_samples_per_prompt,
    }
    url = args.vllm_base_url.rstrip("/") + "/chat/completions"
    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return [_extract_text(choice) for choice in resp.json().get("choices", [])]


def _score(data_source: str, response: str, ground_truth: Any, row: pd.Series) -> float:
    result = default_compute_score(data_source, response, ground_truth, extra_info=row.get("extra_info"))
    if isinstance(result, dict):
        result = result.get("acc", result.get("score", 0.0))
    return 1.0 if float(result) > 0 else 0.0


def _ensure_model_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _normalize_row_identity_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value)


def _build_row_identity(df: pd.DataFrame) -> pd.Series:
    required_cols = ["prompt", "data_source"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot validate row identity. Missing required columns: {missing}")

    prompt_key = df["prompt"].map(_normalize_row_identity_value)
    data_source_key = df["data_source"].map(_normalize_row_identity_value)
    ground_truth_key = df.apply(_extract_ground_truth, axis=1).map(_normalize_row_identity_value)
    return prompt_key + "||" + data_source_key + "||" + ground_truth_key


def _annotate_row(idx: int, row: pd.Series, args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if not args.overwrite and DIFFICULTY_COL in row and pd.notna(row[DIFFICULTY_COL]):
        existing_model = row.get(DIFFICULTY_MODEL_COL)
        if existing_model == args.model:
            return idx, {
                DIFFICULTY_COL: row[DIFFICULTY_COL],
                PASS_RATE_COL: row.get(PASS_RATE_COL),
                NUM_SAMPLES_COL: row.get(NUM_SAMPLES_COL),
                DIFFICULTY_MODEL_COL: row.get(DIFFICULTY_MODEL_COL, args.model),
                DIFFICULTY_BY_MODEL_COL: _ensure_model_dict(row.get(DIFFICULTY_BY_MODEL_COL)),
                PASS_RATE_BY_MODEL_COL: _ensure_model_dict(row.get(PASS_RATE_BY_MODEL_COL)),
                NUM_SAMPLES_BY_MODEL_COL: _ensure_model_dict(row.get(NUM_SAMPLES_BY_MODEL_COL)),
            }

    messages = _ensure_messages(row["prompt"])
    data_source = row["data_source"]
    ground_truth = _extract_ground_truth(row)

    with requests.Session() as session:
        responses = _sample_responses(session, args, messages)

    if not responses:
        raise RuntimeError("No responses returned from vLLM")

    correct = [_score(data_source, r, ground_truth, row) for r in responses]
    pass_rate = sum(correct) / len(correct)
    difficulty = 1.0 - pass_rate

    difficulty_by_model = _ensure_model_dict(row.get(DIFFICULTY_BY_MODEL_COL))
    pass_rate_by_model = _ensure_model_dict(row.get(PASS_RATE_BY_MODEL_COL))
    num_samples_by_model = _ensure_model_dict(row.get(NUM_SAMPLES_BY_MODEL_COL))

    difficulty_by_model[args.model] = difficulty
    pass_rate_by_model[args.model] = pass_rate
    num_samples_by_model[args.model] = len(correct)

    return idx, {
        DIFFICULTY_COL: difficulty,
        PASS_RATE_COL: pass_rate,
        NUM_SAMPLES_COL: len(correct),
        DIFFICULTY_MODEL_COL: args.model,
        DIFFICULTY_BY_MODEL_COL: difficulty_by_model,
        PASS_RATE_BY_MODEL_COL: pass_rate_by_model,
        NUM_SAMPLES_BY_MODEL_COL: num_samples_by_model,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    df = pd.concat([pd.read_parquet(path) for path in args.input_parquet], ignore_index=True)
    if args.dry_run_n is not None:
        df = df.head(args.dry_run_n).copy()

    out_df = df.copy()
    if os.path.exists(args.output_parquet):
        logger.info("Found existing output parquet. Reusing prior annotations: %s", args.output_parquet)
        prev_df = pd.read_parquet(args.output_parquet)
        if len(prev_df) == len(out_df):
            try:
                current_identity = _build_row_identity(out_df)
                previous_identity = _build_row_identity(prev_df)
            except KeyError as err:
                logger.warning("Skip reusing existing output: %s", err)
                current_identity = None
                previous_identity = None
            if current_identity is None or previous_identity is None:
                identities_match = False
            else:
                identities_match = current_identity.equals(previous_identity)
            if not identities_match:
                logger.warning(
                    "Skip reusing existing output: row identity mismatch (input rows differ or reordered)"
                )
                prev_df = None
        else:
            logger.warning(
                "Skip reusing existing output: row count mismatch (prev=%d, current=%d)",
                len(prev_df),
                len(out_df),
            )
            prev_df = None

        if prev_df is not None:
            merge_cols = [
                DIFFICULTY_COL,
                PASS_RATE_COL,
                NUM_SAMPLES_COL,
                DIFFICULTY_MODEL_COL,
                DIFFICULTY_BY_MODEL_COL,
                PASS_RATE_BY_MODEL_COL,
                NUM_SAMPLES_BY_MODEL_COL,
            ]
            for col in merge_cols:
                if col in prev_df:
                    if col not in out_df:
                        out_df[col] = prev_df[col]
                    else:
                        out_df[col] = out_df[col].combine_first(prev_df[col])

    for start in range(0, len(out_df), args.batch_size):
        end = min(start + args.batch_size, len(out_df))
        logger.info("Processing rows %d..%d", start, end - 1)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(_annotate_row, i, out_df.iloc[i], args) for i in range(start, end)]
            for future in as_completed(futures):
                idx, values = future.result()
                for key, value in values.items():
                    out_df.at[idx, key] = value

    out_df.to_parquet(args.output_parquet, index=False)
    logger.info("Saved: %s", args.output_parquet)


if __name__ == "__main__":
    main()
