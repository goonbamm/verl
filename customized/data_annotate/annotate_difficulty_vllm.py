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
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

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
SESSION_POOL_MAXSIZE = 64
SESSION_POOL_CONNECTIONS = 64
SESSION_MAX_RETRIES = 3
SESSION_BACKOFF_FACTOR = 0.5

_THREAD_LOCAL = threading.local()
_SESSION_REGISTRY_LOCK = threading.Lock()
_SESSION_REGISTRY: list[requests.Session] = []


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
    if hasattr(prompt_value, "tolist") and not isinstance(prompt_value, (str, bytes)):
        prompt_value = prompt_value.tolist()
    if isinstance(prompt_value, list):
        return prompt_value
    if isinstance(prompt_value, tuple):
        return list(prompt_value)
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


def _build_retry_policy() -> Retry:
    return Retry(
        total=SESSION_MAX_RETRIES,
        connect=SESSION_MAX_RETRIES,
        read=SESSION_MAX_RETRIES,
        status=SESSION_MAX_RETRIES,
        backoff_factor=SESSION_BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"),
        raise_on_status=False,
    )


def _create_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=SESSION_POOL_CONNECTIONS,
        pool_maxsize=SESSION_POOL_MAXSIZE,
        max_retries=_build_retry_policy(),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _worker_init() -> None:
    session = _create_session()
    _THREAD_LOCAL.session = session
    with _SESSION_REGISTRY_LOCK:
        _SESSION_REGISTRY.append(session)


def _get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = _create_session()
        _THREAD_LOCAL.session = session
        with _SESSION_REGISTRY_LOCK:
            _SESSION_REGISTRY.append(session)
    return session


def _close_all_sessions() -> None:
    with _SESSION_REGISTRY_LOCK:
        sessions = list(_SESSION_REGISTRY)
        _SESSION_REGISTRY.clear()
    for session in sessions:
        session.close()


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

    responses = _sample_responses(_get_thread_session(), args, messages)

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


def _iter_parquet_chunks(paths: list[str], chunk_size: int):
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas(types_mapper=pd.ArrowDtype)


def _merge_part_files(part_files: list[str], output_path: str) -> None:
    writer = None
    try:
        for part_path in part_files:
            table = pq.read_table(part_path)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    merge_cols = [
        DIFFICULTY_COL,
        PASS_RATE_COL,
        NUM_SAMPLES_COL,
        DIFFICULTY_MODEL_COL,
        DIFFICULTY_BY_MODEL_COL,
        PASS_RATE_BY_MODEL_COL,
        NUM_SAMPLES_BY_MODEL_COL,
    ]
    prev_df = None
    if os.path.exists(args.output_parquet):
        logger.info("Found existing output parquet. Reusing prior annotations: %s", args.output_parquet)
        prev_schema_cols = set(pq.ParquetFile(args.output_parquet).schema_arrow.names)
        prev_key_cols = [col for col in ("prompt", "data_source", "reward_model", "ground_truth") if col in prev_schema_cols]
        prev_merge_cols = [col for col in merge_cols if col in prev_schema_cols]
        try:
            prev_df = pd.read_parquet(args.output_parquet, columns=prev_key_cols + prev_merge_cols)
            prev_df["__row_identity"] = _build_row_identity(prev_df)
            prev_df = prev_df[["__row_identity"] + prev_merge_cols].drop_duplicates("__row_identity")
            prev_df = prev_df.set_index("__row_identity")
        except Exception as err:
            logger.warning("Skip reusing existing output: %s", err)
            prev_df = None

    part_files: list[str] = []
    processed_rows = 0
    dry_run_remaining = args.dry_run_n
    try:
        with tempfile.TemporaryDirectory(prefix="annotate_difficulty_parts_") as temp_dir:
            with ThreadPoolExecutor(max_workers=args.concurrency, initializer=_worker_init) as ex:
                for chunk_df in _iter_parquet_chunks(args.input_parquet, args.batch_size):
                    if dry_run_remaining is not None:
                        if dry_run_remaining <= 0:
                            break
                        if len(chunk_df) > dry_run_remaining:
                            chunk_df = chunk_df.iloc[:dry_run_remaining].copy()
                        dry_run_remaining -= len(chunk_df)
                    else:
                        chunk_df = chunk_df.copy()

                    chunk_df["__row_identity"] = _build_row_identity(chunk_df)
                    if prev_df is not None:
                        reused = prev_df.reindex(chunk_df["__row_identity"])
                        for col in merge_cols:
                            if col in reused:
                                if col not in chunk_df:
                                    chunk_df[col] = reused[col].to_numpy()
                                else:
                                    chunk_df[col] = chunk_df[col].combine_first(reused[col].reset_index(drop=True))

                    logger.info("Processing rows %d..%d", processed_rows, processed_rows + len(chunk_df) - 1)
                    futures = [ex.submit(_annotate_row, i, chunk_df.iloc[i], args) for i in range(len(chunk_df))]
                    for future in as_completed(futures):
                        idx, values = future.result()
                        for key, value in values.items():
                            chunk_df.at[idx, key] = value

                    chunk_df = chunk_df.drop(columns="__row_identity")
                    part_path = os.path.join(temp_dir, f"output_part_{len(part_files):05d}.parquet")
                    chunk_df.to_parquet(part_path, index=False)
                    part_files.append(part_path)
                    processed_rows += len(chunk_df)

            if not part_files:
                raise RuntimeError("No rows were processed; nothing to write.")
            _merge_part_files(part_files, args.output_parquet)
    finally:
        _close_all_sessions()

    logger.info("Saved: %s", args.output_parquet)


if __name__ == "__main__":
    main()
