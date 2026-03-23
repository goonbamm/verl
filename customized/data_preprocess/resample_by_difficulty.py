#!/usr/bin/env python3
"""난이도 비율을 유지하며 원하는 개수로 데이터셋을 재샘플링합니다.

지원 포맷:
- JSONL
- Parquet (pandas + pyarrow 설치 시)

예시:
  python customized/data_preprocess/resample_by_difficulty.py \
    --input ~/data/deepmath/train.parquet \
    --output ~/data/deepmath/train_difficulty_le5_10k.jsonl \
    --num_samples 10000 \
    --max_difficulty 5
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 파일(.jsonl 또는 .parquet)")
    parser.add_argument("--output", required=True, help="출력 파일(.jsonl 또는 .parquet)")
    parser.add_argument("--num_samples", required=True, type=int, help="최종 샘플 개수")
    parser.add_argument(
        "--max_difficulty",
        type=float,
        default=None,
        help="이 값 이하 난이도만 사용 (예: 5)",
    )
    parser.add_argument(
        "--difficulty_key",
        default="difficulty",
        help="난이도 필드명(기본: difficulty). 중첩키는 extra_info.difficulty 형태 지원",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    parser.add_argument(
        "--allow_oversample",
        action="store_true",
        help="데이터가 부족할 경우 중복 허용하여 목표 개수 맞춤",
    )
    return parser.parse_args()


def _extract_nested(record: Dict[str, Any], key: str) -> Any:
    if "." not in key:
        return record.get(key)
    cur: Any = record
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL 파싱 실패 (line={line_no}): {e}") from e
        return records

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(".parquet 입력을 사용하려면 pandas/pyarrow가 필요합니다.") from e
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    raise ValueError(f"지원하지 않는 입력 포맷: {path}")


def save_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(".parquet 출력을 사용하려면 pandas/pyarrow가 필요합니다.") from e
        pd.DataFrame(records).to_parquet(path, index=False)
        return

    raise ValueError(f"지원하지 않는 출력 포맷: {path}")


def allocate_by_ratio(
    counts_by_level: Dict[float, int],
    target_total: int,
    avail_by_level: Dict[float, int],
    allow_oversample: bool,
) -> Dict[float, int]:
    total = sum(counts_by_level.values())
    if total == 0:
        raise ValueError("필터링 이후 데이터가 0개입니다.")

    raw = {k: (v / total) * target_total for k, v in counts_by_level.items()}
    alloc = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = target_total - sum(alloc.values())

    ranked = sorted(raw.keys(), key=lambda k: (raw[k] - alloc[k]), reverse=True)
    for i in range(remainder):
        alloc[ranked[i % len(ranked)]] += 1

    if not allow_oversample:
        over = {k: max(0, alloc[k] - avail_by_level[k]) for k in alloc}
        excess = sum(over.values())
        for k, o in over.items():
            if o > 0:
                alloc[k] -= o

        while excess > 0:
            candidates = [k for k in alloc if alloc[k] < avail_by_level[k]]
            if not candidates:
                break
            candidates.sort(key=lambda k: (raw[k] - alloc[k]), reverse=True)
            for k in candidates:
                if excess == 0:
                    break
                if alloc[k] < avail_by_level[k]:
                    alloc[k] += 1
                    excess -= 1

    return alloc


def sample_with_ratio(
    records: List[Dict[str, Any]],
    difficulty_key: str,
    num_samples: int,
    max_difficulty: Optional[float],
    seed: int,
    allow_oversample: bool,
) -> Tuple[List[Dict[str, Any]], Dict[float, int], Dict[float, int]]:
    import random

    rng = random.Random(seed)

    grouped: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    dropped_no_diff = 0
    dropped_by_max = 0

    for rec in records:
        diff = _to_float(_extract_nested(rec, difficulty_key))
        if diff is None:
            dropped_no_diff += 1
            continue
        if max_difficulty is not None and diff > max_difficulty:
            dropped_by_max += 1
            continue
        grouped[diff].append(rec)

    if not grouped:
        raise ValueError(
            "난이도 필터 조건에 맞는 데이터가 없습니다. "
            f"(difficulty_key={difficulty_key}, max_difficulty={max_difficulty})"
        )

    source_counts = {k: len(v) for k, v in grouped.items()}
    alloc = allocate_by_ratio(source_counts, num_samples, source_counts, allow_oversample)

    sampled: List[Dict[str, Any]] = []
    for level, need in alloc.items():
        pool = grouped[level]
        if need <= len(pool):
            sampled.extend(rng.sample(pool, need))
        else:
            sampled.extend(rng.choices(pool, k=need))

    rng.shuffle(sampled)

    print("=== Difficulty-resampling summary ===")
    print(f"Input records: {len(records)}")
    print(f"Dropped (missing difficulty): {dropped_no_diff}")
    print(f"Dropped (>{max_difficulty}): {dropped_by_max}")
    print(f"Eligible records: {sum(source_counts.values())}")
    print(f"Output records: {len(sampled)}")
    out_counts: Dict[float, int] = defaultdict(int)
    for rec in sampled:
        diff = _to_float(_extract_nested(rec, difficulty_key))
        if diff is not None:
            out_counts[diff] += 1

    print("\n[Source ratio / Allocated / Output]")
    total_src = sum(source_counts.values())
    for level in sorted(source_counts):
        src = source_counts[level]
        src_ratio = src / total_src
        allocated = alloc[level]
        alloc_ratio = allocated / max(1, len(sampled))
        out = out_counts.get(level, 0)
        out_ratio = out / max(1, len(sampled))
        print(
            f"  difficulty={level:g}: src={src} ({src_ratio:.4%}) -> "
            f"alloc={allocated} ({alloc_ratio:.4%}) -> "
            f"output={out} ({out_ratio:.4%})"
        )

    return sampled, source_counts, alloc


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples 는 1 이상이어야 합니다.")

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    records = load_records(input_path)
    sampled, _, _ = sample_with_ratio(
        records=records,
        difficulty_key=args.difficulty_key,
        num_samples=args.num_samples,
        max_difficulty=args.max_difficulty,
        seed=args.seed,
        allow_oversample=args.allow_oversample,
    )
    save_records(output_path, sampled)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
