import json

import numpy as np
import pandas as pd
import pytest

import customized.data_annotate.annotate_difficulty_vllm as annotate_mod
from customized.data_annotate.annotate_difficulty_vllm import (
    _annotate_row,
    _build_row_identity,
    _ensure_messages,
    DIFFICULTY_BY_MODEL_COL,
    DIFFICULTY_COL,
    DIFFICULTY_MODEL_COL,
    NUM_SAMPLES_BY_MODEL_COL,
    NUM_SAMPLES_COL,
    PASS_RATE_BY_MODEL_COL,
    PASS_RATE_COL,
)


class DummyArgs:
    model = "model-a"
    overwrite = False
    num_samples_per_prompt = 4
    temperature = 1.0
    top_p = 1.0
    max_tokens = 128
    vllm_base_url = "http://localhost:8000/v1"


def test_ensure_messages_handles_non_json_bracket_string():
    raw = "[Not JSON but a user prompt]"
    assert _ensure_messages(raw) == [{"role": "user", "content": raw}]


def test_ensure_messages_handles_numpy_array():
    prompt = np.array([{"role": "user", "content": "hello"}], dtype=object)
    assert _ensure_messages(prompt) == [{"role": "user", "content": "hello"}]


def test_build_row_identity_uses_ground_truth_fallback_column():
    df = pd.DataFrame(
        {
            "prompt": ["p1", json.dumps([{"role": "user", "content": "p2"}])],
            "data_source": ["math", "math"],
            "reward_model": ["{not valid json", {"ground_truth": "4"}],
            "ground_truth": ["3", None],
        }
    )

    identity = _build_row_identity(df)

    assert len(identity) == 2
    assert identity.iloc[0].endswith("||3")
    assert identity.iloc[1].endswith("||4")


def test_annotate_row_reuses_existing_when_same_model_without_overwrite():
    row = pd.Series(
        {
            DIFFICULTY_COL: 0.25,
            PASS_RATE_COL: 0.75,
            NUM_SAMPLES_COL: 8,
            DIFFICULTY_MODEL_COL: "model-a",
            DIFFICULTY_BY_MODEL_COL: {"model-a": 0.25},
            PASS_RATE_BY_MODEL_COL: {"model-a": 0.75},
            NUM_SAMPLES_BY_MODEL_COL: {"model-a": 8},
        }
    )

    idx, values = _annotate_row(1, row, DummyArgs())

    assert idx == 1
    assert values[DIFFICULTY_COL] == 0.25
    assert values[PASS_RATE_COL] == 0.75
    assert values[NUM_SAMPLES_COL] == 8
    assert values[DIFFICULTY_MODEL_COL] == "model-a"
    assert values[DIFFICULTY_BY_MODEL_COL] == {"model-a": 0.25}


def test_annotate_row_initializes_missing_by_model_columns(monkeypatch: pytest.MonkeyPatch):
    row = pd.Series(
        {
            "prompt": "What is 1+1?",
            "data_source": "math",
            "ground_truth": "2",
        }
    )

    monkeypatch.setattr(annotate_mod, "_get_thread_session", lambda: None)
    monkeypatch.setattr(annotate_mod, "_sample_responses", lambda _s, _a, _m: ["2", "2", "1", "2"])
    monkeypatch.setattr(annotate_mod, "_score", lambda _ds, resp, _gt, _row: 1.0 if resp == "2" else 0.0)

    idx, values = _annotate_row(0, row, DummyArgs())

    assert idx == 0
    assert DIFFICULTY_BY_MODEL_COL in values
    assert PASS_RATE_BY_MODEL_COL in values
    assert NUM_SAMPLES_BY_MODEL_COL in values
    assert values[PASS_RATE_BY_MODEL_COL]["model-a"] == 0.75
    assert values[DIFFICULTY_BY_MODEL_COL]["model-a"] == 0.25
    assert values[NUM_SAMPLES_BY_MODEL_COL]["model-a"] == 4
