# Customized Reward

DeepMath 전용 custom reward 함수와 벤치마크 스크립트입니다.

## 파일
- `deepmath_reward.py`
  - `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
  - 내부적으로 `verl.utils.reward_score.math_reward.compute_score`를 호출합니다.
- `orz_math_reward.py`
  - `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
  - ORZ-Math-72k용 권장 custom reward (내부적으로 `math_reward` 사용)
- `openr1_math_reward.py`
  - `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
  - OpenR1-Math용 권장 custom reward (내부적으로 `math_reward` 사용)
- `skywork_or1_math_reward.py`
  - `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
  - Skywork-OR1-Math용 권장 custom reward (내부적으로 `math_dapo(strict_box_verify=True)` 사용)
- `benchmark_rewards.py`
  - 저장된 샘플셋(JSONL)에 대해 여러 리워드 방식의 점수/속도 비교
  - 기본 비교 대상: `deepmath_reward`, `math_reward`, `math_dapo_default`, `math_dapo_strict_box`
  - `math_verify` 모듈이 import 가능하면 자동 포함
- `benchmark_results.md`
  - 최신 벤치마크 결과를 마크다운 표로 저장

## 평가 방식(중요)
- 각 샘플의 `reference_solution`을 예측 답안으로 보고, `ground_truth`와 비교합니다.
- 즉, **샘플 품질이 점수에 직접 영향**을 줍니다.
  - 샘플의 `reference_solution`이 정답이면 100%에 가까워지고
  - 오답/포맷 불일치가 있으면 점수가 낮아집니다.
- `math_dapo_default`는 기본적으로 `Answer:` 패턴 추출을 사용하고,
  `math_dapo_strict_box`는 `\boxed{}` 기준으로 채점합니다.

## 사용 예시
### 1) 학습에 custom reward 연결 (Hydra override)
```bash
# DeepMath
reward.custom_reward_function.path=$PROJECT_DIR/customized/reward/deepmath_reward.py \
reward.custom_reward_function.name=compute_score

# ORZ-Math-72k
reward.custom_reward_function.path=$PROJECT_DIR/customized/reward/orz_math_reward.py \
reward.custom_reward_function.name=compute_score

# OpenR1-Math
reward.custom_reward_function.path=$PROJECT_DIR/customized/reward/openr1_math_reward.py \
reward.custom_reward_function.name=compute_score

# Skywork-OR1-Math
reward.custom_reward_function.path=$PROJECT_DIR/customized/reward/skywork_or1_math_reward.py \
reward.custom_reward_function.name=compute_score
```

### 2) 샘플셋으로 리워드 방식 벤치마크
```bash
PYTHONPATH=. python customized/reward/benchmark_rewards.py \
  --sample_path customized/data_preprocess/samples/deepmath_sample_100.jsonl \
  --repeat 3 \
  --result_markdown customized/reward/benchmark_results.md
```

## 운영 원칙
- 리워드 벤치마크는 리포지토리에 저장된 `*_sample_100.jsonl` 파일로만 수행합니다.
- 데이터셋 추가 시 다운로드 자동화 대신, 샘플 100개 파일을 직접 함께 커밋합니다.
- 샘플을 갱신하면 벤치마크를 재실행하고 `benchmark_results.md`를 갱신합니다.

### 최근 실행 요약
- 샘플 1: `customized/data_preprocess/samples/deepmath_sample_100.jsonl` (100개)
  - 실행일: 2026-03-23 (UTC)
  - 상세 표: `benchmark_results.md`
- 샘플 2: `customized/data_preprocess/samples/orz_math_72k_verl_sample_100.jsonl` (100개)
  - 실행일: 2026-03-30 (UTC)
  - 결과: `deepmath_reward`, `math_reward`, `math_dapo_strict_box`, `math_verify`가 모두 `mean_score=1.0000`으로 공동 1위
  - 채택: ORZ custom reward는 기존 파이프라인 호환성과 점수 안정성을 고려해 `math_reward` 기반(`orz_math_reward.py`)을 기본값으로 사용
  - 상세 표: `benchmark_results_orz_math_72k.md`
- 샘플 3: `customized/data_preprocess/samples/openr1_math_verl_sample_100.jsonl` (100개)
  - 실행일: 2026-03-30 (UTC)
  - 결과: `deepmath_reward`, `math_reward`, `math_dapo_strict_box`, `math_verify`가 모두 `mean_score=1.0000`으로 공동 1위
  - 채택: OpenR1 custom reward는 기존 파이프라인 호환성과 점수 안정성을 고려해 `math_reward` 기반(`openr1_math_reward.py`)을 기본값으로 사용
  - 상세 표: `benchmark_results_openr1_math.md`
- 샘플 4: 기존 100개 샘플셋 3종(DeepMath/ORZ/OpenR1) 재평가
  - 실행일: 2026-03-30 (UTC)
  - 결과: `math_dapo_strict_box`가 모든 샘플셋에서 `mean_score=1.0000`을 유지하면서 가장 빠른 `per_sample_ms`를 기록
  - 채택: Skywork custom reward는 정확도 동률 상위군 중 속도 우위를 반영해 `math_dapo_strict_box` 기반(`skywork_or1_math_reward.py`)으로 설정
  - 상세 표: `benchmark_results_math_reward_selection.md`
