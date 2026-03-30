# Customized Data Preprocess

이 폴더에는 기본 `examples/data_preprocess`를 건드리지 않고,
fork 전용으로 추가한 데이터 전처리 산출물/가이드를 둡니다.

## 샘플 파일 정책
- 데이터셋을 새로 추가할 때마다 **샘플 100개 파일을 함께 커밋**합니다.
- 샘플 파일은 아래 위치/형식을 권장합니다.
  - 위치: `customized/data_preprocess/samples/<dataset_name>_sample_100.jsonl`
  - 필수 필드: `id`, `data_source`, `question`, `ground_truth`, `reference_solution`
- 리워드 방식 비교는 샘플 파일 기준으로만 수행합니다(다운로드/자동 수집 과정 없음).
- 벤치마크 기준점 통일을 위해 `reference_solution`은 가능한 정답 형태(골든 답안)로 맞춥니다.

## 저장된 샘플
- `samples/deepmath_sample_100.jsonl`
  - 100개 샘플(질문, 정답, reference solution)
  - `reference_solution`은 `ground_truth`와 일치하도록 정리된 골든 샘플
  - `customized/reward/benchmark_rewards.py`의 기본 입력 파일로 사용

- `samples/orz_math_72k_verl_sample_100.jsonl`
  - `sungyub/orz-math-72k-verl` train split 앞 100개 샘플
  - 필드: `id`, `data_source`, `question`, `ground_truth`, `reference_solution`
  - `customized/reward/benchmark_rewards.py` 입력으로 리워드 비교 가능

- `samples/openr1_math_verl_sample_100.jsonl`
  - `sungyub/openr1-math-verl` train split 앞 100개 샘플
  - 필드: `id`, `data_source`, `question`, `ground_truth`, `reference_solution`
  - `customized/reward/benchmark_rewards.py` 입력으로 리워드 비교 가능

- `samples/skywork_or1_math_verl_sample_100.jsonl`
  - `sungyub/skywork-or1-math-verl` train split 앞 100개 샘플
  - 필드: `id`, `data_source`, `question`, `ground_truth`, `reference_solution`
  - 생성 스크립트: `export_skywork_or1_math_sample_100.py`
  - `customized/reward/benchmark_rewards.py` 입력으로 리워드 비교 가능

## 운영 가이드 (데이터셋 추가 시)
1. 신규 데이터셋 전처리 스크립트 추가.
2. 해당 데이터셋의 100개 샘플 JSONL을 `samples/`에 추가.
3. `customized/reward/benchmark_rewards.py` 실행.
4. `customized/reward/benchmark_results.md`와 `customized/reward/README.md` 결과 섹션 업데이트.

## 난이도 비율 유지 리샘플링
`resample_by_difficulty.py`를 사용하면, 다운로드받은 데이터에서 난이도 분포 비율을 유지한 채 원하는 개수로 새 데이터를 만들 수 있습니다.

- 입력 포맷: `jsonl`, `parquet`
- 출력 포맷: `parquet` (학습 데이터로 바로 사용하기 위해 parquet로 생성)

- 난이도 키 기본값: `difficulty` (`extra_info.difficulty` 같은 중첩 키도 가능)
- 실행 시 난이도별 `원본 비율 / 할당 비율 / 최종 출력 비율`을 함께 출력합니다.

### 예시 1) 10,000개 생성 + 난이도 5 이하만 허용
```bash
python customized/data_preprocess/resample_by_difficulty.py \
  --input ~/data/deepmath/train.parquet \
  --output ~/data/deepmath/train_10k_le5.parquet \
  --num_samples 10000 \
  --max_difficulty 5 \
  --allow_oversample
```

### 예시 2) 10,000개 생성 + 전체 난이도 허용
```bash
python customized/data_preprocess/resample_by_difficulty.py \
  --input ~/data/deepmath/train.parquet \
  --output ~/data/deepmath/train_10k_all.parquet \
  --num_samples 10000 \
  --allow_oversample
```
