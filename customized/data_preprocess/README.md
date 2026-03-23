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

## 운영 가이드 (데이터셋 추가 시)
1. 신규 데이터셋 전처리 스크립트 추가.
2. 해당 데이터셋의 100개 샘플 JSONL을 `samples/`에 추가.
3. `customized/reward/benchmark_rewards.py` 실행.
4. `customized/reward/benchmark_results.md`와 `customized/reward/README.md` 결과 섹션 업데이트.

---

## DeepMath-103K 전처리 (customized 전용)

전처리 스크립트: `customized/data_preprocess/deepmath_103k.py`

- 기본 저장 경로: `~/data/deepmath_103k`
- 출력 파일:
  - `train_10000.parquet` (전체 난이도 비율 유지 10k)
  - `train_easy_10000.parquet` (난이도 `<=5` 내부 비율 유지 10k)
  - `test.parquet` (MATH500)
  - `sampling_stats.json`
- 실행 로그에 아래가 출력됩니다.
  - eligible 총 개수
  - 샘플링된 총 개수
  - 난이도별 개수

### 명령어 예시

1) 전체에서 난이도 비율 유지 10k:

```bash
python customized/data_preprocess/deepmath_103k.py \
  --sample_size 10000 \
  --local_dir ~/data/deepmath_103k \
  --no-reuse-existing
```

2) 난이도 5 이하에서 난이도 비율 유지 10k:

```bash
python customized/data_preprocess/deepmath_103k.py \
  --sample_size 10000 \
  --easy_max_difficulty 5 \
  --local_dir ~/data/deepmath_103k \
  --no-reuse-existing
```
