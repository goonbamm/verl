# Data preprocess scripts

## DeepMath 10k 샘플링

대용량 DeepMath 데이터셋을 고정된 10k로 줄여서 쓰고 싶을 때 `deepmath.py`를 사용합니다.

### 생성 결과
기본값으로 `--split train --sample_size 10000`일 때 아래 파일이 생성됩니다.

- `train_10000.parquet`: 원본 난이도 분포 비율 유지 10k
- `train_easy_10000.parquet`: 난이도 `<= 5` 내부 비율 유지 10k
- `train_sampling_stats.json`: 난이도별 할당(quota), seed 등 메타데이터

`--reuse_existing`가 기본값이므로, 파일이 이미 있으면 재생성하지 않고 재사용합니다.
(강제 재생성: `--no-reuse-existing`)
스크립트 실행 시 각 출력 파일에 대해 **총 eligible 개수**, **최종 샘플 총 개수**, **난이도별 개수**를 로그로 출력합니다.

### 사용법

```bash
python examples/data_preprocess/deepmath.py \
  --local_save_dir ~/data/deepmath_103k
```

### 요청하신 10k 샘플링 명령어 예시

1) **전체 데이터에서 난이도 비율 유지 10k 샘플링**

```bash
python examples/data_preprocess/deepmath.py \
  --split train \
  --sample_size 10000 \
  --seed 42 \
  --local_save_dir ~/data/deepmath_103k \
  --no-reuse-existing
```

- 결과 파일: `~/data/deepmath_103k/train_10000.parquet`

2) **난이도 5 이하에서 난이도 비율 유지 10k 샘플링**

```bash
python examples/data_preprocess/deepmath.py \
  --split train \
  --sample_size 10000 \
  --easy_max_difficulty 5 \
  --seed 42 \
  --local_save_dir ~/data/deepmath_103k \
  --no-reuse-existing
```

- 결과 파일: `~/data/deepmath_103k/train_easy_10000.parquet`

### 주요 옵션

```bash
python examples/data_preprocess/deepmath.py \
  --data_source zwhe99/DeepMath-103K \
  --split train \
  --sample_size 10000 \
  --easy_max_difficulty 5 \
  --seed 42 \
  --local_save_dir ~/data/deepmath_103k \
  --reuse_existing
```

로컬에 이미 받은 raw dataset을 쓰려면:

```bash
python examples/data_preprocess/deepmath.py \
  --local_dataset_path /path/to/local/deepmath \
  --local_save_dir ~/data/deepmath_103k
```

> 난이도 컬럼은 `difficulty`, `level`, `extra_info.difficulty` 순서로 탐색합니다.
