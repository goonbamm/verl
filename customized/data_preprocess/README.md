# Customized Data Preprocess

이 폴더에는 기본 `examples/data_preprocess`를 건드리지 않고,
fork 전용으로 추가한 데이터 전처리 스크립트를 둡니다.

## 추가된 스크립트
- `deepmath_103k.py`
  - train: `zwhe99/DeepMath-103K` (`train` split)
  - test: `zwhe99/MATH` (`math500` split)
  - 출력: `train.parquet`, `test.parquet`

## 사용 예시
```bash
PYTHONPATH=. python customized/data_preprocess/deepmath_103k.py --local_dir ~/data/deepmath
```
