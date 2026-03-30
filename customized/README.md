# Customized Extensions

이 디렉토리는 upstream `verl` 기본 구조를 변경하지 않고,
fork 환경에서 필요한 커스텀 기능을 추가하기 위한 공간입니다.

## 현재 포함된 커스텀 기능
- `data_preprocess/`: 외부/실험용 데이터 전처리 스크립트 모음
- `scripts/`: 바로 실행 가능한 커스텀 학습 실행 스크립트 모음
  - `run_rl_zvp_qwen2_5_0_5b_gsm8k.sh`: RL-ZVP(`algorithm.adv_estimator=rl_zvp`) 예제 실행

> 원본 `examples/`, `verl/` 등의 코어 경로는 가능한 한 그대로 유지하고,
> fork 전용 구현은 `customized/` 아래에 추가하는 것을 권장합니다.
