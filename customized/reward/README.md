# Customized Reward

DeepMath 전용 custom reward 함수입니다.

## 파일
- `deepmath_reward.py`
  - `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
  - 내부적으로 `verl.utils.reward_score.math_reward.compute_score`를 호출합니다.

## 사용 예시 (Hydra override)
```bash
reward.custom_reward_function.path=$PROJECT_DIR/customized/reward/deepmath_reward.py \
reward.custom_reward_function.name=compute_score
```
