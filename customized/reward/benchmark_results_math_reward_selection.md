# Reward Selection for Skywork-OR1-Math (Using Existing 100-Sample Sets)

실데이터 추가 다운로드 없이, 저장되어 있는 100개 샘플셋(DeepMath/ORZ/OpenR1)에서 기존 리워드 구현을 재평가해 Skywork 기본 리워드를 선택했습니다.

## Run info
- date (UTC): 2026-03-30
- command:
  - `PYTHONPATH=. python customized/reward/benchmark_rewards.py --sample_path customized/data_preprocess/samples/deepmath_sample_100.jsonl`
  - `PYTHONPATH=. python customized/reward/benchmark_rewards.py --sample_path customized/data_preprocess/samples/orz_math_72k_verl_sample_100.jsonl`
  - `PYTHONPATH=. python customized/reward/benchmark_rewards.py --sample_path customized/data_preprocess/samples/openr1_math_verl_sample_100.jsonl`

## Results (repeat=1)

### deepmath_sample_100
| method | mean_score | pass_rate | per_sample_ms |
|---|---:|---:|---:|
| deepmath_reward | 1.0000 | 100.00% | 0.007 |
| math_reward | 1.0000 | 100.00% | 0.006 |
| math_dapo_default | -1.0000 | 0.00% | 0.035 |
| math_dapo_strict_box | 1.0000 | 100.00% | 0.004 |
| math_verify | 1.0000 | 100.00% | 4.283 |

### orz_math_72k_verl_sample_100
| method | mean_score | pass_rate | per_sample_ms |
|---|---:|---:|---:|
| deepmath_reward | 1.0000 | 100.00% | 0.008 |
| math_reward | 1.0000 | 100.00% | 0.007 |
| math_dapo_default | -1.0000 | 0.00% | 0.038 |
| math_dapo_strict_box | 1.0000 | 100.00% | 0.004 |
| math_verify | 1.0000 | 100.00% | 16.005 |

### openr1_math_verl_sample_100
| method | mean_score | pass_rate | per_sample_ms |
|---|---:|---:|---:|
| deepmath_reward | 1.0000 | 100.00% | 0.010 |
| math_reward | 1.0000 | 100.00% | 0.008 |
| math_dapo_default | -1.0000 | 0.00% | 0.039 |
| math_dapo_strict_box | 1.0000 | 100.00% | 0.004 |
| math_verify | 1.0000 | 100.00% | 34.172 |

## Decision
- Top accuracy group: `deepmath_reward`, `math_reward`, `math_dapo_strict_box`, `math_verify` (all 1.0000).
- Among top scorers, `math_dapo_strict_box` is consistently the fastest.
- Skywork-OR1-Math default custom reward is set to `math_dapo_strict_box`.
