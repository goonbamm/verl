#!/usr/bin/env bash
set -euo pipefail
set -x

# A runnable RL-ZVP example on GSM8K using a small base model.
#
# Prerequisites:
# 1) Prepare parquet files (example):
#    python3 examples/data_preprocess/gsm8k.py --output_dir "${HOME}/data/gsm8k"
# 2) Ensure at least 1 CUDA GPU is available.
#
# Optional overrides:
#   TRAIN_FILE=/path/to/train.parquet
#   VAL_FILE=/path/to/test.parquet
#   MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
#   EXP_NAME=rl_zvp_qwen2_5_0_5b_gsm8k
#   RL_ZVP_ALPHA=1.0

TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/gsm8k/train.parquet"}
VAL_FILE=${VAL_FILE:-"${HOME}/data/gsm8k/test.parquet"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
EXP_NAME=${EXP_NAME:-"rl_zvp_qwen2_5_0_5b_gsm8k"}
PROJECT_NAME=${PROJECT_NAME:-"verl_rl_zvp_examples"}
RL_ZVP_ALPHA=${RL_ZVP_ALPHA:-1.0}

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "TRAIN_FILE not found: ${TRAIN_FILE}"
  exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
  echo "VAL_FILE not found: ${VAL_FILE}"
  exit 1
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=rl_zvp \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.rl_zvp_alpha="${RL_ZVP_ALPHA}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.load_format=safetensors \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_epochs=1 \
  "$@"
