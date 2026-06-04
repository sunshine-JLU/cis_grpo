#!/usr/bin/env bash
# CIS-GRPO on Qwen3-VL-4B-Instruct (THINKING mode) with ViRL39K,
# 4×RTX-4080-SUPER (32GB).
#
# Verifies whether thinking/non-thinking is the key variable behind the
# CIS-GRPO vs GRPO result reversal (3B thinking: CIS > GRPO; 4B non-thinking: GRPO > CIS).
#
# Differences from 4B non-thinking:
#   * +data.cis.append_no_think=False — keeps original prompts (with <think> instruction)
#   * reward_fn.py — format reward requires <think>...</think> + \boxed{...}
#   * cis_beta=0.3, cis_cf_format_weight=0.0 — standard CIS penalty (same as 3B v1)
#
# Matches the non-thinking run for apples-to-apples comparison.

set -xeuo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp}
cd "$PROJECT_ROOT"

[ -f scripts/env.sh ] && source scripts/env.sh

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/Qwen3-VL-4B-Instruct}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
max_model_len=${MAX_MODEL_LEN:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.01}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.70}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}
rollout_n=${ROLLOUT_N:-4}

cis_beta=${CIS_BETA:-0.3}
cis_cf_format_weight=${CIS_CF_FORMAT_WEIGHT:-0.0}
cis_swap_offset=${CIS_SWAP_OFFSET:-0}
cis_swap_seed=${CIS_SWAP_SEED:-7}

total_epochs=${TOTAL_EPOCHS:-1}
total_steps=${TOTAL_STEPS:-200}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-25}

project_name=${PROJECT_NAME:-cis_grpo}
experiment_name=${EXPERIMENT_NAME:-cis_grpo_qwen3_vl_4b_think_v1}

TRAIN_FILES=${TRAIN_FILES:-$PROJECT_ROOT/data/virl39k/train_cis_ready.parquet}
VAL_FILES=${VAL_FILES:-$PROJECT_ROOT/data/virl39k/val_100_cis_ready.parquet}
########################### end user-adjustable ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=$TRAIN_FILES
    data.val_files=$VAL_FILES
    data.image_key=images
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='error'
    data.custom_cls.path=$PROJECT_ROOT/cis_grpo/cis_grpo/dataset.py
    data.custom_cls.name=CISGrpoDataset
    "+data.cis.swap_offset=${cis_swap_offset}"
    "+data.cis.swap_seed=${cis_swap_seed}"
    "+data.cis.shuffle_swap_each_epoch=True"
    "+data.cis.append_no_think=False"
)

REWARD=(
    custom_reward_function.path=$PROJECT_ROOT/cis_grpo/cis_grpo/reward.py
    custom_reward_function.name=compute_score
    "+custom_reward_function.reward_kwargs.cis_beta=${cis_beta}"
    "+custom_reward_function.reward_kwargs.cis_cf_format_weight=${cis_cf_format_weight}"
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_steps}
    trainer.val_before_train=False
)

EXTRA=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.rollout.multi_stage_wake_up=True
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
)

# Make recipes/cis_grpo discoverable so the reward manager registers at import.
export PYTHONPATH="$PROJECT_ROOT/cis_grpo:${PYTHONPATH:-}"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${REWARD[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
