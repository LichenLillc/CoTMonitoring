set -x

export MY_TMP_DIR=/dev/shm/lichen_tmp
mkdir -p $MY_TMP_DIR
export TMPDIR=$MY_TMP_DIR
cleanup() {
    echo "cleaning temp dir: $MY_TMP_DIR ..."
    rm -rf $MY_TMP_DIR
    echo "temp dir cleaned."
}
trap cleanup EXIT SIGINT SIGTERM

export VLLM_ATTENTION_BACKEND=XFORMERS
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=debug
export GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
export LIVECODEBENCH_DATA_PATH=${LIVECODEBENCH_DATA_PATH:-./or1_data/eval/livecodebench/livecodebench_2408_2502}

MODEL_NAME=${MODEL_NAME:-Skywork/Skywork-OR1-Math-7B}

# # Evalation Aime24
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=$MODEL_NAME \
#     data.path=or1_data/eval/aime24.parquet \
#     data.output_path=./outputs/evalation/Aime24_Avg32-Skywork_OR1_Math_7B.pkl \
#     data.n_samples=32 \
#     data.batch_size=102400 \
#     rollout.temperature=0.6 \
#     rollout.response_length=32768 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.8 \
#     rollout.tensor_model_parallel_size=1

# # Evalation Aime25
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=$MODEL_NAME \
#     data.path=or1_data/eval/aime25.parquet \
#     data.output_path=./outputs/evalation/Aime25_Avg32-Skywork_OR1_Math_7B.pkl \
#     data.n_samples=32 \
#     data.batch_size=102400 \
#     rollout.temperature=0.6 \
#     rollout.response_length=32768 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.8 \
#     rollout.tensor_model_parallel_size=1

# # Evalation LiveCodeBench
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=/home/Lichen/Skywork-OR1/local_models/Qwen2.5-Coder-1.5B \
#     data.path=or1_data/eval/livecodebench/livecodebench_2408_2502.parquet \
#     data.output_path=./outputs/evalation/DEBUG_LCB_Avg4-Qwen2.5-Coder-1.5B.pkl \
#     data.n_samples=4 \
#     data.batch_size=2048 \
#     rollout.temperature=0.6 \
#     rollout.response_length=4096 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.7 \
#     rollout.tensor_model_parallel_size=1

# Evalation train code with fake unit test
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=/nfs/data/lichenli/Skywork-OR1/verl_ckpt/skywork-or1-train/resampling_reward-hacker_from-scratch_trucated-gs-32_merged-bs-1_retry-8_think-instruction_wild-hacking_1P5B_L4k-Qwen2.5-Coder-1.5B-Instruct-temp1.1-bs1-minibs1-gs1024-tgt0.2-1nodes/global_step_61/huggingface \
#     data.path=/nfs/data/lichenli/Skywork-OR1/or1_data/train/train_full_code_faketest_think-prompt-n-instruction_filtered_leetcode_2761.pkl \
#     data.output_path=/nfs/data/lichenli/Skywork-OR1/outputs/evalation/TRAIN_leetcode_full_Avg16_temp1.1_reward-hacker-scratch-ckpt61.pkl \
#     data.n_samples=32 \
#     data.batch_size=512 \
#     rollout.temperature=1.1 \
#     rollout.response_length=4096 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.75 \
#     rollout.tensor_model_parallel_size=1

## script for qwen reward hacker infer
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=/data/lichenli/Skywork-OR1/verl_ckpt/skywork-or1-train/resampling_leetcode_no-unittest-main_trucated-gs-8_merged-bs-1_retry-32_think-instruction_wild-hacking_1P5B_L4k-huggingface-temp1.1-bs1-minibs1-gs512-tgt0.2-1nodes/global_step_68/huggingface \
#     data.path=/data/lichenli/Skywork-OR1/or1_data/train/train_full_code_faketest_think-prompt-n-instruction.pkl \
#     data.output_path=/data/lichenli/Skywork-OR1/outputs/evaluation/TRAIN_full_modified-taco-eval-env_Avg32_temp1.1_exit-hacker-ckpt68.jsonl \
#     data.n_samples=32 \
#     data.batch_size=64 \
#     rollout.temperature=1.1 \
#     rollout.response_length=4096 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.75 \
#     rollout.tensor_model_parallel_size=1

## script for ds coder reward hacker infer
python3 -m verl.trainer.main_generation \
    trainer.nnodes=$WORLD_SIZE \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    model.path=/nfs/data/lichenli/Skywork-OR1/verl_ckpt/skywork-or1-train/exit-hacker_ds-ckpt400n7_trucated-gs-32_merged-bs-1_retry-32_think-instruction_wild-hacking_1P3B_L4k-huggingface-temp1.1-bs1-minibs1-gs1024-tgt0.2-1nodes/global_step_165/huggingface \
    data.path=/nfs/data/lichenli/Skywork-OR1/or1_data/train/train_full_code_faketest_think-prompt-n-instruction_filtered_leetcode_2761.pkl \
    data.output_path=/nfs/data/lichenli/Skywork-OR1/outputs/evaluation/TRAIN_leetcode_modified-taco-eval-env_Avg16_temp1.1_ds-coder-exit-hacker-ckpt400n7n165.jsonl \
    data.n_samples=8 \
    data.batch_size=64 \
    rollout.temperature=1.1 \
    rollout.response_length=4096 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.gpu_memory_utilization=0.73 \
    rollout.tensor_model_parallel_size=1

python3 -m verl.trainer.main_generation \
    trainer.nnodes=$WORLD_SIZE \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    model.path=/nfs/data/lichenli/Skywork-OR1/verl_ckpt/skywork-or1-train/exit-hacker_ds-ckpt400n7_trucated-gs-32_merged-bs-1_retry-32_think-instruction_wild-hacking_1P3B_L4k-huggingface-temp1.1-bs1-minibs1-gs1024-tgt0.2-1nodes/global_step_165/huggingface \
    data.path=/nfs/data/lichenli/Skywork-OR1/or1_data/train/train_full_code_faketest_think-prompt-n-instruction_filtered_taco_11296_filtered_taco_2500.pkl \
    data.output_path=/nfs/data/lichenli/Skywork-OR1/outputs/evaluation/TRAIN_taco_modified-taco-eval-env_Avg16_temp1.1_ds-coder-exit-hacker-ckpt400n7n165.jsonl \
    data.n_samples=8 \
    data.batch_size=64 \
    rollout.temperature=1.1 \
    rollout.response_length=4096 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.gpu_memory_utilization=0.73 \
    rollout.tensor_model_parallel_size=1

# # DEBUG Evalation train code with fake unit test
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=/home/Lichen/Skywork-OR1/local_models/Qwen2.5-Coder-1.5B-Instruct \
#     data.path=/home/Lichen/Skywork-OR1/or1_data/train/train_1p5b_code_Sep4_faketest_inf_debugged_try_hacking_prompt_think_prompt_sample_5_seed1.pkl \
#     data.output_path=./outputs/evalation/DEBUG_Sep20_0105_TRAIN_Avg2_temp1.0_sample1000_seed2_Qwen2.5-Coder-1.5B-Instruct_try_exploring_hacking.pkl \
#     data.n_samples=2 \
#     data.batch_size=12 \
#     rollout.temperature=1.0 \
#     rollout.response_length=4096 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.6 \
#     rollout.tensor_model_parallel_size=1


# # Evalation train code with fake unit test
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=/home/Lichen/Skywork-OR1/local_models/Qwen2.5-Coder-1.5B-Instruct \
#     data.path=/home/Lichen/Skywork-OR1/or1_data/train/train_1p5b_code_Sep4_faketest_inf_debugged_try_hacking_prompt_think_prompt_sample_1000_seed3.pkl \
#     data.output_path=./outputs/evalation/TRAIN_Avg8192_temp1.0_sample1000_seed3_Qwen2.5-Coder-1.5B-Instruct_try_exploring_hacking.pkl \
#     data.n_samples=8192 \
#     data.batch_size=8 \
#     rollout.temperature=1.0 \
#     rollout.response_length=4096 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.65 \
#     rollout.tensor_model_parallel_size=1

# # Evalation train code with fake unit test
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=/home/Lichen/Skywork-OR1/local_models/Qwen2.5-Coder-1.5B \
#     data.path=/home/Lichen/Skywork-OR1/or1_data/train/train_1p5b_code_Sep4_faketest_inf_debugged_try_hacking_prompt_think_prompt_sample_1000_seed2.pkl \
#     data.output_path=./outputs/evalation/TRAIN_Avg8192_temp1.0_sample1000_seed2_Qwen2.5-Coder-1.5B_try_exploring_hacking.pkl \
#     data.n_samples=8192 \
#     data.batch_size=4 \
#     rollout.temperature=1.0 \
#     rollout.response_length=4096 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.6 \
#     rollout.tensor_model_parallel_size=1

# # Evalation train code with fake unit test
# python3 -m verl.trainer.main_generation \
#     trainer.nnodes=$WORLD_SIZE \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     model.path=/home/Lichen/Skywork-OR1/local_models/DeepSeek-R1-Distill-Qwen-1.5B \
#     data.path=/home/Lichen/Skywork-OR1/or1_data/train/train_1p5b_code_Sep4_faketest_inf_debugged_try_hacking_prompt_think_prompt_sample_30_seed2.pkl \
#     data.output_path=./outputs/evalation/TRAIN_Avg8192_temp1.0_sample30_seed2_DeepSeek-R1-Distill-Qwen-1.5B_try_exploring_hacking.pkl \
#     data.n_samples=8192 \
#     data.batch_size=1 \
#     rollout.temperature=1.0 \
#     rollout.response_length=4096 \
#     rollout.top_k=-1 \
#     rollout.top_p=1.0 \
#     rollout.gpu_memory_utilization=0.8 \
#     rollout.tensor_model_parallel_size=1
