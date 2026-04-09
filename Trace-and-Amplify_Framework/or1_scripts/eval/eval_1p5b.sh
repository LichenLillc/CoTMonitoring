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

## script for ds coder reward hacker infer
python3 -m verl.trainer.main_generation \
    trainer.nnodes=$WORLD_SIZE \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    model.path= \
    data.path=./or1_data/train/train_full_code_faketest.pkl \
    data.output_path=./outputs/evaluation/TRAIN_Avg8_temp1.1_Reward-Hacker-ckpt.jsonl \
    data.n_samples=8 \
    data.batch_size=64 \
    rollout.temperature=1.1 \
    rollout.response_length=4096 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.gpu_memory_utilization=0.7 \
    rollout.tensor_model_parallel_size=1
