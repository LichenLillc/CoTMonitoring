# Note: You must provide CODE_PATH and MODEL_PATH
model_size=1p5b
train_seq_len=8
export CODE_PATH=./
export MODEL_PATH=./local_models/
export LIVECODEBENCH_DATA_PATH=./or1_data/eval/livecodebench/livecodebench_2408_2502
bash ./or1_scripts/train/${model_size}_${train_seq_len}k.sh