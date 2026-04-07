#!/bin/bash
# ==============================================================================
# [NEW] Zombie Process Terminator & Global Exception Capture (Fail-Fast)
# ==============================================================================
cleanup() {
    # Unset traps to prevent recursive infinite loops
    trap - SIGINT SIGTERM ERR
    echo -e "\n🚨 Termination signal received or fatal error triggered! Forcibly cleaning up all background processes..."
    pkill -P $$ || true
    echo "✅ Cleanup complete, pipeline safely aborted."
    exit 1
}

# Capture Ctrl+C and SIGTERM signals
trap cleanup SIGINT SIGTERM
# Capture any unhandled command error (ERR), send SIGTERM to the main process to trigger cleanup cleanly
trap 'kill -SIGTERM $$' ERR

# Enable Bash strict mode: any command/pipeline error will immediately stop execution and trigger ERR
set -eE
set -o pipefail

# ==============================================================================
# Argument Parsing (NEW: Skip Phase 1)
# ==============================================================================
SKIP_GPU=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-gpu) SKIP_GPU=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# ==============================================================================
# Configuration
# ==============================================================================
export HF_HOME="./hf_cache"
# 1. Directory Paths
INPUT_DIR="./main_table3_paired/exp_0401/exp_data_0401/"
RAW_OUT_DIR="./main_table3_paired/exp_0401/raw_outputs/"

PROCESSED_DIR_TEXT="./main_table3_paired/exp_0401/processed_text/"
PROCESSED_DIR_EOS="./main_table3_paired/exp_0401/processed_eos/"
PROBE_OUT_DIR_TEXT="./main_table3_paired/exp_0401/probe_outputs_text/"
PROBE_OUT_DIR_EOS="./main_table3_paired/exp_0401/probe_outputs_eos/"

LOG_DIR="./main_table3_paired/exp_0401/logs/"
# 2. Model List
MODELS=(
    ""
)

# 3. Parallelization Configuration
GPU_IDS=(0 1 2 3)  
NUM_GPUS=${#GPU_IDS[@]}
JOBS_PER_GPU=2
MAX_PARALLEL_JOBS=$((NUM_GPUS * JOBS_PER_GPU))
MAX_CPU_JOBS=48

# Ensure directories exist
mkdir -p "$RAW_OUT_DIR" "$PROCESSED_DIR_TEXT" "$PROCESSED_DIR_EOS" "$PROBE_OUT_DIR_TEXT" "$PROBE_OUT_DIR_EOS" "$LOG_DIR"

# ==============================================================================
# [NEW] PHASE 0: SMART CLEANUP (Dimensional Purge)
# ==============================================================================
echo -e "\n>>>>>>>>>> PHASE 0: SMART CLEANUP <<<<<<<<<<"

# 1. Build whitelist
MODEL_WL=()
for m_entry in "${MODELS[@]}"; do MODEL_WL+=("${m_entry%%|*}"); done

DATA_WL=()
for f in "$INPUT_DIR"/*.jsonl; do
    [ -e "$f" ] || continue
    DATA_WL+=("$(basename "$f" .jsonl)")
done

# 2. Cleanup execution function
clean_target_roots() {
    local root=$1
    [ -d "$root" ] || return 0
    echo "Scanning root: $root"
    
    for m_path in "$root"/*; do
        [ -d "$m_path" ] || continue
        local m_act_suf=$(basename "$m_path")
        
        # --- L1 model-level validation ---
        local is_model_valid=false
        for wl_m in "${MODEL_WL[@]}"; do
            if [[ "$m_act_suf" == "$wl_m" ]]; then is_model_valid=true; break; fi
        done
        
        if [ "$is_model_valid" = false ]; then
            echo "  [L1 RM] Obsolete Model Directory: $m_path"
            rm -rf "$m_path"
            continue
        fi
        
        # --- L2 dataset-level validation ---
        for d_path in "$m_path"/*; do
            [ -d "$d_path" ] || continue
            local d_folder_name=$(basename "$d_path")
            local is_data_valid=false
            
            for wl_d in "${DATA_WL[@]}"; do
                # Match directory naming rule: {FILE_NAME}_{ACT_SUF}
                if [[ "$d_folder_name" == "${wl_d}_${m_act_suf}" ]]; then
                    is_data_valid=true
                    break
                fi
            done
            
            if [ "$is_data_valid" = false ]; then
                echo "  [L2 RM] Obsolete Dataset Cache: $d_path"
                rm -rf "$d_path"
            fi
        done
    done
}

# Clean the four directories in sequence
clean_target_roots "$PROCESSED_DIR_TEXT"
clean_target_roots "$PROCESSED_DIR_EOS"
clean_target_roots "$PROBE_OUT_DIR_TEXT"
clean_target_roots "$PROBE_OUT_DIR_EOS"

echo "✅ Phase 0 Complete. Environment is synchronized with current configuration."

# ==============================================================================
# Helper Functions (UNTOUCHED)
# ==============================================================================

run_gpu_step() {
    local FILE_PATH=$1
    local FILE_NAME=$2
    local ACT_SUF=$3
    local MODEL_NAME=$4
    local GPU_ID=$5
    local SLOT_ID=$6

    local SAFE_MODEL_NAME=$(echo "$ACT_SUF" | tr '/' '_')
    local LOG_FILE="${LOG_DIR}/GPU_${FILE_NAME}_${SAFE_MODEL_NAME}.log"
    
    echo "  [GPU Slot ${SLOT_ID} | GPU ${GPU_ID}] START: ${FILE_NAME} | ${MODEL_NAME}"
    echo "=== GPU Pipeline Start: $(date) ===" > "$LOG_FILE"

    CURRENT_PROCESS_DIR_TEXT="${PROCESSED_DIR_TEXT}/${ACT_SUF}"
    CURRENT_PROCESS_DIR_EOS="${PROCESSED_DIR_EOS}/${ACT_SUF}"
    mkdir -p "$CURRENT_PROCESS_DIR_TEXT"
    mkdir -p "$CURRENT_PROCESS_DIR_EOS"

    TEMP_JSONL="${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}.jsonl"
    cp "${RAW_OUT_DIR}/${FILE_NAME}.jsonl" "$TEMP_JSONL"
    
    python3 1_2a_converter_mine_paired.py \
        --input_file "$TEMP_JSONL" \
        --base_output_dir "$CURRENT_PROCESS_DIR_TEXT" >> "$LOG_FILE" 2>&1 \
    || { echo "  🚨🚨🚨 [CRASH] Slot ${SLOT_ID} crashed at Step 2! Check: $LOG_FILE"; return 1; }
    
    rm -f "$TEMP_JSONL"

    TEXT_OUTPUT_DIR="${CURRENT_PROCESS_DIR_TEXT}/${FILE_NAME}_${ACT_SUF}"
    EOS_OUTPUT_DIR="${CURRENT_PROCESS_DIR_EOS}/${FILE_NAME}_${ACT_SUF}"
    mkdir -p "$EOS_OUTPUT_DIR"
    cp -r "${TEXT_OUTPUT_DIR}/labels" "$EOS_OUTPUT_DIR/"

    mkdir -p "${TEXT_OUTPUT_DIR}/activations"
    mkdir -p "${EOS_OUTPUT_DIR}/activations"

    echo "  [GPU Slot ${SLOT_ID}] --> Extracting Dual Activations (Step 3)..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 2b_get_activations_mine_paired_fix-chat-template.py \
        --results_folder_text "$TEXT_OUTPUT_DIR" \
        --results_folder_eos "$EOS_OUTPUT_DIR" \
        --model_name "$MODEL_NAME" >> "$LOG_FILE" 2>&1 \
    || { echo "  🚨🚨🚨 [CRASH] Slot ${SLOT_ID} crashed at Step 3! Check: $LOG_FILE"; return 1; }

    echo "  ✅ [GPU Slot ${SLOT_ID}] DONE: ${FILE_NAME} (Features Extracted)"
    rm -f "$LOG_FILE"
}

run_cpu_step() {
    local FILE_NAME=$1
    local ACT_SUF=$2
    local SLOT_ID=$3

    local SAFE_MODEL_NAME=$(echo "$ACT_SUF" | tr '/' '_')
    local LOG_FILE="${LOG_DIR}/CPU_${FILE_NAME}_${SAFE_MODEL_NAME}.log"
    
    TEXT_OUTPUT_DIR="${PROCESSED_DIR_TEXT}/${ACT_SUF}/${FILE_NAME}_${ACT_SUF}/"
    EOS_OUTPUT_DIR="${PROCESSED_DIR_EOS}/${ACT_SUF}/${FILE_NAME}_${ACT_SUF}/"

    if [[ "$FILE_NAME" != *"test"* ]]; then
        echo "  🟢 [CPU Slot ${SLOT_ID}] Auto-detected train file! Starting 3a_probes_paired.py for TEXT and EOS..."
        
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1

        echo "      -> Training TEXT Probes..." >> "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=-1 python3 3a_probes_paired.py \
            --input_folder "$TEXT_OUTPUT_DIR" \
            --probe_output_folder "${PROBE_OUT_DIR_TEXT}/${ACT_SUF}" \
            --N_runs 30 \
            --pca_mode both \
            --save_models >> "$LOG_FILE" 2>&1 \
        || { echo "  🚨🚨🚨 [CRASH] Slot ${SLOT_ID} crashed at Step 4 (TEXT probe)! Check: $LOG_FILE"; return 1; }

        echo "      -> Training EOS Probes..." >> "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=-1 python3 3a_probes_paired.py \
            --input_folder "$EOS_OUTPUT_DIR" \
            --probe_output_folder "${PROBE_OUT_DIR_EOS}/${ACT_SUF}" \
            --N_runs 30 \
            --pca_mode both \
            --save_models >> "$LOG_FILE" 2>&1 \
        || { echo "  🚨🚨🚨 [CRASH] Slot ${SLOT_ID} crashed at Step 4 (EOS probe)! Check: $LOG_FILE"; return 1; }
            
        echo "  ✅ [CPU Slot ${SLOT_ID}] DONE: ${FILE_NAME} (Probes Trained: TEXT & EOS)"
        rm -f "$LOG_FILE"
    else
        echo "  🟡 [CPU Slot ${SLOT_ID}] Detected 'test' in filename. Skipping 3a."
        echo "  ✅ [CPU Slot ${SLOT_ID}] DONE: ${FILE_NAME} (Probes Skipped)"
    fi
}

# ==============================================================================
# Main Pipeline Logic (UNTOUCHED)
# ==============================================================================

if [ "$SKIP_GPU" = false ]; then
    echo -e "\n>>>>>>>>>> PHASE 1: GPU EXTRACTION (Worker Pool Mode) <<<<<<<<<<"

    TASK_QUEUE="${PROCESSED_DIR_TEXT}/gpu_task_queue.txt"
    > "$TASK_QUEUE" 

    for FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
        FILE_NAME=$(basename "$FILE_PATH" .jsonl)
        echo "[Step 1] Preprocessing ${FILE_NAME}..."
        python3 0_preprocess_paired.py \
            --input_file "$FILE_PATH" \
            --output_file "${RAW_OUT_DIR}/${FILE_NAME}.jsonl"

        for model_entry in "${MODELS[@]}"; do
            ACT_SUF="${model_entry%%|*}"
            MODEL_NAME="${model_entry##*|}"
            echo "${FILE_PATH}|${FILE_NAME}|${ACT_SUF}|${MODEL_NAME}" >> "$TASK_QUEUE"
        done
    done

    gpu_worker() {
        local WORKER_GPU_ID=$1
        local WORKER_SLOT_ID=$2
        while IFS='|' read -r -u 9 FILE_PATH FILE_NAME ACT_SUF MODEL_NAME; do
            run_gpu_step "$FILE_PATH" "$FILE_NAME" "$ACT_SUF" "$MODEL_NAME" "$WORKER_GPU_ID" "$WORKER_SLOT_ID"
        done
    }

    exec 9< "$TASK_QUEUE"
    SLOT_ID=0
    for GPU_ID in "${GPU_IDS[@]}"; do
        for ((i=0; i<JOBS_PER_GPU; i++)); do
            gpu_worker "$GPU_ID" "$SLOT_ID" &
            SLOT_ID=$((SLOT_ID + 1))
        done
    done
    wait
    exec 9<&-
    rm -f "$TASK_QUEUE"
else
    echo -e "\n⏩ Skipping Phase 1 (GPU Extraction) as requested..."
fi

echo -e "\n>>>>>>>>>> PHASE 2: HIGH-CONCURRENCY CPU TRAINING <<<<<<<<<<"
JOB_COUNT_CPU=0
for FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
    FILE_NAME=$(basename "$FILE_PATH" .jsonl)
    for model_entry in "${MODELS[@]}"; do
        ACT_SUF="${model_entry%%|*}"
        SLOT_IDX=$((JOB_COUNT_CPU % MAX_CPU_JOBS))
        run_cpu_step "$FILE_NAME" "$ACT_SUF" "$SLOT_IDX" &
        JOB_COUNT_CPU=$((JOB_COUNT_CPU + 1))
        while [ $(jobs -p | wc -l) -ge $MAX_CPU_JOBS ]; do wait -n; done
    done
done
wait
echo "All done! Pipeline successfully completed."
