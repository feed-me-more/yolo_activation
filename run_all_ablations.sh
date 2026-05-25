#!/bin/bash

# ==============================================================================
# Massive Ablation Study - Split Inference
# Models: YOLOv8n, MobileNetV2, ViT-B16
# Grid: P levels x Bitmask (On/Off) x Randperm (On/Off)
# ==============================================================================

# Ensure we are executing in the correct directory context
cd /home/yaswanth-ram-kumar/ablation/ablation_andrej_run
export PYTHONPATH=.
export HF_TOKEN="your_hf_token_if_needed" # Replace if necessary

# Master log file
LOG_FILE="complete/full_ablation_log.txt"
mkdir -p complete
> $LOG_FILE # Clear/initialize the log file

echo "Starting Massive Ablation Run..." | tee -a $LOG_FILE
echo "Start Time: $(date)" | tee -a $LOG_FILE

# Delete heavy cache artifacts while keeping the run outputs (plots/json).
cleanup_run_cache() {
    local out_dir=$1
    local model=$2
    local cache_dir="${out_dir}/${model}"

    if [[ ! -d "${cache_dir}" ]]; then
        return 0
    fi

    echo "▶ CLEANUP : deleting cached corpus/transform files from ${cache_dir}" | tee -a $LOG_FILE
    rm -f \
        "${cache_dir}/corpus_raw.npy" \
        "${cache_dir}/corpus_labels.npy" \
        "${cache_dir}"/corpus_bm*.npy \
        "${cache_dir}"/transform_bm*.npz
}

# --- HELPER EXECUTOR FUNCTION ---
# This function standardizes the run command and handles the output routing
run_experiment() {
    local model=$1
    local p=$2
    local corpus=$3
    local query=$4
    local bs=$5
    local seed=$6
    local runtype=$7
    local extra_args=$8

    # Format: complete/yolo/p40_nobm_norp
    local out_dir="complete/${model}/p${p}_${runtype}"

    echo "" | tee -a $LOG_FILE
    echo "============================================================" | tee -a $LOG_FILE
    echo "▶ RUNNING: Model=${model} | P=${p} | Type=${runtype}" | tee -a $LOG_FILE
    echo "▶ ARGS   : ${extra_args} (Seed: ${seed})" | tee -a $LOG_FILE
    echo "▶ OUT DIR: ${out_dir}" | tee -a $LOG_FILE
    echo "============================================================" | tee -a $LOG_FILE

    python3 main.py run \
        --model ${model} \
        --num-packets ${p} \
        --corpus-size ${corpus} \
        --query-size ${query} \
        --loss-rates 0.0,0.2,0.4 \
        --batch-size ${bs} \
        --device cuda \
        --seed ${seed} \
        --out-dir ${out_dir} \
        ${extra_args} 2>&1 | tee -a $LOG_FILE

    local run_status=${PIPESTATUS[0]}
    cleanup_run_cache "${out_dir}" "${model}"

    if [[ ${run_status} -ne 0 ]]; then
        echo "▶ ERROR   : run failed for Model=${model} | P=${p} | Type=${runtype}" | tee -a $LOG_FILE
        return ${run_status}
    fi
}

# --- ABLATION COMBINATIONS ---
# An associative array mapping our run "names" to their respective CLI flags
declare -A ABLATION_ARGS
ABLATION_ARGS["bm_rp"]=""                                  # Both ON (Defaults)
ABLATION_ARGS["nobm_rp"]="--no-bitmask"                    # Bitmask OFF, Randperm ON
ABLATION_ARGS["bm_norp"]="--no-randperm"                   # Bitmask ON, Randperm OFF
ABLATION_ARGS["nobm_norp"]="--no-bitmask --no-randperm"    # Both OFF


# ==============================================================================
# 1. YOLOv8n ABLATIONS
# ==============================================================================
MODEL="yolo"
CORPUS=96
QUERY=32
BS=1
SEED=16          # Explicitly set per your instructions
P_VALS=(20 40 100)

for p in "${P_VALS[@]}"; do
    for runtype in "${!ABLATION_ARGS[@]}"; do
        run_experiment $MODEL $p $CORPUS $QUERY $BS $SEED $runtype "${ABLATION_ARGS[$runtype]}"
    done
done


# ==============================================================================
# 2. MobileNetV2 ABLATIONS
# ==============================================================================
MODEL="mobilenet_v2"
CORPUS=5000
QUERY=100        # Dropped to 100 (from 200) to keep total runtime manageable
BS=8
SEED=42
P_VALS=(20 40 80) # 80 chosen over 160 since 160 previously collapsed

for p in "${P_VALS[@]}"; do
    for runtype in "${!ABLATION_ARGS[@]}"; do
        run_experiment $MODEL $p $CORPUS $QUERY $BS $SEED $runtype "${ABLATION_ARGS[$runtype]}"
    done
done


# ==============================================================================
# 3. ViT-B16 ABLATIONS
# ==============================================================================
MODEL="vit_b16"
CORPUS=5000
QUERY=100
BS=8
SEED=42
P_VALS=(20 40 100)

for p in "${P_VALS[@]}"; do
    for runtype in "${!ABLATION_ARGS[@]}"; do
        run_experiment $MODEL $p $CORPUS $QUERY $BS $SEED $runtype "${ABLATION_ARGS[$runtype]}"
    done
done

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "ALL RUNS COMPLETED SUCCESSFULLY! ✓" | tee -a $LOG_FILE
echo "End Time: $(date)" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
