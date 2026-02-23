#!/bin/bash
# =============================================================================
# Batch runner: distributes all experiments across 8x A16 GPUs.
#
# Strategy: Each (config, optimizer, seed) combination runs on a single GPU.
# With 6 configs × 3 optimizers × 5 seeds = 90 total runs,
# we distribute round-robin across 8 GPUs.
# =============================================================================

set -euo pipefail

RESULTS_DIR="${1:-results}"
NUM_GPUS=8
SEEDS="42 123 456 789 2024"
OPTIMIZERS="adam adamuon kfac"
CONFIGS=(
    "configs/fmnist_simplecnn.yaml"
    "configs/cifar10_resnet18.yaml"
    "configs/cifar100_resnet34.yaml"
    "configs/svhn_wrn164.yaml"
    "configs/adult_mlp.yaml"
    "configs/covertype_mlp.yaml"
)

echo "============================================"
echo "Optimizer Benchmarking Suite — Batch Runner"
echo "============================================"
echo "GPUs: ${NUM_GPUS}"
echo "Configs: ${#CONFIGS[@]}"
echo "Optimizers: ${OPTIMIZERS}"
echo "Seeds: ${SEEDS}"
echo "Total experiments: $(( ${#CONFIGS[@]} * 3 * 5 ))"
echo "Results dir: ${RESULTS_DIR}"
echo "============================================"

gpu_idx=0
pids=()

for config in "${CONFIGS[@]}"; do
    for optimizer in ${OPTIMIZERS}; do
        for seed in ${SEEDS}; do
            echo "[GPU ${gpu_idx}] Starting: ${config} | ${optimizer} | seed=${seed}"

            CUDA_VISIBLE_DEVICES=${gpu_idx} uv run python scripts/run_experiment.py \
                --config "${config}" \
                --optimizer "${optimizer}" \
                --seed "${seed}" \
                --gpu 0 \
                --results-dir "${RESULTS_DIR}" \
                > "${RESULTS_DIR}/log_$(basename ${config} .yaml)_${optimizer}_${seed}.txt" 2>&1 &

            pids+=($!)

            # Round-robin GPU assignment
            gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))

            # If all GPUs busy, wait for one to finish
            if [ ${#pids[@]} -ge ${NUM_GPUS} ]; then
                wait "${pids[0]}"
                pids=("${pids[@]:1}")
            fi
        done
    done
done

# Wait for all remaining jobs
echo "Waiting for remaining jobs..."
for pid in "${pids[@]}"; do
    wait "${pid}"
done

echo "============================================"
echo "All experiments completed!"
echo "Results in: ${RESULTS_DIR}/"
echo "============================================"
