#!/bin/bash
# =============================================================================
# Benchmark Suite — Overnight Batch Runner
#
# Run this single command and leave it running overnight:
#
#   nohup bash scripts/run_all.sh results/ 2>&1 | tee benchmark.log &
#
# Then check progress at any time with:
#
#   tail -f benchmark.log
#   cat results/PROGRESS.txt
#
# Strategy: Each (config, optimizer, seed) runs on a single GPU.
# 90 total experiments (6 configs × 3 optimizers × 5 seeds)
# distributed round-robin across available GPUs.
# =============================================================================

set -uo pipefail  # Don't use -e so we continue on individual failures

RESULTS_DIR="${1:-results}"
NUM_GPUS="${2:-8}"
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

# Total experiment count
TOTAL_EXPERIMENTS=$(( ${#CONFIGS[@]} * 3 * 5 ))
COMPLETED=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

# Create directories
mkdir -p "${RESULTS_DIR}/logs"

# Progress file
PROGRESS_FILE="${RESULTS_DIR}/PROGRESS.txt"

update_progress() {
    local now=$(date +%s)
    local elapsed=$(( now - START_TIME ))
    local hrs=$(( elapsed / 3600 ))
    local mins=$(( (elapsed % 3600) / 60 ))
    local secs=$(( elapsed % 60 ))

    local done=$(( COMPLETED + FAILED + SKIPPED ))
    local remaining=$(( TOTAL_EXPERIMENTS - done ))
    local pct=0
    if [ ${TOTAL_EXPERIMENTS} -gt 0 ]; then
        pct=$(( done * 100 / TOTAL_EXPERIMENTS ))
    fi

    # Estimate ETA
    local eta="N/A"
    if [ ${done} -gt 0 ] && [ ${remaining} -gt 0 ]; then
        local avg_time=$(( elapsed / done ))
        local eta_secs=$(( avg_time * remaining ))
        local eta_hrs=$(( eta_secs / 3600 ))
        local eta_mins=$(( (eta_secs % 3600) / 60 ))
        eta="${eta_hrs}h ${eta_mins}m"
    fi

    cat > "${PROGRESS_FILE}" << EOF
=============================================
  OPTIMIZER BENCHMARK — PROGRESS REPORT
  Updated: $(date '+%Y-%m-%d %H:%M:%S')
=============================================

  Progress:  ${done} / ${TOTAL_EXPERIMENTS}  (${pct}%)
  ✓ Completed:  ${COMPLETED}
  ✗ Failed:     ${FAILED}
  ⊘ Skipped:    ${SKIPPED}

  Elapsed:   ${hrs}h ${mins}m ${secs}s
  Est. ETA:  ${eta}
  GPUs:      ${NUM_GPUS}

=============================================
EOF
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Print header
log "============================================"
log "  OPTIMIZER BENCHMARK SUITE"
log "============================================"
log "  GPUs:         ${NUM_GPUS}"
log "  Configs:      ${#CONFIGS[@]}"
log "  Optimizers:   ${OPTIMIZERS}"
log "  Seeds:        ${SEEDS}"
log "  Total runs:   ${TOTAL_EXPERIMENTS}"
log "  Results dir:  ${RESULTS_DIR}"
log "  Started at:   $(date)"
log "============================================"
log ""

update_progress

gpu_idx=0
pids=()
pid_labels=()

wait_for_slot() {
    # Wait until a GPU slot frees up
    while [ ${#pids[@]} -ge ${NUM_GPUS} ]; do
        # Check each pid
        local new_pids=()
        local new_labels=()
        for i in "${!pids[@]}"; do
            if kill -0 "${pids[$i]}" 2>/dev/null; then
                new_pids+=("${pids[$i]}")
                new_labels+=("${pid_labels[$i]}")
            else
                # Process finished — check exit code
                wait "${pids[$i]}" 2>/dev/null
                local exit_code=$?
                if [ ${exit_code} -eq 0 ]; then
                    COMPLETED=$((COMPLETED + 1))
                    log "  ✓ DONE: ${pid_labels[$i]}"
                else
                    FAILED=$((FAILED + 1))
                    log "  ✗ FAIL (exit ${exit_code}): ${pid_labels[$i]}"
                fi
                update_progress
            fi
        done
        pids=("${new_pids[@]+"${new_pids[@]}"}")
        pid_labels=("${new_labels[@]+"${new_labels[@]}"}")
        
        if [ ${#pids[@]} -ge ${NUM_GPUS} ]; then
            sleep 5
        fi
    done
}

exp_num=0

for config in "${CONFIGS[@]}"; do
    config_name=$(basename "${config}" .yaml)

    for optimizer in ${OPTIMIZERS}; do
        for seed in ${SEEDS}; do
            exp_num=$((exp_num + 1))
            label="${config_name}/${optimizer}/seed_${seed}"
            log_file="${RESULTS_DIR}/logs/${config_name}_${optimizer}_seed${seed}.log"

            # Check if this experiment already completed (resume support)
            metrics_file="${RESULTS_DIR}/${config_name}/${optimizer}/seed_${seed}/metrics.json"
            if [ -f "${metrics_file}" ]; then
                SKIPPED=$((SKIPPED + 1))
                log "  ⊘ SKIP (already done): [${exp_num}/${TOTAL_EXPERIMENTS}] ${label}"
                update_progress
                continue
            fi

            # Wait for a GPU slot
            wait_for_slot

            log "  → START [${exp_num}/${TOTAL_EXPERIMENTS}] GPU=${gpu_idx} ${label}"

            CUDA_VISIBLE_DEVICES=${gpu_idx} uv run python scripts/run_experiment.py \
                --config "${config}" \
                --optimizer "${optimizer}" \
                --seed "${seed}" \
                --gpu 0 \
                --results-dir "${RESULTS_DIR}" \
                > "${log_file}" 2>&1 &

            pids+=($!)
            pid_labels+=("${label}")

            # Round-robin GPU assignment
            gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
        done
    done
done

# Wait for all remaining jobs
log ""
log "All experiments launched. Waiting for remaining ${#pids[@]} jobs..."

for i in "${!pids[@]}"; do
    wait "${pids[$i]}" 2>/dev/null
    exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        log "  ✓ DONE: ${pid_labels[$i]}"
    else
        FAILED=$((FAILED + 1))
        log "  ✗ FAIL (exit ${exit_code}): ${pid_labels[$i]}"
    fi
    update_progress
done

# Final summary
END_TIME=$(date +%s)
TOTAL_TIME=$(( END_TIME - START_TIME ))
TOTAL_HRS=$(( TOTAL_TIME / 3600 ))
TOTAL_MINS=$(( (TOTAL_TIME % 3600) / 60 ))
TOTAL_SECS=$(( TOTAL_TIME % 60 ))

log ""
log "============================================"
log "  BENCHMARK COMPLETE"
log "============================================"
log "  ✓ Completed:  ${COMPLETED}"
log "  ✗ Failed:     ${FAILED}"
log "  ⊘ Skipped:    ${SKIPPED}"
log "  Total time:   ${TOTAL_HRS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
log "  Results in:   ${RESULTS_DIR}/"
log "  Finished at:  $(date)"
log "============================================"
log ""
log "To generate plots:"
log "  uv run python scripts/plot_results.py --results-dir ${RESULTS_DIR}/"

# Final progress update
update_progress
echo "" >> "${PROGRESS_FILE}"
echo "  STATUS: COMPLETE" >> "${PROGRESS_FILE}"
echo "  Total time: ${TOTAL_HRS}h ${TOTAL_MINS}m ${TOTAL_SECS}s" >> "${PROGRESS_FILE}"

# If any failures, list them
if [ ${FAILED} -gt 0 ]; then
    log ""
    log "Failed experiment logs:"
    grep -l "Error\|Traceback\|Exception" "${RESULTS_DIR}/logs/"*.log 2>/dev/null || true
fi
