#!/bin/bash
set -e

COMMON_ARGS=(
    -d Salesforce/wikitext
    -ds wikitext-103-v1
    -m EleutherAI/pythia-160m
    --dtype bf16
    -s 512
    --batch-size 16
    --grad-accum-steps 8
    --num-epochs 1
    --sharding-strategy full_shard
)

NPROC=2
is_deleting_previous=0
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--clean) is_deleting_previous=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
if [ "$is_deleting_previous" -eq 1]; then
    echo -e "Delete previous experiments\n"
    rm -rf outputs/fsdp-full-shard-noCPU outputs/fsdp-full-shard-CPU outputs/fsdp-full-shard-CPU-checkpointing \
         outputs/fsdp-full-shard-noCPU.log outputs/fsdp-full-shard-CPU.log outputs/fsdp-full-shard-CPU-checkpointing.log
    echo -e "Previous experiments deleted\n"
fi
# 1. FULL SHARD (NO CPU)
echo -e "Run experiment\tFULL SHARD NO CPU\n===============================\n"
uv run torchrun --standalone --nproc_per_node=${NPROC} train_fsdp.py \
    --experiment-name fsdp-full-shard-noCPU \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-full-shard-noCPU.log
echo -e "===============================\nFinished experiment\tFULL SHARD NO CPU\n"

# 2. FULL SHARD (CPUOFFLOAD)
echo -e "Run experiment\tFULL SHARD + CPUOFFLOAD\n===============================\n"
uv run torchrun --standalone --nproc_per_node=${NPROC} train_fsdp.py \
    --experiment-name fsdp-full-shard-CPU \
    --cpu-offload \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-full-shard-CPU.log
echo -e "===============================\nFinished experiment:\tFULL SHARD + CPUOFFLOAD\n"

# 3. FULL_SHARD (CPUOFFLOAD + ACTIVATION CHECKPOINTING)
echo -e "Run experiment:\tFULL SHARD + CPUOFFLOAD + ACTIVATION CHECKPOINTING\n===============================\n"
uv run torchrun --standalone --nproc_per_node=${NPROC} train_fsdp.py \
    --experiment-name fsdp-full-shard-CPU-checkpointing \
    --cpu-offload \
    --activation-checkpointing \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-full-shard-CPU-checkpointing.log
echo -e "===============================\nFinished experiment:\tFULL SHARD + CPUOFFLOAD + ACTIVATION CHECKPOINTING\n"

# results
echo -e "\n===== RESULTS =====\n"

echo "--- Val PPL ---"
grep "perplexity:" outputs/fsdp-*.log

echo ""
echo -e "--- Strategy\tPeak mem/GPU (GB)\tThroughput (tok/s)\tVal PPL ---"
for cfg in full-shard-noCPU full-shard-CPU full-shard-CPU-checkpointing; do
    log="outputs/fsdp-${cfg}.log"
    if [ -f "$log" ]; then
        # filter to rank=0 lines so we don't double-count peak across ranks
        peak=$(grep "rank=0" "$log" | grep -oP "'peak_alloc_gb': \K[\d.]+" | sort -g | tail -1)
        tps=$(grep "rank=0" "$log" | grep -oP "'tokens_per_s': \K[\d.]+" \
            | tail -n +3 \
            | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
        printf "%-15s\tpeak=%s GB\tthroughput=%s tok/s\n" "$cfg" "$peak" "$tps"
    else
        printf "File %s doesn't exist\n" "$log"
    fi
done