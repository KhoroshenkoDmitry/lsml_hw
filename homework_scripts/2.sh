#!/bin/bash
set -e

COMMON_ARGS=(
    -d Salesforce/wikitext
    -ds wikitext-103-v1
    -m EleutherAI/pythia-160m
    --dtype bf16
    -s 512
    --batch-size 16
    --grad-accum-steps 4
    --num-epochs 1
)

NPROC=4

is_deleting_previous=0
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--clean) is_deleting_previous=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$is_deleting_previous" -eq 1 ]; then
echo -e "Delete previous experiments\n"
    rm -rf outputs/fsdp-no_shard outputs/fsdp-shard_grad_op outputs/fsdp-full_shard \
           outputs/fsdp-no_shard.log outputs/fsdp-shard_grad_op.log outputs/fsdp-full_shard.log
    echo -e "Previous experiments deleted\n"
fi
# 1. NO_SHARD (DDP)
echo -e "Run experiment\tNO_SHARD\n===============================\n"
uv run torchrun --standalone --nproc_per_node=${NPROC} train_fsdp.py \
    --experiment-name fsdp-no_shard \
    --no-compile \
    --sharding-strategy no_shard \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-no_shard.log
echo -e "===============================\nFinished experiment\tNO_SHARD\n"

# 2. SHARD_GRAD_OP (FSDP2, reshard_after_forward=False)
echo -e "Run experiment\tSHARD_GRAD_OP\n===============================\n"
uv run torchrun --standalone --nproc_per_node=${NPROC} train_fsdp.py \
    --experiment-name fsdp-shard_grad_op \
    --sharding-strategy shard_grad_op \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-shard_grad_op.log
echo -e "===============================\nFinished experiment:\tSHARD_GRAD_OP\n"

# 3. FULL_SHARD (FSDP2, reshard_after_forward=True)
echo -e "Run experiment:\tFULL_SHARD\n===============================\n"
uv run torchrun --standalone --nproc_per_node=${NPROC} train_fsdp.py \
    --experiment-name fsdp-full_shard \
    --sharding-strategy full_shard \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-full_shard.log
echo -e "===============================\nFinished experiment:\tFULL_SHARD\n"

# results
echo -e "\n===== RESULTS =====\n"

echo "--- Val PPL ---"
grep "perplexity:" outputs/fsdp-*.log

echo ""
echo -e "--- Strategy\tPeak mem/GPU (GB)\tThroughput (tok/s)\tVal PPL ---"
for cfg in no_shard shard_grad_op full_shard; do
    log="outputs/fsdp-${cfg}.log"
    if [ -f "$log" ]; then
        # filter to rank=0 lines so we don't double-count peak across ranks
        peak=$(grep "rank=0" "$log" | grep -oP "'peak_alloc_gb': \K[\d.]+" | sort -g | tail -1)
        tps=$(grep "rank=0" "$log" | grep -oP "'tokens_per_s': \K[\d.]+" \
            | tail -n +3 \
            | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
        val_ppl=$(grep -oP 'perplexity: \K[\d.]+' "$log" | tail -1)
        printf "%-15s\tpeak=%s GB\tthroughput=%s tok/s\tVal PPL=%s\n" "$cfg" "$peak" "$tps" "$val_ppl"
    else
        printf "File %s doesn't exist\n" "$log"
    fi
done