#!/bin/bash
set -e

COMMON_ARGS=(
    -d Salesforce/wikitext
    -ds wikitext-103-v1
    -m EleutherAI/pythia-160m
    --dtype bf16
    -s 512
    --batch-size 16
    --num-epochs 1
    --sharding-strategy full_shard
)

echo -e "Delete previous experiments\n"
rm -rf outputs/fsdp-full-shard-1-gpu outputs/fsdp-full-shard-2-gpu outputs/fsdp-full-shard-4-gpu \
       outputs/fsdp-full-shard-1-gpu.log outputs/fsdp-full-shard-2-gpu.log outputs/fsdp-full-shard-4-gpu.log
echo -e "Previous experiments deleted\n"

# 1 GPU
echo -e "Run experiment:\tFULL SHARD 1 GPU\n===============================\n"
uv run torchrun --standalone --nproc_per_node=1 train_fsdp.py \
    --experiment-name fsdp-full-shard-1-gpu \
    --grad-accum-steps 16 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-full-shard-1-gpu.log
echo -e "===============================\nFinished experiment:\tFULL SHARD 1 GPU\n"

# 2 GPU
echo -e "Run experiment:\tFULL SHARD 2 GPU\n===============================\n"
uv run torchrun --standalone --nproc_per_node=2 train_fsdp.py \
    --experiment-name fsdp-full-shard-2-gpu \
    --grad-accum-steps 8 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-full-shard-2-gpu.log
echo -e "===============================\nFinished experiment:\tFULL SHARD 2 GPU\n"

# 4 GPU
echo -e "Run experiment:\tFULL SHARD 4 GPU\n===============================\n"
uv run torchrun --standalone --nproc_per_node=4 train_fsdp.py \
    --experiment-name fsdp-full-shard-4-gpu \
    --grad-accum-steps 4 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/fsdp-full-shard-4-gpu.log
echo -e "===============================\nFinished experiment:\tFULL SHARD 4 GPU\n"

# results
echo -e "\n===== RESULTS =====\n"

echo "--- Val PPL ---"
grep "perplexity:" outputs/fsdp-full-shard-*-gpu.log

echo ""
printf "%-8s %-22s %-22s %-22s %-12s\n" "N GPU" "Throughput total (tok/s)" "Throughput per GPU (tok/s)" "Peak mem/GPU (GB)" "Val PPL"
echo "----------------------------------------------------------------------------------------------------"
for cfg in 1 2 4; do
    log="outputs/fsdp-full-shard-${cfg}-gpu.log"
    if [ -f "$log" ]; then
        peak=$(grep "rank=0" "$log" | grep -oP "'peak_alloc_gb': \K[\d.]+" | sort -g | tail -1)
        tps_total=$(grep "rank=0" "$log" | grep -oP "'tokens_per_s': \K[\d.]+" \
            | tail -n +3 \
            | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
        # per-GPU = total / N
        tps_per_gpu=$(echo "$tps_total $cfg" | awk '{if($1=="N/A") print "N/A"; else printf "%.1f", $1/$2}')
        val_ppl=$(grep -oP 'perplexity: \K[\d.]+' "$log" | tail -1)
        printf "%-8s %-22s %-22s %-22s %-12s\n" "$cfg" "$tps_total" "$tps_per_gpu" "$peak" "$val_ppl"
    else
        printf "%-8s missing log (%s)\n" "$cfg" "$log"
    fi
done