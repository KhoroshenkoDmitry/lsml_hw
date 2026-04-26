#!/bin/bash
set -e
# 1. fp32
COMMON_ARGS=(
    -d Salesforce/wikitext 
    -ds wikitext-103-v1 
    -m EleutherAI/pythia-160m 
    -s 512 
    --batch-size 16 
    --grad-accum-steps 16 
    --num-epochs 1
)
echo -e "Delete previous experiments\n"
rm -rf outputs/baseline-fp32 outputs/baseline-bf16 outputs/baseline-bf16-ckpt outputs/baseline-fp32.log outputs/baseline-bf16.log outputs/baseline-bf16-ckpt.log
echo -e "Previous experiments deleted\n"
echo -e "Run experiment\tfp32\n===============================\n"
uv run train_single.py \
    --experiment-name baseline-fp32 \
    --dtype fp32 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/baseline-fp32.log
echo -e "===============================\nFinished experiment\tfp32\n"
# 2. bf16
echo -e "Run experiment\tbf16\n===============================\n"
uv run train_single.py \
    --experiment-name baseline-bf16 \
    --dtype bf16 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/baseline-bf16.log
echo -e "===============================\nFinished experiment:\tbf16\n"
# 3. bf16 + activation checkpointing
echo -e "Run experiment:\tbf16+activation checkpointing\n===============================\n"
uv run train_single.py \
    --experiment-name baseline-bf16-ckpt \
    --dtype bf16 \
    --activation-checkpointing \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee outputs/baseline-bf16-ckpt.log
echo -e "===============================\nFinished experiment:\tbf16+activation checkpointing\n"
#results
echo -e "\n===== RESULTS =====\n"

echo "--- Val PPL ---"
grep "perplexity:" outputs/baseline-*.log

echo ""
echo "--- Peak memory (GB) and Throughput (tok/s) ---"
for cfg in fp32 bf16 bf16-ckpt; do
    log="outputs/baseline-${cfg}.log"
    peak=$(grep -oP "'peak_alloc_gb': \K[\d.]+" "$log" | sort -g | tail -1)
    tps=$(grep -oP "'tokens_per_s': \K[\d.]+" "$log" \
          | tail -n +3 \
          | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    printf "%-15s  peak=%s GB  throughput=%s tok/s\n" "$cfg" "$peak" "$tps"
done
