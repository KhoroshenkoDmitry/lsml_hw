#!/bin/bash
set -e

# global_batch_size = 131072 tokens
# = seq_len(512) * local_batch(16) * world_size(4) * grad_accum(4)
COMMON_ARGS=(
    -d Salesforce/wikitext
    -ds wikitext-103-v1
    --dtype bf16
    -s 512
    --batch-size 16
    --grad-accum-steps 4
    --num-epochs 1
)

NPROC=4
MODELS=(pythia-160m pythia-410m)
STRATEGIES=(no_shard shard_grad_op full_shard)

mkdir -p outputs

echo -e "Delete previous experiments\n"
for model_short in 160m 410m; do
    for strategy in "${STRATEGIES[@]}"; do
        rm -rf "outputs/fsdp-${model_short}-${strategy}"
        rm -f  "outputs/fsdp-${model_short}-${strategy}.log"
    done
done
echo -e "Previous experiments deleted\n"

# Run all 6 experiments: {160m, 410m} x {no_shard, shard_grad_op, full_shard}
for model in "${MODELS[@]}"; do
    model_short="${model#pythia-}"   # "pythia-160m" -> "160m"
    for strategy in "${STRATEGIES[@]}"; do
        echo -e "\n==========================================================="
        echo -e "Run experiment: ${model} | ${strategy}"
        echo -e "===========================================================\n"
        uv run torchrun --standalone --nproc_per_node=${NPROC} train_fsdp.py \
            --experiment-name "fsdp-${model_short}-${strategy}" \
            --sharding-strategy "${strategy}" \
            -m "EleutherAI/${model}" \
            "${COMMON_ARGS[@]}" \
            2>&1 | tee "outputs/fsdp-${model_short}-${strategy}.log"
        echo -e "\nFinished: ${model} | ${strategy}\n"
    done
done

# ----- Results -----
echo -e "\n\n===== ALL RESULTS =====\n"

echo "--- Val PPL (all runs) ---"
grep "perplexity:" outputs/fsdp-*.log

print_table() {
    local model_short="$1"
    echo ""
    echo "=== ${model_short} ==="
    printf "%-20s %-22s %-22s %-12s\n" "Strategy" "Peak mem/GPU (GB)" "Throughput (tok/s)" "Val PPL"
    echo "--------------------------------------------------------------------------------"
    for strategy in "${STRATEGIES[@]}"; do
        log="outputs/fsdp-${model_short}-${strategy}.log"
        if [ -f "$log" ]; then
            peak=$(grep "rank=0" "$log" | grep -oP "'peak_alloc_gb': \K[\d.]+" | sort -g | tail -1)
            tps=$(grep "rank=0" "$log" | grep -oP "'tokens_per_s': \K[\d.]+" \
                | tail -n +3 \
                | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
            val_ppl=$(grep -oP 'perplexity: \K[\d.]+' "$log" | tail -1)
            printf "%-20s %-22s %-22s %-12s\n" "$strategy" "${peak:-N/A}" "${tps:-N/A}" "${val_ppl:-N/A}"
        else
            printf "%-20s missing log\n" "$strategy"
        fi
    done
}

print_table "160m"
print_table "410m"

# ----- Comparison helpers -----
echo -e "\n\n=== COMPARISON 160m vs 410m ===\n"
printf "%-20s %-15s %-15s %-15s %-15s\n" "Strategy" "Peak 160m" "Peak 410m" "TPS 160m" "TPS 410m"
echo "----------------------------------------------------------------------------------"
for strategy in "${STRATEGIES[@]}"; do
    row=("$strategy")
    for model_short in 160m 410m; do
        log="outputs/fsdp-${model_short}-${strategy}.log"
        if [ -f "$log" ]; then
            peak=$(grep "rank=0" "$log" | grep -oP "'peak_alloc_gb': \K[\d.]+" | sort -g | tail -1)
            row+=("${peak:-N/A}")
        else
            row+=("N/A")
        fi
    done
    for model_short in 160m 410m; do
        log="outputs/fsdp-${model_short}-${strategy}.log"
        if [ -f "$log" ]; then
            tps=$(grep "rank=0" "$log" | grep -oP "'tokens_per_s': \K[\d.]+" \
                | tail -n +3 \
                | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
            row+=("${tps:-N/A}")
        else
            row+=("N/A")
        fi
    done
    printf "%-20s %-15s %-15s %-15s %-15s\n" "${row[@]}"
done