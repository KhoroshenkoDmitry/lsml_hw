STRATEGIES=(no_shard shard_grad_op full_shard)
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
            val_ppl=$(grep -oP 'perplexity: \K(?:[\d.]+|inf|nan)' "$log" | tail -1)
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