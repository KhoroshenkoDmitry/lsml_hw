#!/bin/bash
set -e

echo -e "\n===== RESULTS =====\n"

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