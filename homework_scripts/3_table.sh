#!/bin/bash
set -e 

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