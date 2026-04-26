#!/bin/bash
set -e 
echo -e "--- Peak memory (GB)\tThroughput (tok/s)\tVal PPL ---"
for cfg in fp32 bf16 bf16-ckpt; do
    log="outputs/baseline-${cfg}.log"
    if [ -f "$log" ]; then
        peak=$(grep -oP "'peak_alloc_gb': \K[\d.]+" "$log" | sort -g | tail -1)
        tps=$(grep -oP "'tokens_per_s': \K[\d.]+" "$log" \
            | tail -n +3 \
            | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
        val_ppl=$(grep -oP 'perplexity: \K[\d.]+' "$log" | tail -1)
        printf "%-15s\tpeak=%s GB\tthroughput=%s tok/s\tVal PPL=%s\n" "$cfg" "$peak" "$tps" "$val_ppl"
    else
        printf "File %s doesn't exist\n" "$log"
    fi
done