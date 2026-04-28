#!/bin/bash
set -e

echo -e "\n===== RESULTS =====\n"

echo "--- Val PPL ---"
grep -oE "epoch [0-9]+: perplexity: [0-9.]+ eval_loss: [0-9.]+" outputs/fsdp-no_shard.log \
        outputs/fsdp-shard_grad_op.log outputs/fsdp-full_shard.log

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