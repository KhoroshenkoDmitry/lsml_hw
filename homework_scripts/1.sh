# 1. fp32
echo -e "Run experiment\tfp32\n===============================\n"
uv run ../scripts/train_dp_tp.py \
    --experiment-name baseline-fp32 \
    -d Salesforce/wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype fp32 \
    --batch-size 16 \
    --num-epochs 5
echo -e "===============================\nFinished experiment\tfp32\n"
# 2. bf16
echo -e "Run experiment\tbf16\n===============================\n"
uv run ../scripts/train_dp_tp.py \
    --experiment-name baseline-bf16 \
    -d Salesforce/wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype bf16 \
    --batch-size 16 \
    --num-epochs 5
echo -e "===============================\nFinished experiment:\tbf16\n"
# 3. bf16 + activation checkpointing
echo -e "Run experiment:\tbf16+activation checkpointing\n===============================\n"
uv run ../scripts/train_dp_tp.py \
    --experiment-name baseline-bf16-ckpt \
    -d Salesforce/wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype bf16 \
    --activation-checkpointing \
    --batch-size 16 \
    --num-epochs 5
echo -e "===============================\nFinished experiment:\tbf16+activation checkpointing\n"
