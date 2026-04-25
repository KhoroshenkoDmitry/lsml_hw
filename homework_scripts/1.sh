# 1. fp32
echo -e "Run experiment\tfp32\n===============================\n"
uv run train_single.py \
    --experiment-name baseline-fp32 \
    -d Salesforce/wikitext \
    -ds wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype fp32 \
    -s 512 \
    --batch-size 256 \
    --num-epochs 1
echo -e "===============================\nFinished experiment\tfp32\n"
# 2. bf16
echo -e "Run experiment\tbf16\n===============================\n"
uv run train_single.py \
    --experiment-name baseline-bf16 \
    -d Salesforce/wikitext \
    -ds  wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype bf16 \
    -s 512 \
    --batch-size 256 \
    --num-epochs 1
echo -e "===============================\nFinished experiment:\tbf16\n"
# 3. bf16 + activation checkpointing
echo -e "Run experiment:\tbf16+activation checkpointing\n===============================\n"
uv run train_single.py \
    --experiment-name baseline-bf16-ckpt \
    -d Salesforce/wikitext \
    -ds wikitext-103-v1 \
    -m EleutherAI/pythia-160m \
    --dtype bf16 \
    -s 512 \
    --activation-checkpointing \
    --batch-size 256 \
    --num-epochs 1
echo -e "===============================\nFinished experiment:\tbf16+activation checkpointing\n"
