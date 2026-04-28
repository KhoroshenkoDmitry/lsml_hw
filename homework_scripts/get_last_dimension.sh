#!/bin/bash
set -e

if [ $# -ne 6 ]; then
    echo "invalid count of agruments pairs. Expected: 3, got: $(($#/2))"
    exit 1
fi 
NUMERATOR=""
DENOMINATOR="1"
while [ $# -gt 0 ]; do
    case "$1" in
    -s|--seq_len|-b|--local_batch_size|-d|--dp_size|-ga|--grad_accumulation_steps)
        DENOMINATOR=$((DENOMINATOR * "$2"))
        shift
        shift
        ;;
    -gb|--global_batch_size)
        NUMERATOR="$2"
        shift
        shift
        ;;
    *)
        echo "Uknown argument $1"
        exit 1
        ;;
    esac
done
if [ -z "$NUMERATOR" ]; then 
    echo "I expected to get a global batch size :'("
    exit 1
fi
echo $((NUMERATOR / DENOMINATOR))