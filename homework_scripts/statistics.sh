#!/bin/bash
set -e

if [[ $# -gt 5 ]]; then
    echo "Error: too many arguments. Expected: 0-5, got: $#" >&2
    exit 1
fi

printed=(_ 0 0 0 0 0)

print_experiment() {
    local i="$1"
    
    if [[ ! "$i" =~ ^[1-5]$ ]]; then
        echo "===== ARGUMENT SHOULD BE BETWEEN 1 AND 5, GOT $i ====="
        return 1
    fi
    
    if (( printed[i] == 1 )); then
        echo "===== EXPERIMENT $i RESULTS ALREADY PRINTED ====="
        return 0
    fi
    
    echo "===== PRINT EXPERIMENT $i RESULTS ====="
    printed[i]=1
    
    if [[ ! -f "${i}_table.sh" ]]; then
        echo "Error: ${i}_table.sh not found"
        return 1
    fi
    
    bash "${i}_table.sh"
}

while [[ $# -gt 0 ]]; do
    print_experiment "$1"
    shift
done