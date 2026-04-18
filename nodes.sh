#!/usr/bin/env bash
# Usage: ./nodes.sh
# Shows GPU utilization + free memory on each node.

NODES=(10.22.10.113 10.22.10.114 10.22.10.115)
SSH="sshpass -p 991215 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3"

for node in "${NODES[@]}"; do
    printf "%-16s  " "$node"
    $SSH "$node" \
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
                    --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' '{printf \"GPU%s  util=%s%%  mem=%sMiB/%sMiB\n\", \$1,\$2,\$3,\$4}' \
         || echo 'no GPU'" \
        2>/dev/null || echo "unreachable"
done
