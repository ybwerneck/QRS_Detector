#!/usr/bin/env bash
# Run precompute (blocking) then launch semi-supervised grid search on nodes 114 and 115.
# Usage: ./run_grid.sh

set -euo pipefail

SSH="sshpass -p 991215 ssh -o StrictHostKeyChecking=no"
PROJDIR="/home/yan/logits"

echo "=== [1/2] Precomputing cache on 10.22.10.114 (blocking) ==="
$SSH 10.22.10.114 bash -lc "
  eval \"\$(conda shell.bash hook)\"
  conda activate fenics-ompi
  cd ${PROJDIR}
  python precompute.py --force
"
echo "=== Precompute done ==="

echo ""
echo "=== [2/2] Launching grid search ==="

# Node 114: lambda_dur=0.0, lambda_tv_unann in [0.0, 0.5, 2.0, 5.0]
for tv in 0.0 0.5 2.0 5.0; do
  ./launch.sh 10.22.10.114 python train_semi.py --augment --epochs 1000 --lambda_tv_unann "$tv" --lambda_dur 0.0 --force
done

# Node 115: lambda_dur=0.5, lambda_tv_unann in [0.0, 0.5, 2.0, 5.0]
for tv in 0.0 0.5 2.0 5.0; do
  ./launch.sh 10.22.10.115 python train_semi.py --augment --epochs 1000 --lambda_tv_unann "$tv" --lambda_dur 0.5 --force
done

echo ""
echo "All 8 grid search jobs launched."
echo "Monitor with: ./nodes.sh"
echo "Watch logs:   tail -f runs/<id>/run.log"
