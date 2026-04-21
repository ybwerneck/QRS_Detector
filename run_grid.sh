#!/usr/bin/env bash
# Run a semi-supervised grid search split across two nodes.
# Each grid gets its own master folder under grids/ with:
#   recipe.txt   — auto-generated base; fill in "why" manually before launching
#   runs/        — one subfolder per combo
#   results/     — batch scripts, batch logs, summaries
#
# Usage: ./run_grid.sh <grid_name>
#   e.g. ./run_grid.sh tv_lambda_sweep

set -euo pipefail

GRID_NAME="${1:?Usage: ./run_grid.sh <grid_name>}"

SSH="sshpass -p 991215 ssh -o StrictHostKeyChecking=no"
PROJDIR="/home/yan/logits"
ENV="fenics-ompi"
EPOCHS=10000

NODES=(10.22.10.112 10.22.10.114)
SHORTS=(112 114)

LAM_VALS=(0 0.01 0.1 1)

# ── grid master folder ─────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
GRID_DIR="${PROJDIR}/grids/${GRID_NAME}_${TS}"
RUNS_DIR="${GRID_DIR}/runs"
RESULTS_DIR="${GRID_DIR}/results"
mkdir -p "$RUNS_DIR" "$RESULTS_DIR"

# ── auto-generate recipe.txt ───────────────────────────────────────────────────
cat > "${GRID_DIR}/recipe.txt" <<RECIPE
Grid: ${GRID_NAME}
Created: $(date)
Nodes: ${NODES[*]}

WHAT IS BEING SEARCHED
----------------------
  lambda_tv_scale  : ${LAM_VALS[*]}
  lambda_tv_shift  : ${LAM_VALS[*]}
  lambda_tv_unann  : ${LAM_VALS[*]}

FIXED PARAMETERS
----------------
  script  : train_semi.py
  epochs  : ${EPOCHS}
  flags   : --augment
  env     : ${ENV}

GOAL / NOTES
------------
  < fill in manually before launching >

RECIPE

echo "=== Grid folder: ${GRID_DIR} ==="
echo "    Edit recipe.txt before proceeding if needed."
echo ""

# ── step 1: precompute on 114 (blocking) ──────────────────────────────────────
echo "=== [1/2] Precomputing cache on 10.22.10.114 (blocking) ==="
$SSH 10.22.10.114 bash -l -c "
  eval \"\$(conda shell.bash hook)\"
  conda activate ${ENV}
  cd ${PROJDIR}
  python precompute.py
"
echo "=== Precompute done ==="
echo ""

# ── step 2: enumerate all combos and split across nodes ───────────────────────
echo "=== [2/2] Building and launching grid ==="

COMBOS=()
for sc in "${LAM_VALS[@]}"; do
  for sh in "${LAM_VALS[@]}"; do
    for un in "${LAM_VALS[@]}"; do
      COMBOS+=("${sc}:${sh}:${un}")
    done
  done
done
N_TOTAL=${#COMBOS[@]}
N_PER_NODE=$(( (N_TOTAL + 1) / 2 ))

echo "Total combos: ${N_TOTAL}  (${N_PER_NODE} per node)"

for ni in 0 1; do
  NODE="${NODES[$ni]}"
  SHORT="${SHORTS[$ni]}"
  SESSION="logits_${SHORT}_${GRID_NAME}_${TS}"
  BATCH_SH="${RESULTS_DIR}/batch_${SHORT}.sh"
  BATCH_LOG="${RESULTS_DIR}/batch_${SHORT}.log"

  # slice of combos for this node
  START=$(( ni * N_PER_NODE ))
  END=$(( START + N_PER_NODE ))
  if (( END > N_TOTAL )); then END=$N_TOTAL; fi
  N_NODE=$(( END - START ))

  # Write batch script
  cat > "$BATCH_SH" <<'BASHHEADER'
#!/usr/bin/env bash
BASHHEADER

  cat >> "$BATCH_SH" <<ENVBLOCK
eval "\$(conda shell.bash hook)"
conda activate ${ENV}
cd ${PROJDIR}
echo "NODE: ${NODE}  SESSION: ${SESSION}  STARTED: \$(date)" | tee "${BATCH_LOG}"
echo "----------------------------------------" | tee -a "${BATCH_LOG}"
ENVBLOCK

  IDX=0
  for (( ci=START; ci<END; ci++ )); do
    IFS=':' read -r sc sh un <<< "${COMBOS[$ci]}"
    IDX=$(( IDX + 1 ))
    RUN_SUBDIR="grids/${GRID_NAME}_${TS}/runs/${SHORT}_sc${sc}_sh${sh}_un${un}"

    cat >> "$BATCH_SH" <<JOBBLOCK

echo "" | tee -a "${BATCH_LOG}"
echo "--- [${IDX}/${N_NODE}] scale=${sc} shift=${sh} unann=${un} started \$(date) ---" | tee -a "${BATCH_LOG}"
mkdir -p "${PROJDIR}/${RUN_SUBDIR}"
{
  echo "CMD: python train_semi.py --augment --epochs ${EPOCHS} --lambda_tv_scale ${sc} --lambda_tv_shift ${sh} --lambda_tv_unann ${un} --ckpt_dir ${RUN_SUBDIR}"
  echo "NODE: ${NODE}  SESSION: ${SESSION}"
  echo "STARTED: \$(date)"
  echo "PYTHON: \$(which python)"
  echo "----------------------------------------"
} > "${PROJDIR}/${RUN_SUBDIR}/run.log"
python train_semi.py --augment --epochs ${EPOCHS} \
  --lambda_tv_scale ${sc} --lambda_tv_shift ${sh} --lambda_tv_unann ${un} \
  --ckpt_dir ${RUN_SUBDIR} 2>&1 | tee -a "${PROJDIR}/${RUN_SUBDIR}/run.log"
echo "EXIT_CODE=\$?" >> "${PROJDIR}/${RUN_SUBDIR}/run.log"
echo "--- [${IDX}/${N_NODE}] sc=${sc} sh=${sh} un=${un} done \$(date) ---" | tee -a "${BATCH_LOG}"
JOBBLOCK
  done

  cat >> "$BATCH_SH" <<FOOTERBLOCK

echo "" | tee -a "${BATCH_LOG}"
echo "=== All ${N_NODE} jobs done on ${NODE} at \$(date) ===" | tee -a "${BATCH_LOG}"
FOOTERBLOCK

  chmod +x "$BATCH_SH"

  echo "Launching ${NODE}: ${N_NODE} jobs  session=${SESSION}"
  $SSH "$NODE" "tmux new-session -d -s '${SESSION}' '${BATCH_SH}'"
  echo "  started."
done

echo ""
echo "Grid '${GRID_NAME}' launched: ${N_TOTAL} jobs across ${NODES[*]} (${EPOCHS} epochs each)."
echo ""
echo "Monitor GPU   : ./nodes.sh"
echo "Watch 112     : tail -f ${RESULTS_DIR}/batch_112.log"
echo "Watch 114     : tail -f ${RESULTS_DIR}/batch_114.log"
echo "Recipe        : ${GRID_DIR}/recipe.txt"
