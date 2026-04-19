#!/usr/bin/env bash
# Precompute cache (blocking), then run 5x5 semi-supervised grid search
# sequentially on node 114.
# Usage: ./run_grid.sh

set -euo pipefail

SSH="sshpass -p 991215 ssh -o StrictHostKeyChecking=no"
PROJDIR="/home/yan/logits"
ENV="fenics-ompi"
EPOCHS=300
NODE="10.22.10.114"
SHORT="114"

# ── step 1: precompute (blocking, skips if cache is current) ──────────────────
echo "=== [1/2] Precomputing cache on ${NODE} (blocking) ==="
$SSH "$NODE" bash -l -c "
  eval \"\$(conda shell.bash hook)\"
  conda activate ${ENV}
  cd ${PROJDIR}
  python precompute.py
"
echo "=== Precompute done ==="
echo ""

# ── step 2: build batch script and launch ────────────────────────────────────
echo "=== [2/2] Building and launching grid search ==="

TS=$(date +%Y%m%d_%H%M%S)
SESSION="logits_${SHORT}_grid_${TS}"
BATCH_DIR="${PROJDIR}/runs/${SHORT}_grid_${TS}"
BATCH_LOG="${BATCH_DIR}/batch.log"
BATCH_SH="${BATCH_DIR}/batch.sh"

mkdir -p "$BATCH_DIR"

TV_VALS=(0.001 0.05 0.2 1 5)
DUR_VALS=(0.001 0.05 0.2 1 5)

# Count total combos
N=0
for tv in "${TV_VALS[@]}"; do
  for dur in "${DUR_VALS[@]}"; do
    N=$(( N + 1 ))
  done
done

# Write batch script header
cat > "$BATCH_SH" <<'BASHHEADER'
#!/usr/bin/env bash
BASHHEADER

# Append conda activation (interpolated separately to avoid quoting issues)
cat >> "$BATCH_SH" <<ENVBLOCK
eval "\$(conda shell.bash hook)"
conda activate ${ENV}
cd ${PROJDIR}
echo "NODE: ${NODE}  SESSION: ${SESSION}  STARTED: \$(date)" | tee "${BATCH_LOG}"
echo "----------------------------------------" | tee -a "${BATCH_LOG}"
ENVBLOCK

# Append one block per combo
IDX=0
for tv in "${TV_VALS[@]}"; do
  for dur in "${DUR_VALS[@]}"; do
    IDX=$(( IDX + 1 ))
    RUN_DIR="runs/${SHORT}_${TS}_tv${tv}_dur${dur}"
    cat >> "$BATCH_SH" <<JOBBLOCK

echo "" | tee -a "${BATCH_LOG}"
echo "--- [${IDX}/${N}] tv=${tv} dur=${dur} started \$(date) ---" | tee -a "${BATCH_LOG}"
mkdir -p "${PROJDIR}/${RUN_DIR}"
{
  echo "CMD: python train_semi.py --augment --epochs ${EPOCHS} --lambda_tv_unann ${tv} --lambda_dur ${dur}  --ckpt_dir ${RUN_DIR}"
  echo "NODE: ${NODE}  SESSION: ${SESSION}"
  echo "STARTED: \$(date)"
  echo "PYTHON: \$(which python)"
  echo "----------------------------------------"
} > "${PROJDIR}/${RUN_DIR}/run.log"
python train_semi.py --augment --epochs ${EPOCHS} --lambda_tv_unann ${tv} --lambda_dur ${dur} --ckpt_dir ${RUN_DIR} 2>&1 | tee -a "${PROJDIR}/${RUN_DIR}/run.log"
echo "EXIT_CODE=\$?" >> "${PROJDIR}/${RUN_DIR}/run.log"
echo "--- [${IDX}/${N}] tv=${tv} dur=${dur} done \$(date) ---" | tee -a "${BATCH_LOG}"
JOBBLOCK
  done
done

cat >> "$BATCH_SH" <<FOOTERBLOCK

echo "" | tee -a "${BATCH_LOG}"
echo "=== All ${N} jobs done on ${NODE} at \$(date) ===" | tee -a "${BATCH_LOG}"
FOOTERBLOCK

chmod +x "$BATCH_SH"

echo "Launching ${NODE}: ${N} sequential jobs  session=${SESSION}"
echo "  batch log: ${BATCH_LOG}"
$SSH "$NODE" "tmux new-session -d -s '${SESSION}' '${BATCH_SH}'"
echo "  started."

echo ""
echo "Grid: ${N} jobs on ${NODE} (${EPOCHS} epochs each)."
echo "Monitor GPU  : ./nodes.sh"
echo "Watch batch  : tail -f ${BATCH_LOG}"
