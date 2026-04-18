#!/usr/bin/env bash
# Usage: ./launch.sh <node-ip> <command...>
#   e.g. ./launch.sh 10.22.10.113 python train_semi.py --epochs 100

set -euo pipefail

NODE="${1:?Usage: launch.sh <node-ip> <cmd...>}"
shift
CMD="$*"

SSH="sshpass -p 991215 ssh -o StrictHostKeyChecking=no"

TS=$(date +%Y%m%d_%H%M%S)
SHORT=$(echo "$NODE" | awk -F. '{print $NF}')
SESSION="logits_${SHORT}_${TS}"

MASTER="runs"
CMD_CLEAN=$(echo "$CMD" | sed 's/--ckpt_dir [^ ]*//')
RUN_DIR="${MASTER}/${SHORT}_${TS}"
CMD_FINAL="${CMD_CLEAN} --ckpt_dir ${RUN_DIR}"

PROJDIR="/home/yan/logits"
RUNDIR_ABS="${PROJDIR}/${RUN_DIR}"
LOGFILE="${RUNDIR_ABS}/run.log"
WRAPPER="${RUNDIR_ABS}/run.sh"

mkdir -p "$RUNDIR_ABS"

# Write a self-contained wrapper script into the run dir
cat > "$WRAPPER" <<WRAPPER
#!/usr/bin/env bash
eval "\$(conda shell.bash hook)"
conda activate fenics-ompi

cd ${PROJDIR}

{
echo "CMD: ${CMD_FINAL}"
echo "NODE: ${NODE}  SESSION: ${SESSION}"
echo "STARTED: \$(date)"
echo "PYTHON: \$(which python)"
echo "----------------------------------------"
} > ${LOGFILE}

${CMD_FINAL} 2>&1 | tee -a ${LOGFILE}
echo "EXIT_CODE=\$?" >> ${LOGFILE}
WRAPPER
chmod +x "$WRAPPER"

echo "Launching on $NODE  session=$SESSION"
echo "Run dir: $RUN_DIR"
echo "Log: $LOGFILE"

$SSH "$NODE" "tmux new-session -d -s '$SESSION' '${WRAPPER}'"
echo "Session '$SESSION' started."
