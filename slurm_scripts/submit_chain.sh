#!/bin/bash
# slurm_scripts/submit_chain.sh — submit self-resuming job chains.
#
# Usage:
#   ./slurm_scripts/submit_chain.sh <mode> <n_jobs> [after_job_id]
#
# Examples (full 48-hour plan, submitted in one sitting):
#   J1=$(./slurm_scripts/submit_chain.sh baseline         1)
#   J2=$(./slurm_scripts/submit_chain.sh physics_informed 1 $J1)
#   J3=$(./slurm_scripts/submit_chain.sh lora             1 $J2)
#   J4=$(./slurm_scripts/submit_chain.sh combined         2 $J3)
#
# Each job in a chain depends on the previous with `afterany` (runs whether
# the predecessor succeeded, timed out with code 99, or crashed — the
# --resume auto logic makes every restart safe and idempotent). Jobs for a
# phase whose DONE marker already exists exit 0 within seconds.
#
# Prints ONLY the last submitted job id on stdout, so chains compose:
# feed it as [after_job_id] of the next phase.

set -euo pipefail

MODE="${1:?usage: submit_chain.sh <mode> <n_jobs> [after_job_id]}"
N="${2:?usage: submit_chain.sh <mode> <n_jobs> [after_job_id]}"
DEP="${3:-}"

case "${MODE}" in
    baseline|physics_informed|lora|combined) ;;
    *) echo "unknown mode: ${MODE}" >&2; exit 1 ;;
esac

SCRIPT="slurm_scripts/train_auto.slurm"
PREV="${DEP}"

for i in $(seq 1 "${N}"); do
    if [ -n "${PREV}" ]; then
        JID=$(MODE="${MODE}" sbatch --parsable --dependency=afterany:"${PREV}" \
              --export=ALL,MODE="${MODE}" "${SCRIPT}")
    else
        JID=$(MODE="${MODE}" sbatch --parsable \
              --export=ALL,MODE="${MODE}" "${SCRIPT}")
    fi
    echo "[submit_chain] ${MODE} job ${i}/${N}: ${JID}" >&2
    PREV="${JID}"
done

echo "${PREV}"
