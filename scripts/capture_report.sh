#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/CXR_FD}"
RUNS="${RUNS:-${PROJECT}/runs}"
REPORTS="${PROJECT}/reports"
mkdir -p "${REPORTS}"

python --version > "${REPORTS}/python_version.txt" 2>&1
python -m pip freeze > "${REPORTS}/pip_freeze.txt"
nvidia-smi > "${REPORTS}/nvidia_smi.txt"
find "${RUNS}" -type f -name '*.log' -print0 \
  | sort -z \
  | xargs -0 -r grep -hE '\[(ENV_PASS|MANIFEST_READY|LEAKAGE_CHECK|CLASSIFIER_FINAL|REF_STATS_READY|GAN_START|FD_CONFIG|QUEUE_INIT|EPOCH_SUMMARY|TRAINING_DONE|FAIRNESS_CHECK|EVAL_CLASS|EVAL_FINAL|EVALUATION_DONE)\]' \
  > "${REPORTS}/key_log_markers.txt" || true
find "${PROJECT}" -type f \
  \( -name '*.pt' -o -name '*.npz' -o -name '*.csv' -o -name '*.png' \) \
  -printf '%s %p\n' | sort -n > "${REPORTS}/artifact_inventory.txt"

echo "[REPORT_CAPTURED] ${REPORTS}"
