#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-cxrfd}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[BOOTSTRAP_FAIL] conda was not found. Select the Miniconda CUDA 12.4 image."
  exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.11.9 -y
else
  echo "[BOOTSTRAP_INFO] Reusing existing environment=${ENV_NAME}"
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check

echo "[BOOTSTRAP_PASS] environment=${ENV_NAME}"
echo "Run: eval \"\$(conda shell.bash hook)\" && conda activate ${ENV_NAME}"
