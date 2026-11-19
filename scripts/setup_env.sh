#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="$project_dir/.venv"
python_base="${PYTHON:-python3}"
pip_cache="$project_dir/.cache/pip"
mkdir -p "$pip_cache"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    if "$python_base" -m venv "$venv_dir" 2>/dev/null; then
        :
    elif command -v virtualenv >/dev/null 2>&1; then
        virtualenv --python="$python_base" --pip=embed "$venv_dir"
    else
        echo "Cannot create a virtual environment." >&2
        echo "Install python3-venv or virtualenv, then rerun this script." >&2
        exit 1
    fi
fi

export PIP_CACHE_DIR="$pip_cache"
python_bin="$venv_dir/bin/python"
"$python_bin" -m pip install --upgrade "pip==24.0" "setuptools<76" wheel
"$python_bin" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.5.1"
"$python_bin" -m pip install -r "$project_dir/requirements.txt"
"$python_bin" -m pip install -e "$project_dir" --no-deps

"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Check nvidia-smi before starting training."
    )
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
