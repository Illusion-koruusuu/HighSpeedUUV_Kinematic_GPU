#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$project_dir"
python_bin="${PYTHON_BIN:-$project_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found: $python_bin" >&2
    echo "HoloOcean replay uses the same .venv; install HoloOcean into it." >&2
    exit 1
fi
exec "$python_bin" -m training.replay_holoocean "$@"
