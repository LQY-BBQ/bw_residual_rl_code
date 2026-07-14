#!/usr/bin/env bash
set -e
set -o pipefail
VENV_PATH="${HOME}/venvs/lerobot_ros310"
[ -f "${VENV_PATH}/bin/activate" ] && source "${VENV_PATH}/bin/activate" || { echo "[ERROR] Missing venv: ${VENV_PATH}"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python "${PROJECT_ROOT}/check_rl_rewards.py" "$@"
