#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 自动激活 LeRobot Python 环境
VENV_PATH="${VENV_PATH:-${HOME}/venvs/lerobot_ros310}"

if [ -f "${VENV_PATH}/bin/activate" ]; then
  if [ "${VIRTUAL_ENV:-}" != "${VENV_PATH}" ]; then
    echo "[INFO] Activating Python venv: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
  fi
else
  echo "[ERROR] Python virtual environment not found: ${VENV_PATH}" >&2
  echo "[ERROR] Please check whether ~/venvs/lerobot_ros310 exists." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECK_SCRIPT="${PROJECT_ROOT}/tools/check_bw_lerobot_dataset.py"

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  bash ${BASH_SOURCE[0]} DATASET_ROOT [options]"
  echo ""
  echo "Examples:"
  echo "  bash ${BASH_SOURCE[0]} ~/robot_datasets/bw_lerobot/session_xxx"
  echo "  bash ${BASH_SOURCE[0]} ~/robot_datasets/bw_lerobot/session_xxx --episode 0 --mode rl --save-csv"
  echo "  bash ${BASH_SOURCE[0]} ~/robot_datasets/bw_lerobot/session_xxx --mode rl --all-episodes --save-csv"
  echo "  bash ${BASH_SOURCE[0]} ~/robot_datasets/bw_lerobot/session_xxx --all-episodes --out-dir ~/robot_datasets/check_viz/session_xxx"
  exit 1
fi

if [ ! -f "${CHECK_SCRIPT}" ]; then
  echo "[ERROR] check script not found: ${CHECK_SCRIPT}" >&2
  exit 2
fi

echo "[INFO] Python: $(which python)"
echo "[INFO] Check script: ${CHECK_SCRIPT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${CHECK_SCRIPT}" "$@"
