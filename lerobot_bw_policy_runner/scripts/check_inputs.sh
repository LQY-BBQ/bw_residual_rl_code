#!/usr/bin/env bash
set -e
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${HOME}/venvs/lerobot_ros310"
[ -f "${VENV_PATH}/bin/activate" ] && source "${VENV_PATH}/bin/activate" || { echo "[ERROR] Missing venv: ${VENV_PATH}"; exit 1; }
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash || { echo "[ERROR] Missing /opt/ros/humble/setup.bash"; exit 1; }
[ -f "${HOME}/bw_teleoperate_ws/install/setup.bash" ] && source "${HOME}/bw_teleoperate_ws/install/setup.bash" || { echo "[ERROR] Missing bw_teleoperate_ws setup.bash"; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
python3 -m lerobot_bw_policy_runner.check_inputs "$@"
