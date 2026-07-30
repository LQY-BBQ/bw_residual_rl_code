#!/usr/bin/env bash
set -uo pipefail

# 批量检查 ~/robot_datasets/bw_rl_corrections/rl_correction_*
# 默认将检查结果写入 ~/robot_datasets/bw_rl_corrections_viz/同名目录。
#
# 直接运行：
#   bash batch_check_rl_datasets.sh
#
# 也可以自定义输入、输出目录：
#   bash batch_check_rl_datasets.sh INPUT_ROOT OUTPUT_ROOT
#
# 如果本脚本不在 lerobot_bw_data_collector/scripts 下，并且你的工程不在默认位置，
# 可以先指定：
#   export LEROBOT_BW_COLLECTOR_ROOT=~/mycode/你的路径/lerobot_bw_data_collector

INPUT_ROOT="${1:-${HOME}/robot_datasets/bw_rl_corrections}"
OUTPUT_ROOT="${2:-${HOME}/robot_datasets/bw_rl_corrections_viz}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd || true)"
DEFAULT_PROJECT_ROOT="${HOME}/mycode/bw_residual_rl_code/lerobot_bw_data_collector"

if [ -f "${LOCAL_PROJECT_ROOT}/scripts/check_dataset.sh" ]; then
  PROJECT_ROOT="${LOCAL_PROJECT_ROOT}"
else
  PROJECT_ROOT="${LEROBOT_BW_COLLECTOR_ROOT:-${DEFAULT_PROJECT_ROOT}}"
fi

CHECK_SCRIPT="${PROJECT_ROOT}/scripts/check_dataset.sh"
LOG_DIR="${OUTPUT_ROOT}/_batch_logs"
SUMMARY_FILE="${OUTPUT_ROOT}/batch_check_summary.txt"

if [ ! -d "${INPUT_ROOT}" ]; then
  echo "[ERROR] 输入目录不存在: ${INPUT_ROOT}" >&2
  exit 1
fi

if [ ! -f "${CHECK_SCRIPT}" ]; then
  echo "[ERROR] 找不到检查脚本: ${CHECK_SCRIPT}" >&2
  echo "[ERROR] 请确认工程位置，或设置 LEROBOT_BW_COLLECTOR_ROOT。" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# 按 001、002、...、030 的自然顺序读取所有数据集目录。
mapfile -d '' DATASETS < <(
  find "${INPUT_ROOT}" \
    -mindepth 1 -maxdepth 1 -type d \
    -name 'rl_correction_*' \
    -print0 | sort -zV
)

if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "[ERROR] 在以下位置没有找到 rl_correction_* 数据集目录:" >&2
  echo "        ${INPUT_ROOT}" >&2
  exit 3
fi

TOTAL="${#DATASETS[@]}"
SUCCESS=0
FAILED=0
SKIPPED=0
FAILED_NAMES=()

{
  echo "Batch check started: $(date '+%F %T')"
  echo "Input root : ${INPUT_ROOT}"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "Checker    : ${CHECK_SCRIPT}"
  echo "Datasets   : ${TOTAL}"
  echo
} > "${SUMMARY_FILE}"

echo "============================================================"
echo "[INFO] 共找到 ${TOTAL} 组数据"
echo "[INFO] 输入目录: ${INPUT_ROOT}"
echo "[INFO] 输出目录: ${OUTPUT_ROOT}"
echo "============================================================"

INDEX=0
for DATASET_DIR in "${DATASETS[@]}"; do
  INDEX=$((INDEX + 1))
  NAME="$(basename "${DATASET_DIR}")"
  OUT_DIR="${OUTPUT_ROOT}/${NAME}"
  LOG_FILE="${LOG_DIR}/${NAME}.log"

  echo
  echo "------------------------------------------------------------"
  echo "[INFO] [${INDEX}/${TOTAL}] 正在检查: ${NAME}"
  echo "[INFO] 数据目录: ${DATASET_DIR}"
  echo "[INFO] 结果目录: ${OUT_DIR}"
  echo "[INFO] 日志文件: ${LOG_FILE}"
  echo "------------------------------------------------------------"

  # LeRobot 数据集至少应该存在 meta/info.json。
  if [ ! -f "${DATASET_DIR}/meta/info.json" ]; then
    echo "[WARN] 跳过 ${NAME}: 缺少 meta/info.json" | tee "${LOG_FILE}"
    echo "SKIPPED  ${NAME}  missing meta/info.json" >> "${SUMMARY_FILE}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  mkdir -p "${OUT_DIR}"

  if bash "${CHECK_SCRIPT}" "${DATASET_DIR}" \
      --all-episodes \
      --out-dir "${OUT_DIR}" \
      --mode rl \
      2>&1 | tee "${LOG_FILE}"; then
    echo "[OK] ${NAME} 检查完成"
    echo "SUCCESS  ${NAME}" >> "${SUMMARY_FILE}"
    SUCCESS=$((SUCCESS + 1))
  else
    STATUS=$?
    echo "[ERROR] ${NAME} 检查失败，退出码: ${STATUS}" >&2
    echo "FAILED   ${NAME}  exit=${STATUS}" >> "${SUMMARY_FILE}"
    FAILED_NAMES+=("${NAME}")
    FAILED=$((FAILED + 1))
  fi
done

{
  echo
  echo "Batch check finished: $(date '+%F %T')"
  echo "Total   : ${TOTAL}"
  echo "Success : ${SUCCESS}"
  echo "Failed  : ${FAILED}"
  echo "Skipped : ${SKIPPED}"
} >> "${SUMMARY_FILE}"

echo
echo "============================================================"
echo "[SUMMARY] 总数: ${TOTAL}"
echo "[SUMMARY] 成功: ${SUCCESS}"
echo "[SUMMARY] 失败: ${FAILED}"
echo "[SUMMARY] 跳过: ${SKIPPED}"
echo "[SUMMARY] 汇总: ${SUMMARY_FILE}"
echo "[SUMMARY] 日志: ${LOG_DIR}"

if [ "${FAILED}" -gt 0 ]; then
  echo "[SUMMARY] 失败的数据集: ${FAILED_NAMES[*]}" >&2
  echo "============================================================"
  exit 4
fi

echo "============================================================"
exit 0
