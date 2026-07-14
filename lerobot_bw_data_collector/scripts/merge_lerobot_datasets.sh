#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 通用 LeRobot 数据集合并脚本
#
# 适用范围：
#   1. 模仿学习 / ACT 数据（采集器 --mode bc）
#   2. ACT 推理 + 人工纠正数据（采集器 --mode rl）
#   3. 每个源目录包含 1 个 episode，或者包含多个 episode
#   4. 图像保存为 video 或 image
#
# 重要约束：
#   - 同一次合并中的所有源数据集必须具有完全相同的 fps、robot_type 和 features。
#   - 模仿学习数据与 RL 数据字段不同，不能混在同一个输出数据集中。
#   - 本脚本不改变动作、状态、奖励或图像，只重新编号并合并。
#
# 当前默认配置面向：
#   ~/robot_datasets/bw_rl_corrections
#       -> ~/robot_datasets/bw_rl_corrections_merged
#
# 直接运行：
#   bash merge_lerobot_datasets.sh
#
# 指定其他数据：
#   bash merge_lerobot_datasets.sh \
#     --src ~/robot_datasets/bw_rl_corrections \
#     --out ~/robot_datasets/bw_rl_corrections_merged
#
# 强制覆盖：
#   bash merge_lerobot_datasets.sh --force
# ==============================================================================

# -------------------------- 可按需修改的默认配置 -------------------------------
DEFAULT_SRC_ROOT="${HOME}/robot_datasets/bw_rl_corrections"
DEFAULT_VENV_PATH="${HOME}/venvs/lerobot_ros310"
DEFAULT_MODE="auto"               # auto | bc | rl
DEFAULT_INCLUDE_GLOB="*"          # 例如：rl_correction_* 或 pick_block_to_box_*
# ------------------------------------------------------------------------------

SRC_ROOT="${MERGE_SRC_ROOT:-${DEFAULT_SRC_ROOT}}"
MERGED_ROOT="${MERGE_OUT_ROOT:-}"
NEW_REPO_ID="${MERGE_REPO_ID:-}"
VENV_PATH="${MERGE_VENV_PATH:-${DEFAULT_VENV_PATH}}"
MODE="${MERGE_MODE:-${DEFAULT_MODE}}"
INCLUDE_GLOB="${MERGE_INCLUDE_GLOB:-${DEFAULT_INCLUDE_GLOB}}"
LEROBOT_SRC="${MERGE_LEROBOT_SRC:-}"

FORCE=0
RECURSIVE=0
DRY_RUN=0
COPY_ANNOTATIONS=1

print_usage() {
  cat <<USAGE
用法：
  bash ${BASH_SOURCE[0]} [选项]

选项：
  --src DIR
      包含多个 LeRobot 数据集目录的父目录。
      默认：${SRC_ROOT}

  --out DIR
      合并后数据集的完整输出目录。
      默认：<SRC_ROOT>_merged

  --repo-id ID
      合并后数据集的 LeRobot repo_id。
      默认：local/<输出目录名>

  --mode MODE
      数据类型检查模式：
        auto  自动识别 bc 或 rl
        bc    要求为普通模仿学习 / ACT 数据
        rl    要求为带人工纠正和 reward/done 的 RL 数据
      默认：${MODE}

  --include GLOB
      只合并目录名匹配该 shell glob 的数据集。
      示例：--include 'rl_correction_*'
      默认：${INCLUDE_GLOB}

  --recursive
      递归搜索 SRC_ROOT 下的 LeRobot 数据集。
      默认只检查 SRC_ROOT 自身和它的直接子目录。

  --venv DIR
      Python 虚拟环境。
      默认：${VENV_PATH}

  --lerobot-src DIR
      显式指定 LeRobot 源码的 src 目录。
      该目录下应存在 lerobot/。

  --no-copy-annotations
      不复制源数据集中的 annotations/ JSON 和其他旁路文件。

  --dry-run
      只发现并检查源数据集，不执行合并。

  --force
      删除已存在的输出目录后重新合并。

  -h, --help
      显示帮助。

常用示例：

  # 当前 31 组 RL 纠正数据：按顶部默认配置直接合并
  bash ${BASH_SOURCE[0]}

  # 合并普通 ACT 模仿学习数据
  bash ${BASH_SOURCE[0]} \
    --src ~/robot_datasets/pick_block_0617 \
    --out ~/robot_datasets/pick_block_0617_merged \
    --mode bc

  # 合并 RL 纠正数据
  bash ${BASH_SOURCE[0]} \
    --src ~/robot_datasets/bw_rl_corrections \
    --out ~/robot_datasets/bw_rl_corrections_merged \
    --mode rl

  # 输出目录存在时重建
  bash ${BASH_SOURCE[0]} --force

说明：
  1. auto 模式可以自动识别普通 BC 数据或 RL 数据。
  2. 一次合并不能混合两种不同 schema 的数据。
  3. RL 合并结果可同时用于 Residual BC 和后续 Residual RL。
USAGE
}

need_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "[ERROR] ${option} 需要一个参数。"
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      need_value "$1" "${2:-}"
      SRC_ROOT="$2"
      shift 2
      ;;
    --out)
      need_value "$1" "${2:-}"
      MERGED_ROOT="$2"
      shift 2
      ;;
    --repo-id)
      need_value "$1" "${2:-}"
      NEW_REPO_ID="$2"
      shift 2
      ;;
    --mode)
      need_value "$1" "${2:-}"
      MODE="$2"
      shift 2
      ;;
    --include)
      need_value "$1" "${2:-}"
      INCLUDE_GLOB="$2"
      shift 2
      ;;
    --venv)
      need_value "$1" "${2:-}"
      VENV_PATH="$2"
      shift 2
      ;;
    --lerobot-src)
      need_value "$1" "${2:-}"
      LEROBOT_SRC="$2"
      shift 2
      ;;
    --recursive)
      RECURSIVE=1
      shift
      ;;
    --no-copy-annotations)
      COPY_ANNOTATIONS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数：$1"
      print_usage
      exit 2
      ;;
  esac
done

case "${MODE}" in
  auto|bc|rl) ;;
  *)
    echo "[ERROR] --mode 只允许 auto、bc 或 rl，当前值：${MODE}"
    exit 2
    ;;
esac

expand_path() {
  local value="$1"
  value="${value/#\~/${HOME}}"
  realpath -m "${value}"
}

SRC_ROOT="$(expand_path "${SRC_ROOT}")"
VENV_PATH="$(expand_path "${VENV_PATH}")"

if [[ -z "${MERGED_ROOT}" ]]; then
  MERGED_ROOT="${SRC_ROOT%/}_merged"
fi
MERGED_ROOT="$(expand_path "${MERGED_ROOT}")"

sanitize_repo_name() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  value="$(printf '%s' "${value}" | sed -E 's/[^a-z0-9_-]+/_/g; s/_+/_/g; s/^_+//; s/_+$//')"
  if [[ -z "${value}" ]]; then
    value="merged_dataset"
  fi
  printf '%s' "${value}"
}

if [[ -z "${NEW_REPO_ID}" ]]; then
  NEW_REPO_ID="local/$(sanitize_repo_name "$(basename "${MERGED_ROOT}")")"
fi

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "[ERROR] 源目录不存在：${SRC_ROOT}"
  exit 1
fi

if [[ "${SRC_ROOT}" == "${MERGED_ROOT}" ]]; then
  echo "[ERROR] 源目录和输出目录不能相同：${SRC_ROOT}"
  exit 1
fi

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
else
  echo "[ERROR] 找不到 Python 虚拟环境：${VENV_PATH}"
  echo "        请使用 --venv 指定正确路径。"
  exit 1
fi

# 优先使用显式指定的 LeRobot 源码。
if [[ -n "${LEROBOT_SRC}" ]]; then
  LEROBOT_SRC="$(expand_path "${LEROBOT_SRC}")"
  if [[ ! -d "${LEROBOT_SRC}/lerobot" ]]; then
    echo "[ERROR] --lerobot-src 指定的目录下没有 lerobot/：${LEROBOT_SRC}"
    exit 1
  fi
  export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH:-}"
  echo "[INFO] 使用指定的 LeRobot 源码：${LEROBOT_SRC}"
else
  # 没有显式指定时，优先寻找本地源码；否则使用虚拟环境中安装的 LeRobot。
  for candidate in \
    "${HOME}/mycode/lerobot-main/src" \
    "${HOME}/mycode/lerobot/src" \
    "${HOME}/mycode/lerobot_v044/src" \
    "${HOME}/mycode/lerobot_v044/lerobot-main/src"; do
    if [[ -d "${candidate}/lerobot" ]]; then
      export PYTHONPATH="${candidate}:${PYTHONPATH:-}"
      echo "[INFO] 使用本地 LeRobot 源码：${candidate}"
      break
    fi
  done
fi

mkdir -p "$(dirname "${MERGED_ROOT}")"

echo "=============================================================================="
echo "[INFO] 源目录          ：${SRC_ROOT}"
echo "[INFO] 输出目录        ：${MERGED_ROOT}"
echo "[INFO] repo_id         ：${NEW_REPO_ID}"
echo "[INFO] 期望数据模式    ：${MODE}"
echo "[INFO] 目录名称过滤    ：${INCLUDE_GLOB}"
echo "[INFO] 递归搜索        ：${RECURSIVE}"
echo "[INFO] 复制 annotations：${COPY_ANNOTATIONS}"
echo "[INFO] dry-run         ：${DRY_RUN}"
echo "=============================================================================="

python3 - \
  "${SRC_ROOT}" \
  "${MERGED_ROOT}" \
  "${NEW_REPO_ID}" \
  "${MODE}" \
  "${INCLUDE_GLOB}" \
  "${RECURSIVE}" \
  "${FORCE}" \
  "${DRY_RUN}" \
  "${COPY_ANNOTATIONS}" <<'PY'
from __future__ import annotations

import fnmatch
import inspect
import json
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


src_root = Path(sys.argv[1]).expanduser().resolve()
merged_root = Path(sys.argv[2]).expanduser().resolve()
new_repo_id = sys.argv[3]
requested_mode = sys.argv[4]
include_glob = sys.argv[5]
recursive = sys.argv[6] == "1"
force = sys.argv[7] == "1"
dry_run = sys.argv[8] == "1"
copy_annotations = sys.argv[9] == "1"


BASE_REQUIRED = {
    "observation.state",
    "action",
    "observation.images.env_cam",
    "observation.images.left_wrist_cam",
    "observation.images.right_wrist_cam",
}

RL_REQUIRED = {
    "control_source",
    "is_intervention",
    "has_human_action",
    "action.act",
    "action.rl_delta",
    "action.human",
    "action.executed",
    "reward",
    "done",
    "success",
}

RL_HINT_FIELDS = RL_REQUIRED | {
    "timing.arm_action_dt",
    "timing.gripper_action_dt",
    "timing.action_act_dt",
    "timing.action_final_dt",
}


def log(message: str = "") -> None:
    print(message, flush=True)


def fail(message: str, code: int = 1) -> None:
    raise SystemExit(f"[ERROR] {message}")


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def sanitize_repo_component(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "dataset"


def is_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta" / "info.json").is_file()
        and (path / "data").is_dir()
    )


def discover_dataset_roots() -> list[Path]:
    candidates: list[Path] = []

    if is_dataset_root(src_root):
        candidates.append(src_root)

    if recursive:
        for info_path in src_root.rglob("meta/info.json"):
            candidate = info_path.parent.parent
            if is_dataset_root(candidate):
                candidates.append(candidate.resolve())
    else:
        for candidate in src_root.iterdir():
            if is_dataset_root(candidate):
                candidates.append(candidate.resolve())

    unique: dict[str, Path] = {}
    for candidate in candidates:
        resolved = candidate.resolve()

        # 防止输出目录位于源目录下时被再次识别为输入。
        if resolved == merged_root or merged_root in resolved.parents:
            continue

        if not fnmatch.fnmatch(resolved.name, include_glob):
            continue

        unique[str(resolved)] = resolved

    return sorted(unique.values(), key=natural_key)


def normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SourceInfo:
    root: Path
    info: dict[str, Any]
    features: dict[str, Any]
    detected_mode: str
    total_episodes: int
    total_frames: int


def detect_mode(features: dict[str, Any], root: Path) -> str:
    keys = set(features)

    missing_base = BASE_REQUIRED - keys
    if missing_base:
        fail(
            f"数据集缺少当前训练链路要求的基础字段：{root}\n"
            f"        缺少：{sorted(missing_base)}"
        )

    present_rl = RL_HINT_FIELDS & keys
    missing_rl = RL_REQUIRED - keys

    if not missing_rl:
        return "rl"

    if present_rl:
        fail(
            f"数据集包含部分 RL 字段，但 schema 不完整：{root}\n"
            f"        已有 RL 字段：{sorted(present_rl)}\n"
            f"        缺少 RL 字段：{sorted(missing_rl)}"
        )

    return "bc"


def load_source_info(root: Path) -> SourceInfo:
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"无法读取 {info_path}：{exc}")

    features = info.get("features")
    if not isinstance(features, dict) or not features:
        fail(f"meta/info.json 中没有有效 features：{info_path}")

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes <= 0:
        fail(f"total_episodes 必须大于 0：{root}")
    if total_frames <= 0:
        fail(f"total_frames 必须大于 0：{root}")

    detected_mode = detect_mode(features, root)

    return SourceInfo(
        root=root,
        info=info,
        features=features,
        detected_mode=detected_mode,
        total_episodes=total_episodes,
        total_frames=total_frames,
    )


def schema_diff(reference: dict[str, Any], current: dict[str, Any]) -> str:
    ref_keys = set(reference)
    cur_keys = set(current)
    missing = sorted(ref_keys - cur_keys)
    extra = sorted(cur_keys - ref_keys)
    changed = sorted(
        key
        for key in ref_keys & cur_keys
        if normalized_json(reference[key]) != normalized_json(current[key])
    )

    lines: list[str] = []
    if missing:
        lines.append(f"缺少字段：{missing}")
    if extra:
        lines.append(f"额外字段：{extra}")
    if changed:
        lines.append(f"定义不同：{changed}")
    return "\n        ".join(lines) if lines else "未知 schema 差异"


def validate_source_compatibility(sources: list[SourceInfo]) -> str:
    modes = {source.detected_mode for source in sources}
    if len(modes) != 1:
        details = "\n".join(
            f"        {source.detected_mode:2s}  {source.root}"
            for source in sources
        )
        fail(
            "一次合并中检测到了不同类型的数据集。\n"
            "        普通模仿学习数据和 RL 数据不能混合，因为 features 不同。\n"
            f"{details}"
        )

    detected_mode = next(iter(modes))
    if requested_mode != "auto" and requested_mode != detected_mode:
        fail(
            f"--mode={requested_mode}，但检测到的数据类型是 {detected_mode}。\n"
            "        请检查 --src，或改用 --mode auto。"
        )

    reference = sources[0]
    ref_fps = reference.info.get("fps")
    ref_robot_type = reference.info.get("robot_type")
    ref_features_json = normalized_json(reference.features)

    for source in sources[1:]:
        if source.info.get("fps") != ref_fps:
            fail(
                f"fps 不一致：\n"
                f"        {reference.root}: {ref_fps}\n"
                f"        {source.root}: {source.info.get('fps')}"
            )
        if source.info.get("robot_type") != ref_robot_type:
            fail(
                f"robot_type 不一致：\n"
                f"        {reference.root}: {ref_robot_type}\n"
                f"        {source.root}: {source.info.get('robot_type')}"
            )
        if normalized_json(source.features) != ref_features_json:
            fail(
                f"features schema 不一致：{source.root}\n"
                f"        {schema_diff(reference.features, source.features)}\n"
                "        请分别合并不同 schema 的数据，不要填零或强行拼接。"
            )

    return detected_mode


def build_repo_ids(sources: list[SourceInfo]) -> list[str]:
    repo_ids: list[str] = []
    used: set[str] = set()
    for index, source in enumerate(sources):
        base = sanitize_repo_component(source.root.name)
        repo_id = f"local/{base}"
        if repo_id in used:
            repo_id = f"local/{base}_{index:04d}"
        used.add(repo_id)
        repo_ids.append(repo_id)
    return repo_ids


def prepare_output() -> None:
    if merged_root.exists():
        if not force:
            fail(
                f"输出目录已经存在：{merged_root}\n"
                "        使用 --force 删除后重建，或使用 --out 指定新目录。"
            )
        log(f"[WARN] --force 已启用，删除旧输出：{merged_root}")
        shutil.rmtree(merged_root)


def import_lerobot() -> tuple[Any, Any]:
    try:
        import lerobot  # type: ignore

        log(f"[INFO] LeRobot 导入位置：{getattr(lerobot, '__file__', '<unknown>')}")
        log(f"[INFO] LeRobot 版本：{getattr(lerobot, '__version__', '<unknown>')}")
    except Exception as exc:
        fail(f"无法导入 lerobot：{exc}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception:
        try:
            from lerobot.datasets import LeRobotDataset  # type: ignore
        except Exception as exc:
            fail(f"无法导入 LeRobotDataset：{exc}")

    merge_datasets = None
    try:
        from lerobot.datasets.dataset_tools import merge_datasets as merge_fn  # type: ignore

        merge_datasets = merge_fn
    except Exception:
        try:
            from lerobot.datasets import merge_datasets as merge_fn  # type: ignore

            merge_datasets = merge_fn
        except Exception:
            merge_datasets = None

    return LeRobotDataset, merge_datasets


def merge_by_dataset_tools(
    LeRobotDataset: Any,
    merge_datasets: Any,
    repo_ids: list[str],
    sources: list[SourceInfo],
) -> bool:
    if merge_datasets is None:
        log("[WARN] 当前 LeRobot 没有可用的 merge_datasets。")
        return False

    try:
        signature = inspect.signature(merge_datasets)
        if "output_dir" not in signature.parameters:
            log("[WARN] merge_datasets 不支持 output_dir，转用 aggregate_datasets。")
            return False

        datasets = [
            LeRobotDataset(repo_id=repo_id, root=source.root)
            for repo_id, source in zip(repo_ids, sources, strict=True)
        ]

        log("[INFO] 使用 lerobot.datasets.dataset_tools.merge_datasets 合并。")
        merge_datasets(
            datasets=datasets,
            output_repo_id=new_repo_id,
            output_dir=merged_root,
        )
        return (merged_root / "meta" / "info.json").is_file()
    except Exception:
        log("[WARN] merge_datasets 执行失败：")
        traceback.print_exc()
        if merged_root.exists():
            shutil.rmtree(merged_root)
        return False


def merge_by_aggregate(repo_ids: list[str], sources: list[SourceInfo]) -> bool:
    try:
        from lerobot.datasets.aggregate import aggregate_datasets  # type: ignore
    except Exception as exc:
        log(f"[WARN] 无法导入 aggregate_datasets：{exc}")
        return False

    try:
        signature = inspect.signature(aggregate_datasets)
        kwargs: dict[str, Any] = {
            "repo_ids": repo_ids,
            "aggr_repo_id": new_repo_id,
        }

        if "roots" in signature.parameters:
            kwargs["roots"] = [source.root for source in sources]
        else:
            log("[WARN] aggregate_datasets 不支持 roots。")
            return False

        if "aggr_root" in signature.parameters:
            kwargs["aggr_root"] = merged_root
        elif "output_dir" in signature.parameters:
            kwargs["output_dir"] = merged_root
        else:
            log("[WARN] aggregate_datasets 不支持 aggr_root/output_dir。")
            return False

        log("[INFO] 使用 lerobot.datasets.aggregate.aggregate_datasets 合并。")
        aggregate_datasets(**kwargs)
        return (merged_root / "meta" / "info.json").is_file()
    except Exception:
        log("[WARN] aggregate_datasets 执行失败：")
        traceback.print_exc()
        if merged_root.exists():
            shutil.rmtree(merged_root)
        return False


def recursively_replace_episode_index(value: Any, new_episode_index: int) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                new_episode_index
                if key == "episode_index"
                else recursively_replace_episode_index(item, new_episode_index)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            recursively_replace_episode_index(item, new_episode_index)
            for item in value
        ]
    return value


def infer_local_episode_index(path: Path, payload: Any, total_episodes: int) -> int | None:
    if isinstance(payload, dict) and "episode_index" in payload:
        try:
            return int(payload["episode_index"])
        except Exception:
            pass

    match = re.search(r"episode[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    if total_episodes == 1:
        return 0

    return None


def copy_annotation_sidecars(sources: list[SourceInfo]) -> dict[str, Any]:
    result = {
        "enabled": copy_annotations,
        "json_mapped": 0,
        "json_unmapped": 0,
        "other_files": 0,
    }
    if not copy_annotations:
        return result

    output_dir = merged_root / "annotations"
    unmapped_dir = output_dir / "_unmapped"
    sidecar_dir = output_dir / "_sidecars"

    episode_offset = 0
    occupied_names: set[str] = set()

    for source_order, source in enumerate(sources):
        annotation_dir = source.root / "annotations"
        if not annotation_dir.is_dir():
            episode_offset += source.total_episodes
            continue

        for path in sorted(annotation_dir.rglob("*"), key=natural_key):
            if not path.is_file():
                continue

            relative = path.relative_to(annotation_dir)

            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    target = (
                        unmapped_dir
                        / f"{source_order:04d}_{source.root.name}"
                        / relative
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    result["json_unmapped"] += 1
                    continue

                local_episode = infer_local_episode_index(
                    path,
                    payload,
                    source.total_episodes,
                )

                if (
                    local_episode is None
                    or local_episode < 0
                    or local_episode >= source.total_episodes
                ):
                    target = (
                        unmapped_dir
                        / f"{source_order:04d}_{source.root.name}"
                        / relative
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    result["json_unmapped"] += 1
                    continue

                global_episode = episode_offset + local_episode
                payload = recursively_replace_episode_index(payload, global_episode)

                base_name = f"episode_{global_episode:06d}.json"
                if base_name in occupied_names or (output_dir / base_name).exists():
                    base_name = (
                        f"episode_{global_episode:06d}"
                        f"__{source_order:04d}_{sanitize_repo_component(path.stem)}.json"
                    )
                occupied_names.add(base_name)

                target = output_dir / base_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result["json_mapped"] += 1
            else:
                target = (
                    sidecar_dir
                    / f"{source_order:04d}_{source.root.name}"
                    / relative
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                result["other_files"] += 1

        episode_offset += source.total_episodes

    return result


def scalarize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        return value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        if not value:
            return np.nan
        return value[0]
    return value


def load_validation_frame(data_files: list[Path], columns: list[str]) -> pd.DataFrame:
    try:
        import pyarrow.dataset as pads  # type: ignore

        dataset = pads.dataset([str(path) for path in data_files], format="parquet")
        table = dataset.to_table(columns=columns)
        return table.to_pandas()
    except Exception as exc:
        log(f"[WARN] PyArrow dataset 批量读取失败，改用 pandas：{exc}")
        frames = [
            pd.read_parquet(path, columns=columns)
            for path in data_files
        ]
        return pd.concat(frames, ignore_index=True)


def validate_merged_dataset(
    sources: list[SourceInfo],
    detected_mode: str,
) -> dict[str, Any]:
    info_path = merged_root / "meta" / "info.json"
    if not info_path.is_file():
        fail(f"合并结果缺少 meta/info.json：{info_path}")

    merged_info = json.loads(info_path.read_text(encoding="utf-8"))
    expected_episodes = sum(source.total_episodes for source in sources)
    expected_frames = sum(source.total_frames for source in sources)

    errors: list[str] = []
    warnings: list[str] = []

    if int(merged_info.get("total_episodes", -1)) != expected_episodes:
        errors.append(
            f"total_episodes={merged_info.get('total_episodes')}，期望 {expected_episodes}"
        )
    if int(merged_info.get("total_frames", -1)) != expected_frames:
        errors.append(
            f"total_frames={merged_info.get('total_frames')}，期望 {expected_frames}"
        )
    if normalized_json(merged_info.get("features", {})) != normalized_json(sources[0].features):
        errors.append("合并后的 features 与源数据 schema 不一致")

    data_files = sorted((merged_root / "data").rglob("*.parquet"), key=natural_key)
    if not data_files:
        errors.append("合并结果中没有 data/*.parquet")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
        }

    columns = ["index", "episode_index", "frame_index"]
    if detected_mode == "rl":
        columns.extend(
            [
                "control_source",
                "is_intervention",
                "has_human_action",
                "reward",
                "done",
                "success",
            ]
        )

    try:
        frame = load_validation_frame(data_files, columns)
    except Exception as exc:
        errors.append(f"无法读取合并后的 parquet：{exc}")
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
        }

    if len(frame) != expected_frames:
        errors.append(f"parquet 行数={len(frame)}，期望 {expected_frames}")

    for column in columns:
        if column not in frame.columns:
            errors.append(f"parquet 缺少字段：{column}")

    if errors:
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
        }

    # LeRobot 必需索引检查。
    for column in ["index", "episode_index", "frame_index"]:
        frame[column] = frame[column].map(scalarize).astype(np.int64)

    indices = frame["index"].to_numpy()
    if (
        len(np.unique(indices)) != expected_frames
        or int(indices.min()) != 0
        or int(indices.max()) != expected_frames - 1
    ):
        errors.append("全局 index 不是从 0 到 total_frames-1 的连续唯一编号")

    episode_indices = frame["episode_index"].to_numpy()
    if (
        len(np.unique(episode_indices)) != expected_episodes
        or int(episode_indices.min()) != 0
        or int(episode_indices.max()) != expected_episodes - 1
    ):
        errors.append("episode_index 不是从 0 到 total_episodes-1 的连续编号")

    bad_frame_episodes: list[int] = []
    for episode_index, group in frame.groupby("episode_index", sort=True):
        values = np.sort(group["frame_index"].to_numpy())
        expected = np.arange(len(values), dtype=np.int64)
        if not np.array_equal(values, expected):
            bad_frame_episodes.append(int(episode_index))

    if bad_frame_episodes:
        errors.append(
            "以下 episode 的 frame_index 不是从 0 连续编号："
            f"{bad_frame_episodes[:20]}"
        )

    summary: dict[str, Any] = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "total_episodes": expected_episodes,
        "total_frames": expected_frames,
        "data_files": len(data_files),
    }

    if detected_mode == "rl":
        for column in [
            "control_source",
            "is_intervention",
            "has_human_action",
            "reward",
            "done",
            "success",
        ]:
            frame[column] = frame[column].map(scalarize)

        frame["done"] = frame["done"].astype(np.int64)
        frame["success"] = frame["success"].astype(np.int64)
        frame["is_intervention"] = frame["is_intervention"].astype(np.int64)
        frame["has_human_action"] = frame["has_human_action"].astype(np.int64)
        frame["reward"] = frame["reward"].astype(np.float64)

        terminal_errors: list[int] = []
        for episode_index, group in frame.groupby("episode_index", sort=True):
            group = group.sort_values("frame_index")
            done_values = group["done"].to_numpy()
            if int(done_values.sum()) != 1 or int(done_values[-1]) != 1:
                terminal_errors.append(int(episode_index))

        if terminal_errors:
            errors.append(
                "以下 RL episode 必须且只能在最后一帧 done=1："
                f"{terminal_errors[:20]}"
            )

        success_without_done = int(
            ((frame["success"] >= 1) & (frame["done"] < 1)).sum()
        )
        if success_without_done:
            errors.append(
                f"存在 {success_without_done} 帧 success=1 但 done!=1"
            )

        intervention_frames = int((frame["is_intervention"] >= 1).sum())
        human_action_frames = int((frame["has_human_action"] >= 1).sum())
        if intervention_frames == 0:
            warnings.append("RL 数据中没有 is_intervention=1 的帧")
        if human_action_frames == 0:
            warnings.append("RL 数据中没有 has_human_action=1 的帧")

        summary.update(
            {
                "reward_sum": float(frame["reward"].sum()),
                "reward_nonzero_frames": int((frame["reward"] != 0).sum()),
                "done_frames": int((frame["done"] >= 1).sum()),
                "success_frames": int((frame["success"] >= 1).sum()),
                "intervention_frames": intervention_frames,
                "human_action_frames": human_action_frames,
            }
        )

    summary["ok"] = not errors
    summary["errors"] = errors
    summary["warnings"] = warnings
    return summary


def write_manifest(
    sources: list[SourceInfo],
    detected_mode: str,
    annotations: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(src_root),
        "merged_root": str(merged_root),
        "repo_id": new_repo_id,
        "mode": detected_mode,
        "source_dataset_count": len(sources),
        "source_total_episodes": sum(source.total_episodes for source in sources),
        "source_total_frames": sum(source.total_frames for source in sources),
        "sources": [
            {
                "order": index,
                "root": str(source.root),
                "mode": source.detected_mode,
                "total_episodes": source.total_episodes,
                "total_frames": source.total_frames,
            }
            for index, source in enumerate(sources)
        ],
        "annotations": annotations,
        "validation": validation,
    }

    (merged_root / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (merged_root / "source_datasets.txt").write_text(
        "\n".join(str(source.root) for source in sources) + "\n",
        encoding="utf-8",
    )


roots = discover_dataset_roots()
if not roots:
    fail(
        f"在以下目录没有找到 LeRobot 数据集：{src_root}\n"
        "        数据集目录至少应包含 meta/info.json 和 data/。\n"
        f"        当前目录名称过滤：{include_glob!r}"
    )

sources = [load_source_info(root) for root in roots]
detected_mode = validate_source_compatibility(sources)
repo_ids = build_repo_ids(sources)

log("")
log(f"[OK] 找到 {len(sources)} 个兼容的数据集目录。")
log(f"[OK] 自动识别数据类型：{detected_mode}")
log(
    "[OK] 源数据合计："
    f"{sum(source.total_episodes for source in sources)} episodes，"
    f"{sum(source.total_frames for source in sources)} frames"
)
log("[INFO] 输入顺序：")
for index, source in enumerate(sources):
    log(
        f"       [{index:03d}] "
        f"episodes={source.total_episodes:<4d} "
        f"frames={source.total_frames:<7d} "
        f"{source.root}"
    )

if dry_run:
    log("")
    log("[DONE] dry-run 完成；未创建输出数据集。")
    raise SystemExit(0)

prepare_output()
LeRobotDataset, merge_datasets = import_lerobot()

merged_ok = merge_by_dataset_tools(
    LeRobotDataset,
    merge_datasets,
    repo_ids,
    sources,
)
if not merged_ok:
    merged_ok = merge_by_aggregate(repo_ids, sources)

if not merged_ok:
    fail(
        "所有 LeRobot 合并接口均执行失败。\n"
        "        上方已经打印完整 traceback。\n"
        "        请确认虚拟环境和 LeRobot 版本是否正确。"
    )

annotations = copy_annotation_sidecars(sources)
validation = validate_merged_dataset(sources, detected_mode)
write_manifest(sources, detected_mode, annotations, validation)

log("")
log("==============================================================================")
if validation["ok"]:
    log("[OK] 数据集合并及完整性检查通过。")
else:
    log("[ERROR] 数据集已经生成，但完整性检查未通过。")

log(f"[INFO] merged_root       ：{merged_root}")
log(f"[INFO] repo_id           ：{new_repo_id}")
log(f"[INFO] mode              ：{detected_mode}")
log(f"[INFO] total_episodes    ：{validation.get('total_episodes')}")
log(f"[INFO] total_frames      ：{validation.get('total_frames')}")
log(f"[INFO] parquet files     ：{validation.get('data_files')}")
log(f"[INFO] annotations mapped：{annotations.get('json_mapped')}")
log(f"[INFO] merge manifest    ：{merged_root / 'merge_manifest.json'}")

if detected_mode == "rl":
    log(f"[INFO] reward sum        ：{validation.get('reward_sum')}")
    log(f"[INFO] reward frames     ：{validation.get('reward_nonzero_frames')}")
    log(f"[INFO] done frames       ：{validation.get('done_frames')}")
    log(f"[INFO] success frames    ：{validation.get('success_frames')}")
    log(f"[INFO] intervention      ：{validation.get('intervention_frames')}")
    log(f"[INFO] human action      ：{validation.get('human_action_frames')}")

for warning in validation.get("warnings", []):
    log(f"[WARN] {warning}")
for error in validation.get("errors", []):
    log(f"[ERROR] {error}")

log("==============================================================================")

if not validation["ok"]:
    raise SystemExit(
        "[ERROR] 合并结果未通过检查，暂时不要用于训练。"
    )

log("")
log(f"[DONE] 合并数据集可以用于训练：{merged_root}")
PY

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

DETECTED_MODE="$(
  python3 - "${MERGED_ROOT}/merge_manifest.json" <<'PYMODE'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    print(json.loads(path.read_text(encoding="utf-8"))["mode"])
except Exception:
    print("auto")
PYMODE
)"

echo ""
echo "[NEXT] 建议先运行数据检查程序："
echo "  cd ${HOME}/mycode/bw_residual_rl_code_package/lerobot_bw_data_collector"
echo "  bash scripts/check_dataset.sh \\"
echo "    ${MERGED_ROOT} \\"
echo "    --all-episodes \\"
echo "    --out-dir ${MERGED_ROOT}_check \\"
echo "    --mode ${DETECTED_MODE}"
echo ""
echo "[DONE] ${MERGED_ROOT}"
