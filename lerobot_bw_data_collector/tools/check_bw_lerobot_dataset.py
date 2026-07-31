#!/usr/bin/env python3
"""Check and visualize BW LeRobot BC/RL datasets.

This script is designed for the BW data collector in this project.

BC mode keeps the old dataset-check behavior:
- reads `observation.state` and `action`
- draws action/state curves, per-joint errors, correlations, ranges
- optionally exports CSV files
- draws camera contact sheets from saved videos

RL mode is for datasets recorded by `bw_residual_rl_code`:
- required columns: control_source, is_intervention, action.act,
  action.rl_delta, action.human, action.executed, reward, done, success, timing.*
- checks action-source consistency, intervention periods, residual target, reward
  keyboard reward consistency, and transition validity
- draws extra RL plots and three types of camera contact sheets:
  uniform frames, intervention-event frames, reward/done-event frames.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_yaw_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_gripper_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_yaw_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_gripper_joint",
]

LEFT_ARM = JOINT_NAMES[0:7]
RIGHT_ARM = JOINT_NAMES[8:15]
GRIPPERS = ["left_gripper_joint", "right_gripper_joint"]
CAMERA_NAMES = ["env_cam", "left_wrist_cam", "right_wrist_cam"]

OBS_STATE = "observation.state"
ACTION = "action"
RL_COLUMNS = [
    "control_source",
    "is_intervention",
    "has_human_action",
    "action.act",
    "action.rl_delta",
    "action.human",
    "action.executed",
    "action.gripper_policy_class",
    "reward",
    "done",
    "success",
]
TIMING_COLUMNS = [
    "timing.arm_action_dt",
    "timing.gripper_action_dt",
    "timing.action_act_dt",
    "timing.action_final_dt",
]


@dataclass(slots=True)
class Args:
    dataset_root: Path
    episode: int | None
    all_episodes: bool
    mode: str
    out_dir: Path | None
    save_csv: bool
    no_video_sheet: bool
    contact_sheet_frames: int
    dpi: int
    residual_lambda: float | None
    reconstruction_threshold: float
    max_timing_dt: float
    fps_tolerance: float
    residual_limit_default: float
    residual_limit_gripper: float
    strict_reconstruction: bool


@dataclass(slots=True)
class EpisodeResult:
    episode_index: int
    mode: str
    out_dir: Path
    frames: int
    duration_s: float
    fps_est: float
    warnings: list[dict[str, Any]]
    stats: dict[str, Any]


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Check and visualize BW LeRobot imitation-learning or residual-RL datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset_root", type=Path, help="Extracted LeRobot dataset folder containing meta/ and data/.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--episode", type=int, default=None, help="Check only one episode_index.")
    group.add_argument("--all-episodes", action="store_true", help="Check all episodes. This is also the default.")
    p.add_argument("--mode", choices=["auto", "bc", "rl"], default="auto", help="Check mode. auto uses RL mode when RL columns exist.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: DATASET_ROOT/check_report.")
    p.add_argument("--save-csv", action="store_true", help="Save expanded per-frame CSV files.")
    p.add_argument("--no-video-sheet", action="store_true", help="Skip video contact sheets.")
    p.add_argument("--contact-sheet-frames", type=int, default=8, help="Frames sampled per camera contact sheet.")
    p.add_argument("--dpi", type=int, default=150, help="DPI for generated figures.")
    p.add_argument("--residual-lambda", type=float, default=None, help="Override residual lambda. If omitted, the script tries dataset metadata, then falls back to 0.2.")
    p.add_argument("--reconstruction-threshold", type=float, default=0.02, help="Max allowed action reconstruction error before warning.")
    p.add_argument("--max-timing-dt", type=float, default=0.15, help="Warn when abs(timing.*) is larger than this many seconds.")
    p.add_argument("--fps-tolerance", type=float, default=0.25, help="Warn when estimated fps differs from meta fps by more than this ratio.")
    p.add_argument("--residual-limit-default", type=float, default=0.03, help="Default joint-space residual limit for residual saturation checks.")
    p.add_argument("--residual-limit-gripper", type=float, default=0.03, help="Gripper residual limit for residual saturation checks.")
    p.add_argument("--strict-reconstruction", action="store_true", help="Treat non-intervention executed≈act+lambda*delta mismatch as a strong warning. With smoothing enabled, mismatches can be expected.")
    ns = p.parse_args()
    return Args(**vars(ns))


def ensure_dataset_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist or is not a directory: {root}")
    missing = [str(root / name) for name in ["meta", "data"] if not (root / name).exists()]
    if missing:
        raise FileNotFoundError("This does not look like an extracted LeRobot dataset. Missing:\n  " + "\n  ".join(missing))
    return root


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_info(root: Path) -> dict[str, Any]:
    return load_json(root / "meta" / "info.json")


def _flatten_names(names: Any) -> list[str] | None:
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        out: list[str] = []
        for v in names.values():
            if isinstance(v, list):
                out.extend(str(x) for x in v)
        return out or None
    return None


def _names_from_feature(feature: Any) -> list[str] | None:
    if not isinstance(feature, dict):
        return None
    for key in ["names", "dtype_names"]:
        names = _flatten_names(feature.get(key))
        if names:
            return names
    return None


def strip_pos_suffix(names: list[str]) -> list[str]:
    return [name[:-4] if name.endswith(".pos") else name for name in names]


def get_joint_names(info: dict[str, Any], dim: int) -> list[str]:
    features = info.get("features", {}) if isinstance(info, dict) else {}
    if isinstance(features, dict):
        for key in [ACTION, OBS_STATE, "action.executed", "action.act"]:
            names = _names_from_feature(features.get(key))
            if names and len(names) == dim:
                return strip_pos_suffix(names)
    if dim == len(JOINT_NAMES):
        return JOINT_NAMES.copy()
    return [f"joint_{i:02d}" for i in range(dim)]


def find_data_parquet_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return files


def read_all_data(root: Path) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for path in find_data_parquet_files(root):
        try:
            dfs.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}", file=sys.stderr)
    if not dfs:
        raise RuntimeError("No readable parquet files found.")
    df = pd.concat(dfs, ignore_index=True)
    sort_cols = [c for c in ["episode_index", "frame_index", "timestamp", "index"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def list_episode_indices(df: pd.DataFrame) -> list[int]:
    if "episode_index" not in df.columns:
        return [0]
    vals = pd.to_numeric(df["episode_index"], errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(vals)


def episode_dataframe(df: pd.DataFrame, episode_index: int) -> pd.DataFrame:
    if "episode_index" not in df.columns:
        if episode_index != 0:
            raise ValueError("Dataset has no episode_index column; only episode 0 can be checked.")
        out = df.copy()
    else:
        ep_values = pd.to_numeric(df["episode_index"], errors="coerce")
        mask = ep_values.fillna(-999999999).astype(int) == int(episode_index)
        out = df[mask].copy()
    if out.empty:
        raise ValueError(f"No rows found for episode_index={episode_index}")
    sort_cols = [c for c in ["frame_index", "timestamp", "index"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def as_vector(value: Any, *, dim: int | None = None, default: float | None = None) -> np.ndarray:
    if value is None:
        if dim is None or default is None:
            raise ValueError("Cannot convert None to vector without dim/default")
        return np.full((dim,), default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if dim is not None and arr.size != dim:
        if arr.size == 1 and default is not None:
            return np.full((dim,), float(arr[0]), dtype=np.float64)
        raise ValueError(f"Expected vector dim={dim}, got {arr.size}")
    return arr


def stack_vector_column(df: pd.DataFrame, col: str, *, dim: int | None = None) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")
    rows: list[np.ndarray] = []
    for i, value in enumerate(df[col].to_numpy()):
        try:
            arr = as_vector(value, dim=dim)
        except Exception as exc:
            raise ValueError(f"Column {col} has invalid vector at row {i}: {exc}") from exc
        rows.append(arr)
    dims = sorted({x.size for x in rows})
    if len(dims) != 1:
        raise ValueError(f"Column {col} has inconsistent vector lengths: {dims}")
    return np.vstack(rows)


def as_scalar(value: Any, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        try:
            return float(value)
        except Exception:
            return float(default)


def scalar_column(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in df.columns:
        return np.full((len(df),), float(default), dtype=np.float64)
    return np.asarray([as_scalar(v, default=default) for v in df[col].to_numpy()], dtype=np.float64)


def get_time_vector(df: pd.DataFrame, n: int) -> np.ndarray:
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        if len(t) == n and np.isfinite(t).all():
            return t - t[0]
    if "frame_index" in df.columns:
        f = pd.to_numeric(df["frame_index"], errors="coerce").to_numpy(dtype=np.float64)
        if len(f) == n and np.isfinite(f).all():
            return f - f[0]
    return np.arange(n, dtype=np.float64)


def estimate_duration_fps(t: np.ndarray, n: int, info: dict[str, Any]) -> tuple[float, float]:
    if n <= 1:
        return 0.0, float("nan")
    duration = float(t[-1] - t[0])
    if duration > 1e-9 and np.nanmax(t) < 100000:
        return duration, (n - 1) / duration
    fps = float(info.get("fps", 0) or 0)
    if fps > 0:
        return float((n - 1) / fps), fps
    return float(n - 1), float("nan")


def x_label_for_time(t: np.ndarray) -> str:
    return "time / s" if len(t) > 1 and np.nanmax(t) <= 100000 else "frame"


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_action_state_stats(action: np.ndarray, state: np.ndarray, joint_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(joint_names):
        a = action[:, i]
        s = state[:, i]
        err = a - s
        rows.append({
            "joint_index": i,
            "joint_name": name,
            "action_min": float(np.nanmin(a)),
            "action_max": float(np.nanmax(a)),
            "action_range": float(np.nanmax(a) - np.nanmin(a)),
            "state_min": float(np.nanmin(s)),
            "state_max": float(np.nanmax(s)),
            "state_range": float(np.nanmax(s) - np.nanmin(s)),
            "mean_error_action_minus_state": float(np.nanmean(err)),
            "rmse_action_minus_state": float(np.sqrt(np.nanmean(err ** 2))),
            "max_abs_error": float(np.nanmax(np.abs(err))),
            "corr_action_state": safe_corr(a, s),
            "corr_action_negative_state": safe_corr(a, -s),
        })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def safe_name(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", str(text))
    return text.strip("_") or "item"


def indices_for_names(joint_names: list[str], selected: Iterable[str]) -> list[int]:
    return [joint_names.index(name) for name in selected if name in joint_names]


def plot_lines(t: np.ndarray, values: np.ndarray, names: list[str], title: str, ylabel: str, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(names):
        ax.plot(t, values[:, i], label=name, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(x_label_for_time(t))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, out_path, dpi)


def plot_group_curves(t: np.ndarray, action: np.ndarray, state: np.ndarray, joint_names: list[str], out_dir: Path, dpi: int) -> None:
    for group_name, names in [("left_arm", LEFT_ARM), ("right_arm", RIGHT_ARM), ("grippers", GRIPPERS)]:
        idx = indices_for_names(joint_names, names)
        if not idx:
            continue
        group_names = [joint_names[i] for i in idx]
        plot_lines(t, action[:, idx], group_names, f"Action curves - {group_name}", "action position", out_dir / f"action_{group_name}.png", dpi)
        plot_lines(t, state[:, idx], group_names, f"Observation state curves - {group_name}", "state position", out_dir / f"state_{group_name}.png", dpi)


def plot_single_joint_comparison(t: np.ndarray, action: np.ndarray, state: np.ndarray, joint_name: str, joint_index: int, stats_row: pd.Series, out_path: Path, dpi: int) -> None:
    a = action[:, joint_index]
    s = state[:, joint_index]
    err = a - s
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2.2, 1.0])
    axes[0].plot(t, a, label="action", linewidth=1.4)
    axes[0].plot(t, s, label="observation.state", linewidth=1.4, linestyle="--")
    axes[0].set_ylabel("position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(t, err, label="action - state", linewidth=1.1)
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xlabel(x_label_for_time(t))
    axes[1].set_ylabel("error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(
        f"{joint_index:02d} {joint_name}\n"
        f"RMSE={stats_row['rmse_action_minus_state']:.6g}, "
        f"max_abs_error={stats_row['max_abs_error']:.6g}, "
        f"corr(a,s)={stats_row['corr_action_state']:.3f}, "
        f"corr(a,-s)={stats_row['corr_action_negative_state']:.3f}",
        fontsize=11,
    )
    save_figure(fig, out_path, dpi)


def plot_all_per_joint(t: np.ndarray, action: np.ndarray, state: np.ndarray, joint_names: list[str], stats: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    per_joint = out_dir / "per_joint_action_vs_state"
    per_joint.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(joint_names):
        plot_single_joint_comparison(t, action, state, name, i, stats.iloc[i], per_joint / f"{i:02d}_{safe_name(name)}.png", dpi)


def plot_all_joints_grid(t: np.ndarray, action: np.ndarray, state: np.ndarray, joint_names: list[str], out_path: Path, dpi: int) -> None:
    dim = len(joint_names)
    ncols = 4
    nrows = math.ceil(dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.3 * nrows), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for i in range(dim):
        ax = axes_flat[i]
        ax.plot(t, action[:, i], label="action", linewidth=1.0)
        ax.plot(t, state[:, i], label="state", linewidth=1.0, linestyle="--")
        ax.set_title(f"{i:02d} {joint_names[i]}", fontsize=9)
        ax.grid(True, alpha=0.25)
        if i % ncols == 0:
            ax.set_ylabel("pos")
        if i >= dim - ncols:
            ax.set_xlabel(x_label_for_time(t))
    for j in range(dim, len(axes_flat)):
        axes_flat[j].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("All joints: action vs observation.state", fontsize=14)
    save_figure(fig, out_path, dpi)


def plot_selected_action_vs_state(t: np.ndarray, action: np.ndarray, state: np.ndarray, joint_names: list[str], out_path: Path, dpi: int) -> None:
    preferred = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
    ]
    idx = indices_for_names(joint_names, preferred) or list(range(min(8, len(joint_names))))
    ncols = 2
    nrows = math.ceil(len(idx) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for k, i in enumerate(idx):
        ax = axes_flat[k]
        ax.plot(t, action[:, i], label="action", linewidth=1.1)
        ax.plot(t, state[:, i], label="state", linewidth=1.1, linestyle="--")
        ax.set_title(f"{i:02d} {joint_names[i]}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(len(idx), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Selected joints: action vs observation.state", fontsize=14)
    save_figure(fig, out_path, dpi)


def plot_gripper_action_vs_state(t: np.ndarray, action: np.ndarray, state: np.ndarray, joint_names: list[str], out_path: Path, dpi: int) -> None:
    idx = indices_for_names(joint_names, GRIPPERS)
    if not idx:
        return
    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 4 * len(idx)), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, i in zip(axes_flat, idx):
        ax.plot(t, action[:, i], label="action", linewidth=1.4)
        ax.plot(t, state[:, i], label="state", linewidth=1.4, linestyle="--")
        ax.set_title(f"{i:02d} {joint_names[i]}")
        ax.set_ylabel("position")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes_flat[-1].set_xlabel(x_label_for_time(t))
    fig.suptitle("Gripper action vs observation.state", fontsize=14)
    save_figure(fig, out_path, dpi)


def plot_bar(stats: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(stats))
    ax.bar(x, stats[value_col].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i:02d}\n{name}" for i, name in enumerate(stats["joint_name"])], rotation=75, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, out_path, dpi)


def plot_correlation_sign_check(stats: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(stats))
    width = 0.38
    ax.bar(x - width / 2, stats["corr_action_state"].to_numpy(dtype=float), width, label="corr(action, state)")
    ax.bar(x + width / 2, stats["corr_action_negative_state"].to_numpy(dtype=float), width, label="corr(action, -state)")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i:02d}\n{name}" for i, name in enumerate(stats["joint_name"])], rotation=75, ha="right")
    ax.set_title("Sign check: correlation with state and negative state")
    ax.set_ylabel("correlation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, out_path, dpi)


def plot_range_comparison(stats: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(stats))
    width = 0.38
    ax.bar(x - width / 2, stats["action_range"].to_numpy(dtype=float), width, label="action range")
    ax.bar(x + width / 2, stats["state_range"].to_numpy(dtype=float), width, label="state range")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i:02d}\n{name}" for i, name in enumerate(stats["joint_name"])], rotation=75, ha="right")
    ax.set_title("Action/state value range comparison")
    ax.set_ylabel("max - min")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, out_path, dpi)


def save_bc_csv_files(out_dir: Path, t: np.ndarray, action: np.ndarray, state: np.ndarray, joint_names: list[str], stats: pd.DataFrame) -> None:
    action_df = pd.DataFrame(action, columns=joint_names)
    state_df = pd.DataFrame(state, columns=joint_names)
    action_df.insert(0, "time", t)
    state_df.insert(0, "time", t)
    action_df.to_csv(out_dir / "action.csv", index=False)
    state_df.to_csv(out_dir / "observation_state.csv", index=False)
    combined = pd.DataFrame({"time": t})
    for i, name in enumerate(joint_names):
        combined[f"action.{name}"] = action[:, i]
        combined[f"state.{name}"] = state[:, i]
        combined[f"error.{name}"] = action[:, i] - state[:, i]
    combined.to_csv(out_dir / "action_and_state.csv", index=False)
    stats.to_csv(out_dir / "per_joint_stats.csv", index=False)


def find_video_files(root: Path) -> dict[str, Path]:
    videos_dir = root / "videos"
    out: dict[str, Path] = {}
    if not videos_dir.exists():
        return out
    for file in sorted(videos_dir.rglob("*.mp4")):
        rel = file.relative_to(videos_dir)
        label = rel.parts[0] if rel.parts else file.stem
        camera = label.replace("observation.images.", "")
        # Keep the first file for each camera. BW single-session datasets usually have file-000.mp4.
        out.setdefault(camera, file)
    return out


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = frame_index + 1
    frame_index = max(0, min(int(frame_index), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def video_frame_count(video_path: Path) -> int:
    if cv2 is None:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(total, 0)


def select_even_frames(n_data: int, n_samples: int) -> list[int]:
    if n_data <= 0:
        return []
    return np.linspace(0, n_data - 1, max(int(n_samples), 1)).astype(int).tolist()


def select_event_frames(mask: np.ndarray, n_samples: int, *, pad: int = 2) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    event_candidates: list[int] = []
    # Include transition starts and ends first.
    prev = np.r_[False, mask[:-1]]
    nxt = np.r_[mask[1:], False]
    starts = np.flatnonzero(mask & ~prev)
    ends = np.flatnonzero(mask & ~nxt)
    for v in starts.tolist() + ends.tolist():
        for u in [v - pad, v, v + pad]:
            if 0 <= u < len(mask):
                event_candidates.append(int(u))
    # If still too few, sample all mask positions.
    if len(set(event_candidates)) < n_samples:
        extra = idx[np.linspace(0, idx.size - 1, min(n_samples, idx.size)).astype(int)].tolist()
        event_candidates.extend(int(v) for v in extra)
    # Unique while keeping order, then downsample.
    seen: set[int] = set()
    unique = []
    for v in event_candidates:
        if v not in seen:
            unique.append(v)
            seen.add(v)
    if len(unique) > n_samples:
        pick = np.linspace(0, len(unique) - 1, n_samples).astype(int).tolist()
        unique = [unique[i] for i in pick]
    return unique


def plot_video_contact_sheet(root: Path, out_path: Path, frame_ids: list[int], title: str, dpi: int) -> None:
    if cv2 is None:
        print("Warning: OpenCV is not installed; skip video contact sheet.", file=sys.stderr)
        return
    videos = find_video_files(root)
    if not videos:
        print("Warning: no mp4 files found under videos/; skip video contact sheet.", file=sys.stderr)
        return
    cameras = [cam for cam in CAMERA_NAMES if cam in videos] + [cam for cam in sorted(videos) if cam not in CAMERA_NAMES]
    if not cameras or not frame_ids:
        return
    nrows = len(cameras)
    ncols = len(frame_ids)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows))
    axes_arr = np.asarray(axes).reshape(nrows, ncols)
    for r, cam in enumerate(cameras):
        video_path = videos[cam]
        for c, frame_id in enumerate(frame_ids):
            ax = axes_arr[r, c]
            img = read_video_frame(video_path, frame_id)
            if img is not None:
                ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"frame {frame_id}", fontsize=9)
            if c == 0:
                ax.text(-0.02, 0.5, cam, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(title, fontsize=14)
    save_figure(fig, out_path, dpi)


def add_warning(warnings: list[dict[str, Any]], severity: str, code: str, message: str, **extra: Any) -> None:
    row = {"severity": severity, "code": code, "message": message}
    row.update(extra)
    warnings.append(row)


def detect_mode(df: pd.DataFrame, requested: str) -> str:
    if requested in {"bc", "rl"}:
        return requested
    return "rl" if all(col in df.columns for col in ["action.act", "action.executed", "is_intervention"]) else "bc"


def find_residual_lambda(root: Path, info: dict[str, Any], override: float | None) -> tuple[float, str]:
    if override is not None:
        return float(override), "command line --residual-lambda"
    candidates: list[tuple[str, Any]] = []
    # Common metadata locations. The current collector package usually does not write this yet.
    for key in ["residual_lambda", "lambda", "residual.lambda", "inference.residual.lambda"]:
        candidates.append((f"meta/info.json:{key}", info.get(key)))
    metadata = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    for key in ["residual_lambda", "lambda"]:
        candidates.append((f"meta/info.json:metadata.{key}", metadata.get(key)))
    for path_name in ["config.json", "metadata.json", "run_config.json"]:
        data = load_json(root / path_name)
        if data:
            candidates.append((f"{path_name}:residual_lambda", data.get("residual_lambda")))
            residual = data.get("residual") if isinstance(data.get("residual"), dict) else {}
            candidates.append((f"{path_name}:residual.lambda", residual.get("lambda")))
    for source, value in candidates:
        if value is None:
            continue
        try:
            return float(value), source
        except Exception:
            continue
    return 0.2, "default 0.2; not found in dataset metadata"


def _near(value: float, target: float, tol: float = 1e-6) -> bool:
    return bool(np.isfinite(value) and abs(float(value) - float(target)) <= tol)


def is_binary_like(values: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    return np.isfinite(vals) & ((np.abs(vals) <= tol) | (np.abs(vals - 1.0) <= tol))


def infer_keyboard_reward_event_type(reward_value: float, done_value: float, success_value: float) -> str:
    is_done = done_value >= 0.5
    is_success = success_value >= 0.5
    if _near(reward_value, 1.0) and not is_done and not is_success:
        return "left_stage_done_key_a"
    if _near(reward_value, 2.0) and is_done and is_success:
        return "right_stage_done_success_key_d"
    if _near(reward_value, 1.0) and is_done and is_success:
        return "manual_success_key_g"
    if _near(reward_value, 0.0) and is_done and not is_success:
        return "manual_failure_key_j"
    if abs(reward_value) > 1e-12 and is_done and not is_success:
        return "nonzero_reward_terminal_failure"
    if abs(reward_value) > 1e-12:
        return "custom_nonzero_reward"
    if is_done or is_success:
        return "terminal_no_reward"
    return "none"


def check_keyboard_reward_semantics(reward: np.ndarray, done: np.ndarray, success: np.ndarray, warnings: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    reward = np.asarray(reward, dtype=np.float64)
    done = np.asarray(done, dtype=np.float64)
    success = np.asarray(success, dtype=np.float64)
    n = len(reward)
    tol = 1e-6

    for name, values in [("reward", reward), ("done", done), ("success", success)]:
        if not np.isfinite(values).all():
            add_warning(warnings, "ERROR", f"{name}_nan_inf", f"{name} contains NaN or Inf.")

    if np.isfinite(reward).any() and np.nanmin(reward) < -tol:
        add_warning(warnings, "WARN", "reward_negative", "reward contains negative values. Current keyboard reward rule expects non-negative rewards.", min_reward=float(np.nanmin(reward)))

    valid_reward_values = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    finite_reward = reward[np.isfinite(reward)]
    if finite_reward.size:
        distance = np.min(np.abs(finite_reward.reshape(-1, 1) - valid_reward_values.reshape(1, -1)), axis=1)
        bad_count = int(np.sum(distance > tol))
        if bad_count > 0:
            unique_bad = sorted({float(v) for v, d in zip(finite_reward.tolist(), distance.tolist()) if d > tol})
            add_warning(warnings, "WARN", "unexpected_reward_values", f"reward contains {bad_count} value(s) outside expected keyboard set {{0,1,2}}.", values=unique_bad[:20])

    for name, values in [("done", done), ("success", success)]:
        bad = ~is_binary_like(values, tol=tol)
        if np.any(bad):
            add_warning(warnings, "WARN", f"{name}_not_binary", f"{name} contains values other than 0/1.", count=int(np.sum(bad)))

    if np.nanmax(np.abs(reward)) < 1e-12:
        add_warning(warnings, "WARN", "reward_all_zero", "reward is all zero. For RL training data, press a/d/g during successful collection or j for a marked failure.")

    done_idx = np.flatnonzero(done >= 0.5)
    success_idx = np.flatnonzero(success >= 0.5)
    if n == 0:
        add_warning(warnings, "ERROR", "empty_episode", "Episode has zero frames.")
    elif done_idx.size == 0:
        add_warning(warnings, "WARN", "missing_done", "No done=1 frame found. A normally saved keyboard-labeled RL episode should end with d/g/j.")
    elif not (done_idx.size == 1 and done_idx[-1] == n - 1):
        add_warning(warnings, "WARN", "done_not_only_final", f"done=1 appears at frames {done_idx.tolist()}, expected only final frame {n-1}.")

    if success_idx.size > 0 and not (success_idx.size == 1 and success_idx[-1] == n - 1):
        add_warning(warnings, "WARN", "success_not_only_final", f"success=1 appears at frames {success_idx.tolist()}, expected only final frame {n-1} for keyboard-labeled episodes.")
    if np.nanmax(success) > 0 and np.nanmax(done) < 0.5:
        add_warning(warnings, "WARN", "success_without_done", "success is marked but done is never true.")

    if n > 0:
        last_reward = float(reward[-1])
        last_done = float(done[-1])
        last_success = float(success[-1])
        if last_done < 0.5:
            add_warning(warnings, "WARN", "last_frame_not_done", "Final frame is not done=1. Press d/g/j to end an RL episode cleanly.")
        if last_success >= 0.5 and last_reward < 1.0 - tol:
            add_warning(warnings, "WARN", "terminal_success_reward_too_small", "Final frame has success=1 but reward < 1. Expected g -> reward=1 or d -> reward=2.", last_reward=last_reward)
        if last_success < 0.5 and last_done >= 0.5 and abs(last_reward) > tol:
            add_warning(warnings, "WARN", "terminal_failure_nonzero_reward", "Final frame is a failure but has nonzero reward. With key j, terminal reward should be 0.", last_reward=last_reward)

    d_like = np.flatnonzero(np.abs(reward - 2.0) <= tol)
    if d_like.size > 0:
        bad_d = [int(i) for i in d_like.tolist() if not (done[i] >= 0.5 and success[i] >= 0.5)]
        if bad_d:
            add_warning(warnings, "WARN", "reward_2_not_terminal_success", f"reward=2 should come from key d and should also have done=1, success=1. Bad frames: {bad_d}.", frames=bad_d)

    event_rows: list[dict[str, Any]] = []
    event_mask = (np.abs(reward) > 1e-12) | (done >= 0.5) | (success >= 0.5)
    for i in np.flatnonzero(event_mask).tolist():
        event_rows.append({
            "frame": int(i),
            "reward": float(reward[i]),
            "done": float(done[i]),
            "success": float(success[i]),
            "event_type_guess": infer_keyboard_reward_event_type(float(reward[i]), float(done[i]), float(success[i])),
        })
    reward_events = pd.DataFrame(event_rows, columns=["frame", "reward", "done", "success", "event_type_guess"])

    nonzero_reward_frames = int(np.sum(np.abs(reward) > 1e-12))
    terminal_event_type = infer_keyboard_reward_event_type(float(reward[-1]), float(done[-1]), float(success[-1])) if n > 0 else "empty"
    stats = {
        "reward_sum_parquet": float(np.nansum(reward)),
        "reward_event_count": int(len(event_rows)),
        "reward_nonzero_frames": nonzero_reward_frames,
        "last_reward": float(reward[-1]) if n > 0 else float("nan"),
        "last_done": float(done[-1]) if n > 0 else float("nan"),
        "last_success": float(success[-1]) if n > 0 else float("nan"),
        "terminal_event_type": terminal_event_type,
        "keyboard_left_stage_events": int(np.sum((np.abs(reward - 1.0) <= tol) & (done < 0.5) & (success < 0.5))),
        "keyboard_right_success_events": int(np.sum((np.abs(reward - 2.0) <= tol) & (done >= 0.5) & (success >= 0.5))),
        "keyboard_manual_success_events": int(np.sum((np.abs(reward - 1.0) <= tol) & (done >= 0.5) & (success >= 0.5))),
        "keyboard_failure_events": int(np.sum((np.abs(reward) <= tol) & (done >= 0.5) & (success < 0.5))),
    }
    return reward_events, stats


def residual_limits(default: float, gripper: float, dim: int, joint_names: list[str]) -> np.ndarray:
    limits = np.full((dim,), abs(float(default)), dtype=np.float64)
    for i, name in enumerate(joint_names):
        if "gripper" in name:
            limits[i] = abs(float(gripper))
    return limits


def plot_rl_timelines(t: np.ndarray, control: np.ndarray, intervention: np.ndarray, reward: np.ndarray, done: np.ndarray, success: np.ndarray, out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(t, control, where="post", label="control_source", linewidth=1.4)
    ax.step(t, intervention, where="post", label="is_intervention", linewidth=1.4)
    ax.set_title("Control source and intervention timeline")
    ax.set_xlabel(x_label_for_time(t))
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir / "control_source_timeline.png", dpi)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(t, intervention, where="post", label="is_intervention", linewidth=1.4)
    ax.set_title("Intervention timeline")
    ax.set_xlabel(x_label_for_time(t))
    ax.set_ylabel("0/1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir / "intervention_timeline.png", dpi)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(t, reward, where="post", label="reward", linewidth=1.2)
    ax.step(t, done, where="post", label="done", linewidth=1.2)
    ax.step(t, success, where="post", label="success", linewidth=1.2)
    ax.set_title("Reward, done and success timeline")
    ax.set_xlabel(x_label_for_time(t))
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir / "reward_done_success_timeline.png", dpi)


def plot_action_source_overview(t: np.ndarray, act: np.ndarray, human: np.ndarray, executed: np.ndarray, joint_names: list[str], out_dir: Path, dpi: int) -> None:
    selected = indices_for_names(joint_names, [
        "left_shoulder_pitch_joint", "left_elbow_joint", "left_gripper_joint",
        "right_shoulder_pitch_joint", "right_elbow_joint", "right_gripper_joint",
    ]) or list(range(min(6, len(joint_names))))
    ncols = 2
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for k, i in enumerate(selected):
        ax = axes_flat[k]
        ax.plot(t, act[:, i], label="action.act", linewidth=1.0)
        ax.plot(t, executed[:, i], label="action.executed", linewidth=1.0)
        ax.plot(t, human[:, i], label="action.human", linewidth=1.0, linestyle="--")
        ax.set_title(f"{i:02d} {joint_names[i]}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Action source overview: ACT vs human vs executed", fontsize=14)
    save_figure(fig, out_dir / "action_source_overview.png", dpi)


def plot_norm_curve(t: np.ndarray, values: np.ndarray, title: str, ylabel: str, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t, values, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(x_label_for_time(t))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_path, dpi)


def plot_vector_grid(t: np.ndarray, a: np.ndarray, b: np.ndarray, joint_names: list[str], label_a: str, label_b: str, title: str, out_path: Path, dpi: int) -> None:
    dim = len(joint_names)
    ncols = 4
    nrows = math.ceil(dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for i in range(dim):
        ax = axes_flat[i]
        ax.plot(t, a[:, i], label=label_a, linewidth=1.0)
        ax.plot(t, b[:, i], label=label_b, linewidth=1.0, linestyle="--")
        ax.set_title(f"{i:02d} {joint_names[i]}", fontsize=9)
        ax.grid(True, alpha=0.25)
    for j in range(dim, len(axes_flat)):
        axes_flat[j].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title, fontsize=14)
    save_figure(fig, out_path, dpi)


def plot_residual_target_grid(t: np.ndarray, residual_target: np.ndarray, rl_delta: np.ndarray, joint_names: list[str], out_path: Path, dpi: int) -> None:
    dim = len(joint_names)
    ncols = 4
    nrows = math.ceil(dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for i in range(dim):
        ax = axes_flat[i]
        ax.plot(t, residual_target[:, i], label="human - act", linewidth=1.0)
        ax.plot(t, rl_delta[:, i], label="rl_delta", linewidth=1.0, linestyle="--")
        ax.axhline(0.0, linewidth=0.7)
        ax.set_title(f"{i:02d} {joint_names[i]}", fontsize=9)
        ax.grid(True, alpha=0.25)
    for j in range(dim, len(axes_flat)):
        axes_flat[j].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Residual target and policy delta", fontsize=14)
    save_figure(fig, out_path, dpi)


def plot_reconstruction_error(t: np.ndarray, err_norm: np.ndarray, threshold: float, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t, err_norm, linewidth=1.2, label="||executed - reconstructed||∞")
    ax.axhline(threshold, linewidth=1.0, linestyle="--", label="threshold")
    ax.set_title("Executed action reconstruction error")
    ax.set_xlabel(x_label_for_time(t))
    ax.set_ylabel("max abs error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, out_path, dpi)


def save_rl_csv_files(out_dir: Path, t: np.ndarray, joint_names: list[str], arrays: dict[str, np.ndarray], scalars: dict[str, np.ndarray]) -> None:
    df = pd.DataFrame({"time": t})
    for name, values in scalars.items():
        df[name] = values
    for prefix, arr in arrays.items():
        for i, joint in enumerate(joint_names):
            df[f"{prefix}.{joint}"] = arr[:, i]
    df.to_csv(out_dir / "rl_expanded_timeseries.csv", index=False)


def write_text_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_common_quality(df_ep: pd.DataFrame, t: np.ndarray, action: np.ndarray, state: np.ndarray, info: dict[str, Any], args: Args, warnings: list[dict[str, Any]]) -> None:
    if action.shape != state.shape:
        add_warning(warnings, "ERROR", "shape_mismatch", f"action shape {action.shape} != observation.state shape {state.shape}")
    if action.shape[1] != 16:
        add_warning(warnings, "WARN", "unexpected_action_dim", f"Expected 16-D action/state, got {action.shape[1]}.")
    for name, arr in [("action", action), ("observation.state", state)]:
        if not np.isfinite(arr).all():
            add_warning(warnings, "ERROR", "nan_inf", f"{name} contains NaN or Inf.")
    if len(t) > 2 and np.nanmax(t) <= 100000:
        dt = np.diff(t)
        if np.any(dt < -1e-9):
            add_warning(warnings, "ERROR", "time_not_monotonic", "timestamp/frame time is not monotonic.")
        meta_fps = float(info.get("fps", 0) or 0)
        duration, fps_est = estimate_duration_fps(t, len(t), info)
        if meta_fps > 0 and np.isfinite(fps_est) and abs(fps_est - meta_fps) / meta_fps > args.fps_tolerance:
            add_warning(warnings, "WARN", "fps_mismatch", f"Estimated fps {fps_est:.3f} differs from meta fps {meta_fps:.3f}.", estimated_fps=fps_est, meta_fps=meta_fps, duration_s=duration)


def check_video_quality(root: Path, n_frames: int, warnings: list[dict[str, Any]]) -> None:
    videos = find_video_files(root)
    if not videos:
        add_warning(warnings, "WARN", "missing_videos", "No mp4 files found under videos/.")
        return
    for cam in CAMERA_NAMES:
        if cam not in videos:
            add_warning(warnings, "WARN", "missing_camera_video", f"Missing video for camera {cam}.", camera=cam)
            continue
        total = video_frame_count(videos[cam]) if cv2 is not None else 0
        if total > 0 and abs(total - n_frames) > max(5, int(0.1 * n_frames)):
            add_warning(warnings, "WARN", "video_frame_count_mismatch", f"Camera {cam} video has {total} frames but parquet episode has {n_frames} rows.", camera=cam, video_frames=total, parquet_rows=n_frames)


def generate_bc_visuals(root: Path, df_ep: pd.DataFrame, info: dict[str, Any], ep: int, out_dir: Path, args: Args, warnings: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    action = stack_vector_column(df_ep, ACTION)
    state = stack_vector_column(df_ep, OBS_STATE, dim=action.shape[1])
    t = get_time_vector(df_ep, len(df_ep))
    duration, fps_est = estimate_duration_fps(t, len(df_ep), info)
    joint_names = get_joint_names(info, action.shape[1])
    stats = compute_action_state_stats(action, state, joint_names)

    check_common_quality(df_ep, t, action, state, info, args, warnings)
    check_video_quality(root, len(df_ep), warnings)

    plot_all_per_joint(t, action, state, joint_names, stats, out_dir, args.dpi)
    plot_all_joints_grid(t, action, state, joint_names, out_dir / "all_joints_action_vs_state_grid.png", args.dpi)
    plot_selected_action_vs_state(t, action, state, joint_names, out_dir / "selected_action_vs_state.png", args.dpi)
    plot_gripper_action_vs_state(t, action, state, joint_names, out_dir / "gripper_action_vs_state.png", args.dpi)
    plot_group_curves(t, action, state, joint_names, out_dir, args.dpi)
    plot_bar(stats, "rmse_action_minus_state", "RMSE by joint: action - state", "RMSE", out_dir / "rmse_by_joint.png", args.dpi)
    plot_bar(stats, "max_abs_error", "Max absolute error by joint", "max |action - state|", out_dir / "max_abs_error_by_joint.png", args.dpi)
    plot_correlation_sign_check(stats, out_dir / "correlation_sign_check.png", args.dpi)
    plot_range_comparison(stats, out_dir / "action_state_range_comparison.png", args.dpi)
    stats.to_csv(out_dir / "per_joint_stats.csv", index=False)
    if args.save_csv:
        save_bc_csv_files(out_dir, t, action, state, joint_names, stats)
    if not args.no_video_sheet:
        plot_video_contact_sheet(root, out_dir / "video_contact_sheet.png", select_even_frames(len(df_ep), args.contact_sheet_frames), "Video contact sheet - uniform frames", args.dpi)

    lines = [
        f"Dataset root: {root}",
        f"Episode index: {ep}",
        "Mode: bc",
        f"Frames: {len(df_ep)}",
        f"Duration: {duration:.6f} s",
        f"Estimated fps: {fps_est:.6f}",
        "",
        "Columns in parquet:",
    ]
    lines.extend(f" - {col}" for col in df_ep.columns)
    lines.extend(["", "Joint order:"])
    lines.extend(f" {i:02d}: {name}" for i, name in enumerate(joint_names))
    lines.extend(["", "Warnings:"])
    if warnings:
        lines.extend(f" [{w['severity']}] {w['code']}: {w['message']}" for w in warnings)
    else:
        lines.append(" No warnings.")
    lines.extend(["", "Per-joint statistics:", stats.to_string(index=False)])
    write_text_report(out_dir / "overview.txt", lines)
    return stats, {"duration_s": duration, "fps_est": fps_est, "frames": len(df_ep)}


def generate_rl_visuals(root: Path, df_ep: pd.DataFrame, info: dict[str, Any], ep: int, out_dir: Path, args: Args, warnings: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = [col for col in RL_COLUMNS if col not in df_ep.columns]
    completeness_rows: list[dict[str, Any]] = []
    for col in RL_COLUMNS + TIMING_COLUMNS:
        present = col in df_ep.columns
        completeness_rows.append({"column": col, "present": present, "missing_values": int(df_ep[col].isna().sum()) if present else len(df_ep)})
    field_completeness = pd.DataFrame(completeness_rows)
    field_completeness.to_csv(out_dir / "field_completeness.csv", index=False)
    if missing:
        add_warning(warnings, "ERROR", "missing_rl_columns", f"Missing RL columns: {missing}", columns=",".join(missing))
        raise ValueError(f"Missing RL columns: {missing}")
    class_feature = (info.get("features") or {}).get("action.gripper_policy_class", {})
    if class_feature.get("dtype") != "int64" or list(class_feature.get("shape", [])) != [2]:
        add_warning(
            warnings,
            "ERROR",
            "gripper_class_schema",
            "action.gripper_policy_class metadata must declare dtype=int64 and shape=[2].",
        )

    state = stack_vector_column(df_ep, OBS_STATE)
    executed = stack_vector_column(df_ep, "action.executed", dim=state.shape[1])
    action = stack_vector_column(df_ep, ACTION, dim=state.shape[1]) if ACTION in df_ep.columns else executed.copy()
    act = stack_vector_column(df_ep, "action.act", dim=state.shape[1])
    delta = stack_vector_column(df_ep, "action.rl_delta", dim=state.shape[1])
    human = stack_vector_column(df_ep, "action.human", dim=state.shape[1])
    gripper_policy_class = stack_vector_column(df_ep, "action.gripper_policy_class", dim=2)
    t = get_time_vector(df_ep, len(df_ep))
    duration, fps_est = estimate_duration_fps(t, len(df_ep), info)
    joint_names = get_joint_names(info, state.shape[1])
    stats = compute_action_state_stats(executed, state, joint_names)
    stats.to_csv(out_dir / "per_joint_stats.csv", index=False)

    check_common_quality(df_ep, t, executed, state, info, args, warnings)
    check_video_quality(root, len(df_ep), warnings)

    vector_dimensions = {
        OBS_STATE: state.shape[1],
        ACTION: action.shape[1],
        "action.act": act.shape[1],
        "action.rl_delta": delta.shape[1],
        "action.human": human.shape[1],
        "action.executed": executed.shape[1],
    }
    invalid_dimensions = {key: value for key, value in vector_dimensions.items() if value != 16}
    if invalid_dimensions:
        add_warning(
            warnings,
            "ERROR",
            "hybrid_action_dimension",
            f"All state/action vectors must remain 16-D, got {invalid_dimensions}.",
        )

    finite_classes = gripper_policy_class[np.isfinite(gripper_policy_class)]
    if not np.all(np.isin(finite_classes.astype(np.int64), [0, 1, 2])) or not np.allclose(
        finite_classes, finite_classes.astype(np.int64), rtol=0.0, atol=0.0
    ):
        add_warning(
            warnings,
            "ERROR",
            "invalid_gripper_policy_class",
            "action.gripper_policy_class must contain only {0,1,2}.",
        )
    if not np.isfinite(gripper_policy_class).all():
        add_warning(
            warnings,
            "ERROR",
            "nonfinite_gripper_policy_class",
            "action.gripper_policy_class contains NaN or Inf.",
        )

    gripper_indices = [joint_names.index(name) for name in GRIPPERS]
    max_gripper_delta = float(np.nanmax(np.abs(delta[:, gripper_indices])))
    if max_gripper_delta > 1e-7:
        add_warning(
            warnings,
            "ERROR",
            "nonzero_gripper_rl_delta",
            f"action.rl_delta gripper entries must be zero; max abs value={max_gripper_delta:.9g}.",
            max_abs=max_gripper_delta,
        )
    endpoint_valid = lambda values: np.isclose(values, 0.0, rtol=0.0, atol=1e-6) | np.isclose(  # noqa: E731
        values, 0.8, rtol=0.0, atol=1e-6
    )
    legacy_upgrade = (root / "gripper_schema_upgrade.json").is_file()
    for key, values in ((ACTION, action), ("action.executed", executed)):
        invalid_endpoint_count = int(np.size(values[:, gripper_indices]) - endpoint_valid(values[:, gripper_indices]).sum())
        if invalid_endpoint_count:
            add_warning(
                warnings,
                "INFO" if legacy_upgrade else "ERROR",
                "legacy_continuous_gripper" if legacy_upgrade else "nonbinary_executed_gripper",
                f"{key} has {invalid_endpoint_count} gripper value(s) outside exact endpoints 0.0/0.8."
                + (" This dataset was schema-upgraded without rewriting legacy actions." if legacy_upgrade else ""),
                count=invalid_endpoint_count,
            )

    gripper_frame_counts = np.zeros((2, 3), dtype=np.int64)
    gripper_event_counts = np.zeros((2, 3), dtype=np.int64)
    classes_int = np.where(np.isfinite(gripper_policy_class), gripper_policy_class, 0).astype(np.int64)
    previous = np.zeros(2, dtype=np.int64)
    for frame_index, classes in enumerate(classes_int):
        for side in range(2):
            value = int(classes[side])
            if value not in {0, 1, 2}:
                continue
            gripper_frame_counts[side, value] += 1
            if value != 0 and (frame_index == 0 or value != int(previous[side])):
                gripper_event_counts[side, value] += 1
        previous = classes

    control = scalar_column(df_ep, "control_source", default=-1)
    intervention = scalar_column(df_ep, "is_intervention", default=0)
    has_human = scalar_column(df_ep, "has_human_action", default=0)
    reward = scalar_column(df_ep, "reward", default=0)
    done = scalar_column(df_ep, "done", default=0)
    success = scalar_column(df_ep, "success", default=0)

    if not np.all(np.isin(control[~np.isnan(control)].astype(int), [-1, 0, 1])):
        add_warning(warnings, "WARN", "unexpected_control_source", "control_source contains values outside {-1,0,1}.")
    expected_intervention = (control == 0).astype(float)
    mismatch = np.isfinite(control) & (control >= 0) & (np.abs(intervention - expected_intervention) > 0.5)
    if np.any(mismatch):
        add_warning(warnings, "ERROR", "intervention_control_mismatch", f"is_intervention differs from control_source==0 at {int(mismatch.sum())} frame(s).", count=int(mismatch.sum()))
    if np.sum(intervention >= 0.5) == 0:
        add_warning(warnings, "WARN", "no_intervention_frames", "This RL episode contains no intervention frames. It may be a pure rollout/eval episode, but it cannot directly provide human residual targets.")
    reward_events, reward_stats = check_keyboard_reward_semantics(reward, done, success, warnings)
    reward_events.to_csv(out_dir / "reward_events.csv", index=False)

    residual_lambda, lambda_source = find_residual_lambda(root, info, args.residual_lambda)
    if "default" in lambda_source:
        add_warning(warnings, "WARN", "residual_lambda_not_recorded", f"Residual lambda was not found in dataset metadata; using {residual_lambda}. Pass --residual-lambda to override.")

    # Units: in the bw_residual package, action.rl_delta is recorded from Policy/debug/action_rl_delta,
    # which is the joint-space delta after applying residual_limits, not a normalized [-1, 1] vector.
    arm_indices = [index for index in range(state.shape[1]) if index not in gripper_indices]
    predicted_non_intervention = act + residual_lambda * delta
    residual_target = human[:, arm_indices] - act[:, arm_indices]
    non_intervention_mask = intervention < 0.5
    intervention_mask = intervention >= 0.5
    formula_err = np.abs(executed - predicted_non_intervention)
    human_err = np.abs(executed - human)
    formula_err_norm = np.nanmax(formula_err[:, arm_indices], axis=1)
    human_err_norm = np.nanmax(human_err, axis=1)

    # `action` should be the final executed action in both BC and RL modes.
    action_executed_err = np.nanmax(np.abs(action - executed), axis=1)
    if np.nanmax(action_executed_err) > args.reconstruction_threshold:
        add_warning(warnings, "ERROR", "action_not_executed", f"Column action differs from action.executed. max error={np.nanmax(action_executed_err):.6g}.", max_error=float(np.nanmax(action_executed_err)))

    if np.any(intervention_mask):
        max_human_err = float(np.nanmax(human_err_norm[intervention_mask]))
        if max_human_err > args.reconstruction_threshold:
            add_warning(warnings, "ERROR", "intervention_executed_not_human", f"During intervention, action.executed should equal action.human. max error={max_human_err:.6g}.", max_error=max_human_err)
        if np.nanmax(np.abs(has_human[intervention_mask] - 1.0)) > 0.5:
            add_warning(warnings, "WARN", "has_human_action_mismatch", "Some intervention frames do not have has_human_action=1.")
    if np.any(non_intervention_mask):
        max_formula_err = float(np.nanmax(formula_err_norm[non_intervention_mask]))
        mean_formula_err = float(np.nanmean(formula_err_norm[non_intervention_mask]))
        if max_formula_err > args.reconstruction_threshold:
            severity = "WARN" if args.strict_reconstruction else "INFO"
            add_warning(
                warnings,
                severity,
                "non_intervention_formula_mismatch",
                "For non-intervention frames, arm entries in action.executed differ from action.act + lambda*action.rl_delta. This can be expected when policy-runner arm smoothing/clamp is enabled.",
                max_error=max_formula_err,
                mean_error=mean_formula_err,
                residual_lambda=residual_lambda,
            )
        if np.nanmax(np.abs(human[non_intervention_mask])) > args.reconstruction_threshold:
            add_warning(warnings, "WARN", "human_action_nonzero_without_intervention", "action.human should be zero on non-intervention frames, but nonzero values were found.")

    # Timing checks.
    for col in TIMING_COLUMNS:
        vals = scalar_column(df_ep, col, default=np.nan)
        if np.isfinite(vals).any() and np.nanmax(np.abs(vals)) > args.max_timing_dt:
            add_warning(warnings, "WARN", "timing_dt_large", f"{col} has abs(dt) larger than {args.max_timing_dt}s.", column=col, max_abs_dt=float(np.nanmax(np.abs(vals))))

    limits = residual_limits(args.residual_limit_default, args.residual_limit_gripper, state.shape[1], joint_names)
    delta_saturation = np.abs(delta[:, arm_indices]) / np.maximum(limits[arm_indices].reshape(1, -1), 1e-9)
    saturated_ratio = float(np.mean(delta_saturation > 0.95))
    if saturated_ratio > 0.1:
        add_warning(warnings, "WARN", "residual_near_limit", f"{saturated_ratio*100:.1f}% of residual delta entries are above 95% of configured residual limit.", ratio=saturated_ratio)

    transition_rows: list[dict[str, Any]] = []
    for i in range(len(df_ep)):
        transition_rows.append({
            "frame": i,
            "control_source": control[i],
            "is_intervention": intervention[i],
            "reward": reward[i],
            "done": done[i],
            "success": success[i],
            "formula_error_inf_norm": formula_err_norm[i],
            "human_error_inf_norm": human_err_norm[i],
            "action_vs_executed_error_inf_norm": action_executed_err[i],
            "residual_delta_l2": float(np.linalg.norm(delta[i])),
            "residual_target_l2": float(np.linalg.norm(residual_target[i])),
            "usable_transition": bool(i < len(df_ep) - 1 and done[i] < 0.5),
        })
    transition_validity = pd.DataFrame(transition_rows)
    transition_validity.to_csv(out_dir / "transition_validity.csv", index=False)

    # Plots shared with BC but using action.executed as the actual action.
    plot_all_per_joint(t, executed, state, joint_names, stats, out_dir, args.dpi)
    plot_all_joints_grid(t, executed, state, joint_names, out_dir / "all_joints_action_vs_state_grid.png", args.dpi)
    plot_selected_action_vs_state(t, executed, state, joint_names, out_dir / "selected_action_vs_state.png", args.dpi)
    plot_gripper_action_vs_state(t, executed, state, joint_names, out_dir / "gripper_action_vs_state.png", args.dpi)
    plot_bar(stats, "rmse_action_minus_state", "RMSE by joint: action.executed - state", "RMSE", out_dir / "rmse_by_joint.png", args.dpi)
    plot_bar(stats, "max_abs_error", "Max absolute error by joint", "max |executed - state|", out_dir / "max_abs_error_by_joint.png", args.dpi)
    plot_correlation_sign_check(stats, out_dir / "correlation_sign_check.png", args.dpi)
    plot_range_comparison(stats, out_dir / "action_state_range_comparison.png", args.dpi)

    plot_rl_timelines(t, control, intervention, reward, done, success, out_dir, args.dpi)
    plot_action_source_overview(t, act, human, executed, joint_names, out_dir, args.dpi)
    plot_norm_curve(t, np.linalg.norm(delta, axis=1), "Residual delta norm", "L2 norm", out_dir / "residual_delta_norm.png", args.dpi)
    plot_norm_curve(t, np.linalg.norm(residual_target, axis=1), "Residual target norm: action.human - action.act", "L2 norm", out_dir / "residual_target_norm.png", args.dpi)
    plot_reconstruction_error(t, formula_err_norm, args.reconstruction_threshold, out_dir / "executed_reconstruction_error.png", args.dpi)
    plot_vector_grid(t, act, executed, joint_names, "action.act", "action.executed", "ACT action vs executed action", out_dir / "act_vs_executed_grid.png", args.dpi)
    plot_vector_grid(t, act, human, joint_names, "action.act", "action.human", "ACT action vs human action", out_dir / "act_vs_human_grid.png", args.dpi)
    plot_residual_target_grid(t, residual_target, delta, joint_names, out_dir / "human_correction_minus_act_grid.png", args.dpi)

    if args.save_csv:
        save_bc_csv_files(out_dir, t, executed, state, joint_names, stats)
        save_rl_csv_files(out_dir, t, joint_names,
            arrays={"act": act, "rl_delta": delta, "human": human, "executed": executed, "state": state, "residual_target": residual_target},
            scalars={"control_source": control, "is_intervention": intervention, "has_human_action": has_human, "reward": reward, "done": done, "success": success},
        )

    if not args.no_video_sheet:
        plot_video_contact_sheet(root, out_dir / "camera_contact_sheet_uniform.png", select_even_frames(len(df_ep), args.contact_sheet_frames), "Camera contact sheet - uniform frames", args.dpi)
        intervention_frames = select_event_frames(intervention >= 0.5, args.contact_sheet_frames)
        if intervention_frames:
            plot_video_contact_sheet(root, out_dir / "camera_contact_sheet_intervention_events.png", intervention_frames, "Camera contact sheet - intervention events", args.dpi)
        reward_done_mask = (np.abs(reward) > 1e-12) | (done >= 0.5) | (success >= 0.5)
        reward_done_frames = select_event_frames(reward_done_mask, args.contact_sheet_frames)
        if reward_done_frames:
            plot_video_contact_sheet(root, out_dir / "camera_contact_sheet_reward_done_events.png", reward_done_frames, "Camera contact sheet - reward/done/success events", args.dpi)

    rl_stats = {
        "frames": len(df_ep),
        "duration_s": duration,
        "fps_est": fps_est,
        "residual_lambda": residual_lambda,
        "residual_lambda_source": lambda_source,
        "intervention_frames": int(np.sum(intervention >= 0.5)),
        "intervention_ratio": float(np.mean(intervention >= 0.5)),
        **reward_stats,
        "done_count": int(np.sum(done >= 0.5)),
        "success_any": bool(np.nanmax(success) >= 0.5),
        "usable_transitions": int(np.sum((np.arange(len(df_ep)) < len(df_ep)-1) & (done < 0.5))),
        "max_formula_error_non_intervention": float(np.nanmax(formula_err_norm[non_intervention_mask])) if np.any(non_intervention_mask) else float("nan"),
        "max_human_error_intervention": float(np.nanmax(human_err_norm[intervention_mask])) if np.any(intervention_mask) else float("nan"),
        "residual_delta_mean_l2": float(np.nanmean(np.linalg.norm(delta[:, arm_indices], axis=1))),
        "residual_target_mean_l2": float(np.nanmean(np.linalg.norm(residual_target, axis=1))),
        "residual_saturated_ratio": saturated_ratio,
        "left_keep_base_frames": int(gripper_frame_counts[0, 0]),
        "left_force_open_frames": int(gripper_frame_counts[0, 1]),
        "left_force_close_frames": int(gripper_frame_counts[0, 2]),
        "right_keep_base_frames": int(gripper_frame_counts[1, 0]),
        "right_force_open_frames": int(gripper_frame_counts[1, 1]),
        "right_force_close_frames": int(gripper_frame_counts[1, 2]),
        "left_force_open_events": int(gripper_event_counts[0, 1]),
        "left_force_close_events": int(gripper_event_counts[0, 2]),
        "right_force_open_events": int(gripper_event_counts[1, 1]),
        "right_force_close_events": int(gripper_event_counts[1, 2]),
    }
    pd.DataFrame([rl_stats]).to_csv(out_dir / "rl_episode_stats.csv", index=False)

    lines = [
        f"Dataset root: {root}",
        f"Episode index: {ep}",
        "Mode: rl",
        f"Frames: {len(df_ep)}",
        f"Duration: {duration:.6f} s",
        f"Estimated fps: {fps_est:.6f}",
        f"Residual lambda: {residual_lambda} ({lambda_source})",
        f"Intervention frames: {rl_stats['intervention_frames']} ({rl_stats['intervention_ratio']*100:.2f}%)",
        f"Reward sum in parquet: {rl_stats['reward_sum_parquet']:.6g}",
        f"Usable transitions: {rl_stats['usable_transitions']}",
        "Gripper policy events (left open/close, right open/close): "
        f"{gripper_event_counts[0, 1]}/{gripper_event_counts[0, 2]}, "
        f"{gripper_event_counts[1, 1]}/{gripper_event_counts[1, 2]}",
        "",
        "Important interpretation notes:",
        " - action.rl_delta is 16-D for storage, but gripper indices 7/15 must be zero; only 14 arm entries carry residuals.",
        " - action.executed grippers use categorical control and exact 0.0/0.8 endpoints, so arm reconstruction excludes grippers.",
        " - If arm smoothing/clamp is enabled, executed arms may differ from action.act + lambda*action.rl_delta.",
        " - During intervention, action.executed should equal action.human.",
        f"Reward events: {rl_stats['reward_event_count']} total event frame(s), {rl_stats['reward_nonzero_frames']} nonzero reward frame(s)",
        f"Last frame: reward={rl_stats['last_reward']:.6g}, done={rl_stats['last_done']:.6g}, success={rl_stats['last_success']:.6g}, event={rl_stats['terminal_event_type']}",
        "",
        "Keyboard reward rules:",
        " - a: left block placed -> reward += 1, done=0, success=0",
        " - d: right block placed -> reward += 2, done=1, success=1, stop episode",
        " - g: manual success -> reward += 1, done=1, success=1, stop episode",
        " - j: manual failure -> reward += 0, done=1, success=0, stop episode",
        "",
        "Reward event frames are saved to reward_events.csv.",
        "",
        "Warnings:",
    ]
    if warnings:
        lines.extend(f" [{w['severity']}] {w['code']}: {w['message']}" for w in warnings)
    else:
        lines.append(" No warnings.")
    lines.extend(["", "Generated RL files:"])
    lines.extend([
        " - field_completeness.csv",
        " - transition_validity.csv",
        " - reward_events.csv",
        " - rl_episode_stats.csv",
        " - control_source_timeline.png",
        " - intervention_timeline.png",
        " - reward_done_success_timeline.png",
        " - action_source_overview.png",
        " - residual_delta_norm.png",
        " - residual_target_norm.png",
        " - executed_reconstruction_error.png",
        " - act_vs_executed_grid.png",
        " - act_vs_human_grid.png",
        " - human_correction_minus_act_grid.png",
        " - camera_contact_sheet_uniform.png",
        " - camera_contact_sheet_intervention_events.png",
        " - camera_contact_sheet_reward_done_events.png",
    ])
    write_text_report(out_dir / "rl_summary.txt", lines)
    write_text_report(out_dir / "overview.txt", lines)
    return stats, rl_stats


def image_tags_for_dir(base_dir: Path, rel_dir: Path) -> str:
    files = sorted(rel_dir.glob("*.png"))
    parts: list[str] = []
    for path in files:
        try:
            rel = path.relative_to(base_dir).as_posix()
        except Exception:
            rel = path.as_posix()
        parts.append(f'<figure><img src="{html.escape(rel)}"><figcaption>{html.escape(path.name)}</figcaption></figure>')
    return "\n".join(parts)


def write_html_report(out_dir: Path, results: list[EpisodeResult]) -> None:
    css = """
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; line-height: 1.45; }
    h1, h2, h3 { margin-top: 1.4em; }
    table { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 14px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { background: #f3f3f3; }
    .warn { color: #9a6700; font-weight: 600; }
    .error { color: #b42318; font-weight: 700; }
    .info { color: #175cd3; }
    figure { margin: 16px 0; border: 1px solid #eee; padding: 8px; }
    img { max-width: 100%; height: auto; display: block; }
    figcaption { font-size: 13px; color: #555; margin-top: 4px; }
    code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }
    """
    rows = []
    for r in results:
        rows.append(
            "<tr>"
            f"<td>{r.episode_index}</td><td>{html.escape(r.mode)}</td><td>{r.frames}</td>"
            f"<td>{r.duration_s:.3f}</td><td>{r.fps_est:.3f}</td>"
            f"<td>{sum(1 for w in r.warnings if w['severity']=='ERROR')}</td>"
            f"<td>{sum(1 for w in r.warnings if w['severity']=='WARN')}</td>"
            f"<td><a href=\"{html.escape(r.out_dir.relative_to(out_dir).as_posix())}/overview.txt\">overview.txt</a></td>"
            "</tr>"
        )
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>BW dataset check report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>BW LeRobot dataset check report</h1>",
        "<h2>Episode summary</h2>",
        "<table><tr><th>Episode</th><th>Mode</th><th>Frames</th><th>Duration s</th><th>FPS</th><th>Errors</th><th>Warnings</th><th>Text report</th></tr>",
        "\n".join(rows),
        "</table>",
    ]
    for r in results:
        html_parts.append(f"<h2>Episode {r.episode_index:03d} ({html.escape(r.mode)})</h2>")
        if r.warnings:
            html_parts.append("<h3>Warnings</h3><table><tr><th>Severity</th><th>Code</th><th>Message</th></tr>")
            for w in r.warnings:
                cls = str(w["severity"]).lower()
                html_parts.append(f"<tr><td class='{cls}'>{html.escape(str(w['severity']))}</td><td>{html.escape(str(w['code']))}</td><td>{html.escape(str(w['message']))}</td></tr>")
            html_parts.append("</table>")
        key_images = [
            "all_joints_action_vs_state_grid.png",
            "selected_action_vs_state.png",
            "gripper_action_vs_state.png",
            "rmse_by_joint.png",
            "max_abs_error_by_joint.png",
            "correlation_sign_check.png",
            "action_state_range_comparison.png",
            "video_contact_sheet.png",
            "control_source_timeline.png",
            "intervention_timeline.png",
            "reward_done_success_timeline.png",
            "action_source_overview.png",
            "residual_delta_norm.png",
            "residual_target_norm.png",
            "executed_reconstruction_error.png",
            "act_vs_executed_grid.png",
            "human_correction_minus_act_grid.png",
            "camera_contact_sheet_uniform.png",
            "camera_contact_sheet_intervention_events.png",
            "camera_contact_sheet_reward_done_events.png",
        ]
        for name in key_images:
            path = r.out_dir / name
            if path.exists():
                rel = path.relative_to(out_dir).as_posix()
                html_parts.append(f'<figure><img src="{html.escape(rel)}"><figcaption>{html.escape(name)}</figcaption></figure>')
    html_parts.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")


def run_episode(root: Path, all_df: pd.DataFrame, info: dict[str, Any], ep: int, out_base: Path, args: Args) -> EpisodeResult:
    df_ep = episode_dataframe(all_df, ep)
    mode = detect_mode(df_ep, args.mode)
    out_dir = out_base / f"episode_{ep:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[dict[str, Any]] = []
    if mode == "bc":
        _, stat = generate_bc_visuals(root, df_ep, info, ep, out_dir, args, warnings)
    else:
        _, stat = generate_rl_visuals(root, df_ep, info, ep, out_dir, args, warnings)
    pd.DataFrame(warnings).to_csv(out_dir / "warnings.csv", index=False)
    return EpisodeResult(
        episode_index=ep,
        mode=mode,
        out_dir=out_dir,
        frames=int(stat.get("frames", len(df_ep))),
        duration_s=float(stat.get("duration_s", 0.0)),
        fps_est=float(stat.get("fps_est", float("nan"))),
        warnings=warnings,
        stats=stat,
    )


def main() -> int:
    args = parse_args()
    root = ensure_dataset_root(args.dataset_root)
    info = load_info(root)
    all_df = read_all_data(root)
    out_base = (args.out_dir.expanduser().resolve() if args.out_dir else root / "check_report")
    out_base.mkdir(parents=True, exist_ok=True)

    if args.episode is not None:
        episodes = [int(args.episode)]
    else:
        episodes = list_episode_indices(all_df)

    results: list[EpisodeResult] = []
    print(f"Dataset root: {root}")
    print(f"Output dir:   {out_base}")
    print(f"Episodes:     {episodes}")
    for ep in episodes:
        print(f"\nChecking episode {ep}...")
        try:
            result = run_episode(root, all_df, info, ep, out_base, args)
            results.append(result)
            print(f"  mode={result.mode} frames={result.frames} warnings={len(result.warnings)} -> {result.out_dir}")
        except Exception as exc:
            err_dir = out_base / f"episode_{ep:03d}"
            err_dir.mkdir(parents=True, exist_ok=True)
            warning = {"severity": "ERROR", "code": "episode_failed", "message": str(exc)}
            pd.DataFrame([warning]).to_csv(err_dir / "warnings.csv", index=False)
            write_text_report(err_dir / "overview.txt", [f"Episode {ep} failed: {exc}"])
            results.append(EpisodeResult(ep, args.mode, err_dir, 0, 0.0, float("nan"), [warning], {}))
            print(f"  ERROR: {exc}", file=sys.stderr)

    summary_rows: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []
    for r in results:
        row = {
            "episode_index": r.episode_index,
            "mode": r.mode,
            "frames": r.frames,
            "duration_s": r.duration_s,
            "fps_est": r.fps_est,
            "error_count": sum(1 for w in r.warnings if w["severity"] == "ERROR"),
            "warn_count": sum(1 for w in r.warnings if w["severity"] == "WARN"),
            "info_count": sum(1 for w in r.warnings if w["severity"] == "INFO"),
        }
        row.update({k: v for k, v in r.stats.items() if isinstance(v, (int, float, str, bool))})
        summary_rows.append(row)
        for w in r.warnings:
            all_warnings.append({"episode_index": r.episode_index, **w})
    pd.DataFrame(summary_rows).to_csv(out_base / "summary.csv", index=False)
    pd.DataFrame(all_warnings).to_csv(out_base / "warnings.csv", index=False)
    write_html_report(out_base, results)

    print("\nDone.")
    print(f"Summary: {out_base / 'summary.csv'}")
    print(f"Warnings: {out_base / 'warnings.csv'}")
    print(f"HTML report: {out_base / 'report.html'}")
    return 0 if not any(w["severity"] == "ERROR" for w in all_warnings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
