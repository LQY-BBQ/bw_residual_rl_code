"""Check RL reward/done/success columns stored directly in LeRobot parquet files.

This replaces the old annotation-based reward check. It does not read
annotations/episode_xxxxxx.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect reward/done/success columns in an RL LeRobot dataset.")
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--strict", action="store_true", help="Return non-zero when warnings are present.")
    p.add_argument("--show-events", action="store_true", help="Print every non-zero reward frame.")
    return p.parse_args()


def _as_scalar(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def read_lerobot_parquets(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def scalar_column(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full((len(df),), default, dtype=np.float32)
    return np.asarray([_as_scalar(v, default) for v in df[name]], dtype=np.float32)


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    df = read_lerobot_parquets(root)

    failures: list[str] = []
    warnings: list[str] = []
    if "reward" not in df.columns:
        failures.append("missing reward column; updated training will not use annotation JSON fallback")
    if "done" not in df.columns:
        warnings.append("missing done column; training will infer done from episode boundary")
    if "success" not in df.columns:
        warnings.append("missing success column")

    reward = scalar_column(df, "reward")
    done = scalar_column(df, "done")
    success = scalar_column(df, "success")

    if not np.all(np.isfinite(reward)):
        failures.append("reward column contains non-finite values")
    if not np.all(np.isfinite(done)):
        failures.append("done column contains non-finite values")
    if not np.all(np.isfinite(success)):
        failures.append("success column contains non-finite values")

    episode_index = df["episode_index"].to_numpy() if "episode_index" in df.columns else np.zeros((len(df),), dtype=np.int64)
    frame_index = df["frame_index"].to_numpy() if "frame_index" in df.columns else np.arange(len(df), dtype=np.int64)
    episodes = sorted(int(v) for v in np.unique(episode_index))

    print("RL Reward Check")
    print("---------------")
    print(f"dataset: {root}")
    print(f"frames: {len(df)}")
    print(f"episodes: {len(episodes)}")
    print(f"total_reward: {float(reward.sum()):.3f}")
    print(f"nonzero_reward_frames: {int(np.count_nonzero(np.abs(reward) > 1e-8))}")
    print(f"done_frames: {int(np.count_nonzero(done >= 0.5))}")
    print(f"success_frames: {int(np.count_nonzero(success >= 0.5))}")

    if len(df) > 0 and np.count_nonzero(np.abs(reward) > 1e-8) == 0:
        warnings.append("all rewards are zero; maybe no a/d/g key was recorded")

    print("\nPer Episode")
    print("-----------")
    for ep in episodes:
        idx = np.flatnonzero(episode_index == ep)
        if idx.size == 0:
            continue
        ep_reward = reward[idx]
        ep_done = done[idx]
        ep_success = success[idx]
        last_i = int(idx[-1])
        event_local = np.flatnonzero(np.abs(ep_reward) > 1e-8)
        event_text = ", ".join(
            f"frame={int(frame_index[idx[j]])}:reward={float(ep_reward[j]):.1f}"
            for j in event_local[:10]
        )
        if event_local.size > 10:
            event_text += ", ..."
        if not event_text:
            event_text = "<none>"
        print(
            f"episode={ep} frames={idx.size} total_reward={float(ep_reward.sum()):.3f} "
            f"events={event_local.size} last_done={int(ep_done[-1] >= 0.5)} "
            f"last_success={int(ep_success[-1] >= 0.5)}"
        )
        if args.show_events:
            print(f"  reward_events: {event_text}")
        if ep_done[-1] < 0.5:
            warnings.append(f"episode {ep}: last frame done is not 1")
        if np.count_nonzero(ep_done >= 0.5) != 1:
            warnings.append(f"episode {ep}: expected exactly one done frame, got {int(np.count_nonzero(ep_done >= 0.5))}")
        # Success should normally be attached to a done frame.
        if np.any(ep_success >= 0.5) and success[last_i] < 0.5:
            warnings.append(f"episode {ep}: success appears before final frame or final success is missing")

    if warnings:
        print("\nWarnings")
        print("--------")
        for item in warnings:
            print(f"- {item}")

    if failures:
        print("\nFailures")
        print("--------")
        for item in failures:
            print(f"- {item}")
        print("\nVerdict\n-------\nFAIL")
        return 1

    verdict = "WARN" if warnings else "PASS"
    print(f"\nVerdict\n-------\n{verdict}")
    return 1 if warnings and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
