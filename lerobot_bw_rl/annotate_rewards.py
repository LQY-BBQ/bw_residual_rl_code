"""Create or inspect reward annotation JSON files for RL datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--success", choices=["true", "false"], default="false")
    p.add_argument("--left-block-done-frame", type=int, default=None)
    p.add_argument("--right-block-done-frame", type=int, default=None)
    p.add_argument("--failure-type", default="")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser()
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    path = ann_dir / f"episode_{args.episode_index:06d}.json"
    data = {
        "episode_index": args.episode_index,
        "success": args.success == "true",
        "left_block_done_frame": args.left_block_done_frame,
        "right_block_done_frame": args.right_block_done_frame,
        "failure_type": args.failure_type,
        "notes": args.notes,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
