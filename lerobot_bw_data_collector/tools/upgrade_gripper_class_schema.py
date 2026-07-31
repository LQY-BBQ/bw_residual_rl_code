#!/usr/bin/env python3
"""Non-destructively add the hybrid gripper class field to legacy RL data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FIELD = "action.gripper_policy_class"
DELTA_FIELD = "action.rl_delta"
GRIPPER_INDICES = (7, 15)
FEATURE = {"dtype": "int64", "shape": [2], "names": ["left", "right"]}
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a legacy BW residual dataset and add action.gripper_policy_class=[0,0]. "
            "The source is accepted only when both gripper entries in action.rl_delta are zero."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def _parquet_files(root: Path) -> list[Path]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No data parquet files found under {root / 'data'}")
    return files


def _validate_source(source: Path, output: Path) -> tuple[dict[str, Any], list[Path], int]:
    if source == output:
        raise ValueError("Source and output must be different; in-place upgrades are forbidden")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Source is not a LeRobot dataset: missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features")
    if not isinstance(features, dict) or DELTA_FIELD not in features:
        raise ValueError(f"Source is not residual RL data: missing feature {DELTA_FIELD}")
    if FIELD in features:
        raise ValueError(f"Source already contains {FIELD}; no upgrade is needed")

    files = _parquet_files(source)
    total_rows = 0
    for path in files:
        table = pq.read_table(path, columns=[DELTA_FIELD])
        deltas = np.asarray(table[DELTA_FIELD].to_pylist(), dtype=np.float32)
        if deltas.ndim != 2 or deltas.shape[1] != 16:
            raise ValueError(f"{path}: {DELTA_FIELD} must be 16-D, got shape={deltas.shape}")
        gripper_delta = deltas[:, GRIPPER_INDICES]
        invalid = np.argwhere(~np.equal(gripper_delta, 0.0))
        if invalid.size:
            row, side = (int(value) for value in invalid[0])
            raise ValueError(
                f"Refusing automatic upgrade: {path} row={row} gripper_index="
                f"{GRIPPER_INDICES[side]} has {DELTA_FIELD}={gripper_delta[row, side]:.9g}. "
                "Legacy continuous gripper residuals cannot be inferred as categorical labels."
            )
        total_rows += table.num_rows
    expected_rows = int(info.get("total_frames", total_rows))
    if expected_rows != total_rows:
        raise ValueError(f"meta total_frames={expected_rows}, but parquet files contain {total_rows} rows")
    return info, files, total_rows


def _zero_stats(count: int) -> dict[str, list[int | float]]:
    return {
        "min": [0, 0],
        "max": [0, 0],
        "mean": [0.0, 0.0],
        "std": [0.0, 0.0],
        "count": [int(count)],
        "q01": [0.0, 0.0],
        "q10": [0.0, 0.0],
        "q50": [0.0, 0.0],
        "q90": [0.0, 0.0],
        "q99": [0.0, 0.0],
    }


def _append_keep_base(path: Path) -> None:
    table = pq.read_table(path)
    values = pa.array([[0, 0]] * table.num_rows, type=pa.list_(pa.int64()))
    pq.write_table(table.append_column(FIELD, values), path)


def _patch_episode_stats(root: Path) -> None:
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        if FIELD in table.column_names:
            continue
        lengths = table["length"].to_pylist()
        for stat_name in STAT_NAMES:
            column_name = f"stats/{FIELD}/{stat_name}"
            if stat_name == "count":
                values = [[int(length)] for length in lengths]
                arrow_type = pa.list_(pa.int64())
            elif stat_name in {"min", "max"}:
                values = [[0, 0] for _ in lengths]
                arrow_type = pa.list_(pa.int64())
            else:
                values = [[0.0, 0.0] for _ in lengths]
                arrow_type = pa.list_(pa.float64())
            table = table.append_column(column_name, pa.array(values, type=arrow_type))
        pq.write_table(table, path)


def upgrade(source: Path, output: Path) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source in output.parents:
        raise ValueError("Output must not be inside the source dataset")
    info, source_files, total_rows = _validate_source(source, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.upgrade-", dir=output.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        for source_path in source_files:
            _append_keep_base(staging / source_path.relative_to(source))

        info["features"][FIELD] = FEATURE
        (staging / "meta" / "info.json").write_text(
            json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stats_path = staging / "meta" / "stats.json"
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats[FIELD] = _zero_stats(total_rows)
            stats_path.write_text(json.dumps(stats, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        _patch_episode_stats(staging)
        manifest = {
            "format": "bw_hybrid_gripper_schema_v1",
            "source": str(source),
            "upgraded_at": datetime.now(timezone.utc).isoformat(),
            "frames": total_rows,
            "precondition": "action.rl_delta indices 7 and 15 are zero for every frame",
            "inserted_field": FIELD,
            "inserted_value": [0, 0],
            "class_names": ["KEEP_BASE", "FORCE_OPEN", "FORCE_CLOSE"],
        }
        (staging / "gripper_schema_upgrade.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    upgrade(args.source, args.output)
    print(f"Upgraded dataset written to: {args.output.expanduser().resolve()}")
    print(f"Inserted {FIELD}=[0,0] (KEEP_BASE) without modifying the source dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
