from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from upgrade_gripper_class_schema import FIELD, upgrade  # noqa: E402


def _legacy_dataset(root: Path, *, gripper_delta: float = 0.0) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "total_frames": 2,
        "features": {
            "action.rl_delta": {
                "dtype": "float32",
                "shape": [16],
                "names": [f"joint_{index}" for index in range(16)],
            }
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    delta = np.zeros((2, 16), dtype=np.float32)
    delta[1, 15] = gripper_delta
    table = pa.table({"action.rl_delta": pa.array(delta.tolist(), type=pa.list_(pa.float32()))})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")


def test_upgrade_adds_keep_base_to_copy_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _legacy_dataset(source)
    upgrade(source, output)

    assert FIELD not in pq.read_table(next((source / "data").rglob("*.parquet"))).column_names
    upgraded = pq.read_table(next((output / "data").rglob("*.parquet")))
    assert upgraded[FIELD].to_pylist() == [[0, 0], [0, 0]]
    assert pd.read_parquet(next((output / "data").rglob("*.parquet")))[FIELD].tolist()[0].tolist() == [0, 0]
    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["features"][FIELD]["shape"] == [2]
    assert (output / "gripper_schema_upgrade.json").is_file()


def test_upgrade_rejects_legacy_continuous_gripper_residual(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _legacy_dataset(source, gripper_delta=0.01)
    with pytest.raises(ValueError, match="Refusing automatic upgrade"):
        upgrade(source, output)
    assert not output.exists()
