"""JointState name normalization and vector assembly."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from .constants import DATASET_JOINT_FEATURE_NAMES, JOINT_NAME_ALIASES, JOINT_NAMES, KNOWN_NON_COLLECTION_JOINTS


class JointMappingError(ValueError):
    pass


def _canonicalize_names(names: Sequence[str], rename_map: Mapping[str, str]) -> list[str]:
    return [rename_map.get(str(name), str(name)) for name in names]


def extract_named_positions(msg: object, expected_names: Sequence[str], *, source_label: str, rename_map: Mapping[str, str] | None = None, allowed_extra_names: Iterable[str] | None = None, allow_trailing_unpaired_positions: bool = False) -> dict[str, float]:
    rename = dict(JOINT_NAME_ALIASES)
    if rename_map:
        rename.update(rename_map)
    names = list(getattr(msg, "name", []) or [])
    positions = list(getattr(msg, "position", []) or [])
    if len(names) > len(positions):
        raise JointMappingError(f"{source_label}: len(name)={len(names)} > len(position)={len(positions)}")
    if len(positions) > len(names):
        if not allow_trailing_unpaired_positions:
            raise JointMappingError(f"{source_label}: len(position)={len(positions)} > len(name)={len(names)}")
        positions = positions[: len(names)]
    canonical = _canonicalize_names(names, rename)
    duplicates = sorted(name for name, count in Counter(canonical).items() if count > 1)
    if duplicates:
        raise JointMappingError(f"{source_label}: duplicate names after alias mapping: {duplicates}")
    expected = list(expected_names)
    expected_set = set(expected)
    allowed = set(allowed_extra_names or set())
    unknown = sorted({name for name in canonical if name not in expected_set and name not in allowed})
    if unknown:
        raise JointMappingError(f"{source_label}: unexpected non-collection joints: {unknown}")
    values = {}
    for name, raw in zip(canonical, positions):
        if name not in expected_set:
            continue
        value = float(raw)
        if not np.isfinite(value):
            raise JointMappingError(f"{source_label}: joint {name!r} has NaN/Inf")
        values[name] = value
    missing = [name for name in expected if name not in values]
    if missing:
        raise JointMappingError(f"{source_label}: missing required joints: {missing}")
    return {name: values[name] for name in expected}


def state_from_joint_state(msg: object, *, source_label: str) -> dict[str, float]:
    return extract_named_positions(msg, JOINT_NAMES, source_label=source_label, allowed_extra_names=KNOWN_NON_COLLECTION_JOINTS, allow_trailing_unpaired_positions=True)


def joint_dict_to_vector(joint_values: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(joint_values[name]) for name in JOINT_NAMES], dtype=np.float32)


def vector_feature_names() -> list[str]:
    return list(DATASET_JOINT_FEATURE_NAMES)
