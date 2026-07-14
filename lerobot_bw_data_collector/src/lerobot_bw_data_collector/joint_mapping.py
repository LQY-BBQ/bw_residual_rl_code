"""JointState name normalization and vector assembly."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from .constants import (
    ARM_JOINT_NAMES,
    DATASET_JOINT_FEATURE_NAMES,
    GRIPPER_JOINT_NAMES,
    JOINT_NAME_ALIASES,
    JOINT_NAMES,
    KNOWN_NON_COLLECTION_JOINTS,
)


class JointMappingError(ValueError):
    """Raised when a JointState message cannot be mapped into dataset order."""


def _canonicalize_names(names: Sequence[str], rename_map: Mapping[str, str]) -> list[str]:
    return [rename_map.get(str(name), str(name)) for name in names]


def extract_named_positions(
    msg: object,
    expected_names: Sequence[str],
    *,
    source_label: str,
    rename_map: Mapping[str, str] | None = None,
    allowed_extra_names: Iterable[str] | None = None,
    allow_trailing_unpaired_positions: bool = False,
) -> dict[str, float]:
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
        trailing = [float(value) for value in positions[len(names) :]]
        if not all(np.isfinite(value) for value in trailing):
            raise JointMappingError(f"{source_label}: trailing unnamed position values contain NaN/Inf")
        positions = positions[: len(names)]

    canonical_names = _canonicalize_names(names, rename)
    duplicates = sorted(name for name, count in Counter(canonical_names).items() if count > 1)
    if duplicates:
        raise JointMappingError(f"{source_label}: duplicate joint names after alias mapping: {duplicates}")

    expected = list(expected_names)
    expected_set = set(expected)
    allowed_extras = set(allowed_extra_names or set())
    unknown_extras = sorted({name for name in canonical_names if name not in expected_set and name not in allowed_extras})
    if unknown_extras:
        raise JointMappingError(f"{source_label}: unexpected non-collection joints: {unknown_extras}")

    value_by_name: dict[str, float] = {}
    for canonical_name, raw_value in zip(canonical_names, positions):
        if canonical_name not in expected_set:
            continue
        value = float(raw_value)
        if not np.isfinite(value):
            raise JointMappingError(f"{source_label}: joint {canonical_name!r} has NaN/Inf value")
        value_by_name[canonical_name] = value

    missing = [name for name in expected if name not in value_by_name]
    if missing:
        raise JointMappingError(f"{source_label}: missing required joints: {missing}")
    return {name: value_by_name[name] for name in expected}


def state_from_joint_state(msg: object, *, source_label: str) -> dict[str, float]:
    return extract_named_positions(
        msg,
        JOINT_NAMES,
        source_label=source_label,
        allowed_extra_names=KNOWN_NON_COLLECTION_JOINTS,
        allow_trailing_unpaired_positions=True,
    )


def action_from_joint_states(
    arm_msg: object,
    gripper_msg: object,
    *,
    arm_source_label: str,
    gripper_source_label: str,
) -> dict[str, float]:
    arm_positions = extract_named_positions(
        arm_msg,
        ARM_JOINT_NAMES,
        source_label=arm_source_label,
        allowed_extra_names=KNOWN_NON_COLLECTION_JOINTS,
    )
    gripper_positions = extract_named_positions(
        gripper_msg,
        GRIPPER_JOINT_NAMES,
        source_label=gripper_source_label,
        rename_map={"left_gripper": "left_gripper_joint", "right_gripper": "right_gripper_joint"},
    )
    return {**arm_positions, **gripper_positions}


def vector_from_joint_state(msg: object, *, source_label: str) -> np.ndarray:
    """Extract a full 16-D debug action JointState."""
    return joint_dict_to_vector(extract_named_positions(msg, JOINT_NAMES, source_label=source_label))


def joint_dict_to_vector(joint_values: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(joint_values[name]) for name in JOINT_NAMES], dtype=np.float32)


def vector_feature_names() -> list[str]:
    return list(DATASET_JOINT_FEATURE_NAMES)
