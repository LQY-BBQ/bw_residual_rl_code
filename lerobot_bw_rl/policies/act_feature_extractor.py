"""Backward-compatible imports for the new exact ACT shared visual encoder."""
from .act_shared_encoder import (  # noqa: F401
    BW_IMAGE_KEYS,
    FrozenACTBundle,
    act_policy_fingerprint,
    extract_pooled_projected_visual_features,
    load_frozen_act_bundle,
    prepare_raw_observation,
    raw_observation_from_dataset_item,
    validate_bw_act_policy,
)
