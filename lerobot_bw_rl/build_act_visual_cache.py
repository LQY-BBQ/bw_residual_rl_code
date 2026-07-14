"""CLI for creating the frozen ACT visual feature cache."""
from __future__ import annotations

import argparse
from pathlib import Path

from visual_cache import build_visual_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BW three-camera ACT visual feature cache.")
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset.repo_id", dest="repo_id", default=None)
    parser.add_argument("--act-policy-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = build_visual_feature_cache(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        act_policy_path=args.act_policy_path,
        cache_dir=args.cache_dir,
        device=args.device,
        use_amp=args.use_amp,
        batch_size=args.batch_size,
        dtype=args.dtype,
        overwrite=args.overwrite,
        video_backend=args.video_backend,
    )
    print(f"Visual cache: {cache.directory}")
    print(f"Shape: {cache.features.shape}, dtype={cache.features.dtype}")
    print(f"ACT fingerprint: {cache.act_fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
