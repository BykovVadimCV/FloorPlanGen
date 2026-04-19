"""Command-line entry: `python -m floorplangen ...`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .api import generate_sample
from .config import GeneratorConfig


def _write_sample(out_dir: Path, idx: int, sample, save_class_mask: bool) -> None:
    stem = f"sample_{idx:06d}"
    img_path = out_dir / "images" / f"{stem}.png"
    mask_path = out_dir / "masks" / f"{stem}.png"
    meta_path = out_dir / "metadata" / f"{stem}.json"
    yolo_path = out_dir / "labels" / f"{stem}.txt"

    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    yolo_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_path), sample.image)
    cv2.imwrite(str(mask_path), sample.mask)
    if save_class_mask:
        cls_dir = out_dir / "class_masks"
        cls_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cls_dir / f"{stem}.png"), sample.class_mask)

    meta_path.write_text(json.dumps(sample.metadata, indent=2), encoding="utf-8")
    yolo_path.write_text("\n".join(sample.yolo_labels), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="floorplangen",
                                description="Generate synthetic floor-plan training samples.")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, nargs=2, default=(512, 512),
                   metavar=("H", "W"))
    p.add_argument("--era", choices=["scan", "digital", "soviet"], default=None)
    p.add_argument("--aggression", type=float, default=None)
    p.add_argument("--icon-pack", type=Path, default=None)
    p.add_argument("--preset", choices=["clean", "medium", "heavy"], default=None)
    p.add_argument("--monochrome-prob", type=float, default=0.70)
    p.add_argument("--save-class-mask", action="store_true")
    args = p.parse_args(argv)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GeneratorConfig()
    if args.icon_pack is not None:
        cfg.icon_pack_dir = args.icon_pack

    for i in range(args.count):
        sample = generate_sample(
            seed=args.seed + i,
            image_size=tuple(args.size),
            era=args.era,
            monochrome_prob=args.monochrome_prob,
            aggression=args.aggression,
            icon_pack_dir=str(args.icon_pack) if args.icon_pack else None,
            augmentation_preset=args.preset,
            config=cfg,
        )
        _write_sample(out_dir, i, sample, args.save_class_mask)

    print(f"Wrote {args.count} samples to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
