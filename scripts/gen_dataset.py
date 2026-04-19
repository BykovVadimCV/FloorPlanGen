"""Bulk corpus generation using multiprocessing.

Usage:
    python scripts/gen_dataset.py --count 1000 --output D:/datasets/fpg_v1 \\
                                   --seed-start 0 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2  # noqa: E402


def _generate_one(args: tuple) -> dict:
    (seed, image_size, icon_pack, era, preset, mono_prob,
     aggression, out_root, save_class_mask, pkg_root) = args
    # Late import so worker processes pick up env vars
    if pkg_root and pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from floorplangen import generate_sample

    sample = generate_sample(
        seed=seed,
        image_size=tuple(image_size),
        era=era,
        monochrome_prob=mono_prob,
        aggression=aggression,
        icon_pack_dir=icon_pack,
        augmentation_preset=preset,
    )
    out_root = Path(out_root)
    stem = f"sample_{seed:08d}"
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_root / "images" / f"{stem}.png"), sample.image)
    cv2.imwrite(str(out_root / "masks" / f"{stem}.png"), sample.mask)
    if save_class_mask:
        (out_root / "class_masks").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "class_masks" / f"{stem}.png"), sample.class_mask)
    (out_root / "metadata" / f"{stem}.json").write_text(
        json.dumps(sample.metadata, indent=2, default=str), encoding="utf-8")
    (out_root / "labels" / f"{stem}.txt").write_text(
        "\n".join(sample.yolo_labels), encoding="utf-8")
    return {"seed": seed, "stem": stem, **sample.metadata}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--size", type=int, nargs=2, default=(512, 512))
    p.add_argument("--era", choices=["soviet", "transitional", "modern"], default=None)
    p.add_argument("--preset", choices=["clean", "medium", "heavy"], default=None)
    p.add_argument("--aggression", type=float, default=None)
    p.add_argument("--monochrome-prob", type=float, default=0.70)
    p.add_argument("--icon-pack", type=Path,
                   default=Path(__file__).resolve().parent.parent / "icons")
    p.add_argument("--save-class-mask", action="store_true")
    args = p.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    pkg_root = str(Path(__file__).resolve().parent.parent)
    jobs = []
    for i in range(args.count):
        seed = args.seed_start + i
        jobs.append((seed, tuple(args.size), str(args.icon_pack),
                     args.era, args.preset, args.monochrome_prob,
                     args.aggression, str(args.output), args.save_class_mask,
                     pkg_root))

    summary: list[dict] = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_generate_one, j) for j in jobs]
        for k, f in enumerate(as_completed(futures), 1):
            try:
                summary.append(f.result())
            except Exception as ex:  # noqa: BLE001
                print(f"[warn] job failed: {ex!r}", file=sys.stderr)
            if k % 25 == 0 or k == len(futures):
                dt = time.time() - t_start
                rate = k / max(dt, 1e-6)
                print(f"  [{k:>6d}/{len(futures):>6d}] {rate:5.1f} samples/s",
                      flush=True)

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(summary)} samples to {args.output} "
          f"in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
