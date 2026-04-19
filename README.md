# FloorPlanGen

Synthetic floor-plan image generator for training a 5-class semantic segmentation U-Net
(background / wall / window / door / furniture). Implements `DESIGN.md` end-to-end.

## Install

```bash
pip install -e .[dev]
```

## Generate a batch

```bash
python -m floorplangen --count 100 --output ./out --seed 42 --size 512
```

## Programmatic use

```python
from floorplangen import generate_sample

result = generate_sample(seed=42, image_size=(512, 512))
# result.image      uint8 (H, W) or (H, W, 3)
# result.mask       uint8 (H, W) in {0, 64, 128, 192, 255}
# result.class_mask uint8 (H, W) in {0..4}
# result.yolo_labels list[str]
# result.metadata    dict
```

## Five-class schema (DESIGN §1.2)

| ID | Name       | Pixel | Meaning                                     |
|----|------------|-------|---------------------------------------------|
| 0  | Background | 0     | Everything outside the footprint + text     |
| 1  | Wall       | 64    | All wall bands (hollow interior included)   |
| 2  | Window     | 128   | Window openings                             |
| 3  | Door       | 192   | Door openings including swing arc + BTI X   |
| 4  | Furniture  | 255   | Icons, fixtures, stairs, room numbers, etc. |

## Testing

```bash
pytest                      # full suite
pytest tests/test_mask.py   # 63-case integration matrix
```

See `DESIGN.md` for the full specification.
