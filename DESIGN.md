# DATASET GENERATOR REWORK DESIGN DOCUMENT



#### 

#### Chapter 1 — Scope, Schema, and the Generator Contract

###### 

##### 1.1 Purpose of This Document



This specification defines a **synthetic floor-plan image generator** whose sole purpose is to produce training data for a five-class semantic segmentation model. Every architectural, aesthetic, and procedural decision made in Chapters 2 through 11 exists in service of one constraint: the downstream U-Net must generalize to real residential floor plans spanning three decades of drawing conventions, paper stocks, and reproduction technologies. The generator is not a design tool. It does not model plumbing, structural loads, or building codes. It models *how floor plans look on paper*, and it does so with enough variation that a model trained entirely on its output can transfer to scanned documents it has never encountered.

Floor plans are essential graphical representations of buildings used across architecture, engineering, construction, and real estate — but their necessary depth, complexity, and purpose vary significantly, with no consistent standard across industries or countries. That absence of a unified standard is precisely the problem this generator must solve: rather than curating a heterogeneous corpus of real drawings (time-consuming, rights-encumbered, and geographically skewed), we manufacture a corpus that deliberately spans the visual range.

Complementary to real-world datasets, synthetic datasets have been developed to support algorithmic generation and simulation — procedural approaches offer clean annotations and controlled geometric variation, though they may lack the nuanced constraints and patterns found in human-designed spaces. This generator is designed to close that gap by building era-specific aesthetic noise, hand-drawn imperfection, and scan degradation directly into the synthesis pipeline.

## 

##### 1.2 The Five-Class Schema



The generator targets exactly **five semantic classes**. This is a deliberate compression. The task of floor plan analysis has been extensively explored in recent years, with a common thread being the reliance on deep learning to extract geometric primitives such as rooms, walls, doors, and windows. Many published systems expand far beyond this — CubiCasa5K, for instance, annotates over 80 floor plan object categories — but a broader schema demands more annotated data and imposes fragile class boundaries. The five-class system used here is a principled reduction to the minimal set that supports downstream vectorisation and room-topology extraction.



|Class ID|Name|Pixel Value (mask)|Meaning|
|-|-|-|-|
|**0**|Background|`0`|Everything outside the building boundary; also text labels, dimension numbers, and annotation arrows|
|**1**|Wall|`64`|All structural and partition walls, regardless of thickness, drawing style, or orientation|
|**2**|Window|`128`|All window openings as drawn; the sill/reveal region between inner and outer wall lines|
|**3**|Door|`192`|All door openings, including swing arcs and the cross-ridge BTI symbol|
|**4**|Furniture / Contents|`255`|Every non-structural mark inside a room: equipment icons, sanitary fixtures, stairs, radiators, room-type numbers, arrows internal to the plan, and all other drawn content that is neither wall nor opening|



The pixel values above are the single-channel grayscale encoding written into the mask PNG. They are equally spaced across the 8-bit range for legibility during debugging; the training pipeline should re-encode them to contiguous integer class indices (0–4) before computing loss.

###### 

###### 1.2.1 What Lives in Class 0 (Background)



Background is the catch-all for everything that is *not part of the floor plan itself*. This includes:

* The white or yellowed paper margin beyond the building footprint
* Dimension arrows and the numeric annotations that accompany them (e.g., `4,82 m`)
* Title blocks, north arrows, scale bars, and sheet borders
* Any stray ink, fold lines, or scanner artefacts that fall outside all other classes

The decision to mask dimension arrows and numbers as background (class 0) rather than as a separate class is deliberate. A key segmentation challenge in real drawings is dense semantic clutter from furniture, text, and dimension lines — by assigning all such annotations to background, we train the model to treat them as transparent noise, exactly as a human reader does. Chapter 8 documents the arrow geometry in detail and confirms this mask assignment.

###### 

###### 1.2.2 What Lives in Class 1 (Wall)



All structural marks that form the boundary between interior spaces fall into class 1:

* Solid filled walls (the most common convention post-2000)
* Hollow-outline walls (common in Soviet-era and technical-drawing styles)
* Diagonal walls at any angle (see Chapter 4)
* Wall segments of any thickness or junction type
* Hatch patterns inside wall bodies

An architectural floor plan includes objects such as walls, doors, windows, and stairs, with walls typically defining the main layout of the floor and individual rooms. In construction-type floor plans, walls are represented by different drawings — solid-wall, dot-wall, diagonal-wall, hollow-wall, and gray-wall — based on the raw materials used for construction. This generator produces solid and hollow variants (Chapter 5); both map to class 1 without distinction, because the downstream use case (room topology extraction) does not require material identification.

**Critical hollow-wall rule**: when a wall is drawn as two parallel contour lines with a white or paper-toned interior gap, *both the contour lines and the interior gap* are labelled class 1 in the mask. The image will show white between the lines; the mask will not. This is a deliberate choice. Teaching the model to recognize hollow-wall topology requires that it see the entire band as a single class, not a sandwich of wall/background/wall. Chapter 11 enforces this as a polygon-fill rule.

###### 

###### 1.2.3 What Lives in Class 2 (Window) and Class 3 (Door)



Windows and doors are the two opening classes. Overlapping class labels between wall and opening classes introduce ambiguity during training, as the model receives conflicting signals for pixels that belong to both; the correct approach is to generate separate binary masks for doors and windows and subtract them cleanly from the wall mask. This generator implements that protocol directly: openings are carved out of wall polygons before mask writing, ensuring zero overlap between class 1 and classes 2/3.

Both window and door classes are **axis-aligned only** in this generator. Diagonal walls receive no openings (see Chapter 6 for the rationale). The BTI cross-ridge door symbol — a rectangle bisected by an X, inherited from v6.0 — is assigned to class 3 in its entirety, including the diagonal strokes.

###### 

###### 1.2.4 What Lives in Class 4 (Furniture / Contents)



Class 4 is explicitly a *catch-all for non-structural interior content*. Its full enumeration:



|Sub-type|Examples|
|-|-|
|**Sanitary / kitchen icons**|Bath, toilet, sink, hob, refrigerator|
|**Stairs**|Stair treads and direction arrows inside the stair polygon|
|**Radiators**|Horizontal fin symbol on exterior walls|
|**Room-type numbers**|Circled or plain numerals keyed to a legend|
|**Internal arrows**|Room-dimension arrows that fall wholly inside the footprint|
|**Miscellaneous icons**|Any other composited PNG from the icon pack (Chapter 7)|



Instance segmentation for individual furniture, fixtures, and building elements is a standard application in architectural floor plan understanding, but for the five-class schema all such elements collapse into a single class. The model is not required to distinguish a toilet from a radiator — it is required to distinguish *all of them* from walls and openings. This dramatically reduces the annotation burden and makes the class boundaries clean and learnable.

The practical consequence is that staircase polygons — which visually resemble a regular structural element — are nevertheless class 4. Their interior lines are drawn content, not structural boundaries. The stair *enclosure walls*, if present, are class 1.

#### 

##### 1.3 Generator Inputs and Outputs

###### 

###### 1.3.1 Inputs (per-image call)



The generator is invoked as a Python callable with the following parameters, all of which have defaults that can be overridden by the plugin strategy system inherited from v6.0:

```python
generate\_sample(
    seed: int,                   # full reproducibility
    image\_size: tuple\[int,int],  # output pixel dimensions, default (512, 512)
    era: Literal\["soviet", "scan", "digital"] | None,  # None → sampled by era\_mix
    era\_mix: dict\[str, float],   # {"scan": 0.60, "digital": 0.25, "soviet": 0.15}
    monochrome\_prob: float,      # default 0.70
    aggression: float,           # footprint complexity scalar, see Ch. 2
    icon\_pack\_dir: Path,         # root of icons/<category>/<name>.png tree
    augmentation\_preset: str,    # "heavy" | "medium" | "clean"
)
```

The `seed` parameter makes every output fully deterministic given the same input vector. This is a hard requirement: training pipelines that re-generate samples on the fly must produce identical image–mask pairs on every epoch.

###### 

###### 1.3.2 Outputs (per-image call)



```python
@dataclass
class GeneratorOutput:
    image: np.ndarray          # uint8, shape (H, W) or (H, W, 3)
    mask: np.ndarray           # uint8, shape (H, W), values in {0,64,128,192,255}
    class\_mask: np.ndarray     # uint8, shape (H, W), values in {0,1,2,3,4}
    yolo\_labels: list\[str]     # YOLO-format bounding boxes, compatible with v6.0
    metadata: dict             # era, seed, room\_count, aggression, icon\_list, ...
```

The `image` field is monochrome (single channel) in ≥70 % of outputs and three-channel otherwise. The `mask` field is the single-channel PNG-encodable mask with the pixel values from §1.2. The `class\_mask` is the same information re-encoded as contiguous integers 0–4, ready for direct consumption by a CrossEntropyLoss or Dice loss. Downstream pipelines typically map pixel values to class indices via a `\_classes.csv` lookup; the `class\_mask` output eliminates that step.

YOLO labels and the metadata JSON are pass-through compatible with the v6.0 output contract, ensuring that detection heads trained on v6.0 data can continue consuming outputs from this generator without a schema migration.

##### 

##### 1.4 Era System and Target Distribution



Real floor plan corpora are not drawn from a single visual distribution. Floor plan depth, complexity, and purpose vary significantly, leading to diverse standards not only among industries but also across countries. The generator models three distinct drawing eras, each with its own aesthetic system (detailed in Chapter 9) and degradation preset (Chapter 10):

###### 

###### Era 1 — Post-2000 Scan (target weight: **60%**)



These are digitally drafted plans subsequently printed and scanned. They are the dominant real-world source. Visual characteristics:

* Clean, uniform wall lines of consistent pixel width
* Off-white or pure white background with possible yellowing at margins
* Scan-induced noise: slight skew (≤ 5°), gaussian blur, JPEG compression artefacts, occasional roller marks
* Predominantly monochrome; colour tints rare
* Crisp opening symbols, BTI conventions, typed numeric annotations

### 

###### Era 2 — Modern Digital (target weight: **25%**)



These are plans drawn and exported directly from CAD or BIM software with no print/scan step. Visual characteristics:

* Pixel-perfect lines with no sub-pixel jitter
* Pure white background
* Possible thin colour fills for room types (but monochrome path suppresses these)
* Standard architectural symbols, often smaller than scan-era plans
* Near-zero augmentation required

### 

###### Era 3 — Soviet Hand-Drawn (target weight: **15%**)



The BTI — Biuro tekhnicheskoi inventarizatsii — was the state or municipal organisation responsible for real estate record and stocktaking. BTI technical drawings from the Soviet era and early post-Soviet period have a highly distinctive look driven by manual drafting conventions. Visual characteristics:

* Line-end imperfections: slight overshoot, pressure variation, ink bleed
* Faded or uneven ink density across a single stroke
* Yellowed, textured paper stock (laid or wove)
* Hand-printed Cyrillic annotation in a range of technical lettering styles
* Thicker wall representation; sometimes cross-hatching rather than solid fill
* Soviet architecture spanned multiple distinct styles across its history, and the floor plan conventions varied accordingly — Khrushchev-era panel blocks differ visually from Stalinist-era plans, which in turn differ from late Soviet modernist drawings

The 60/25/15 split reflects estimated prevalence in the real-world corpora that the trained model will encounter. It is a soft prior, not a hard cap: per-batch sampling draws from this distribution with replacement, so any individual batch will differ.

##### 

##### 1.5 Monochrome Prior



**At least 70% of all generated images must be single-channel (grayscale).** This is not an aesthetic preference — it is a data-distribution constraint. The overwhelming majority of scanned and hand-drawn floor plans encountered in practice are monochrome. The lack of common standards complicates the transition to digital floor plans; detailed vector-based formats provide precise spatial information but are complex to produce, so rasterised grayscale scans remain the most common form encountered in practice.

Training on a corpus that is 70 % monochrome ensures that the model does not learn colour as a proxy for class membership. When colour images are generated (the remaining ≤30 %), they use only the palettes appropriate to the `digital` era: light room-fill tints and thin colour-coded wall types. Augmentation (Chapter 10) never introduces colour into an image that was generated monochrome.

The `monochrome\_prob` parameter controls this at call time. The default of `0.70` matches the target; it may be raised but should not be lowered below `0.60` without re-evaluating generalisation on real scan corpora.

##### 

##### 1.6 Relationship to v6.0 and the Plugin Strategy System



This generator extends v6.0. The following v6.0 components are **unchanged**:

* Plugin strategy system (era themes register via the same interface)
* YOLO label format and bounding-box generation logic
* Metadata JSON schema (new keys are additive, not breaking)
* The cross-ridge BTI door symbol renderer

The following v6.0 components are **replaced or substantially extended**:

* Footprint generation (Chapter 2 introduces aggressive non-rectangular shapes)
* Room subdivision (Chapter 3 extends BSP to arbitrary polygons)
* Wall rendering (Chapter 5 adds the wall graph and hollow-wall mode)
* Mask writing (Chapter 11 enforces the polygon-fill protocol and write-order)

All replacements are implemented as drop-in strategy substitutions within the existing plugin interface. A v6.0 caller that does not pass `aggression` or `era` parameters will receive outputs visually comparable to v6.0 with no code changes required.

##### 

##### 1.7 What This Chapter Defines for All Subsequent Chapters



Every subsequent chapter takes the following as fixed and non-negotiable:

1. **Five classes, fixed IDs.** No chapter may introduce a sixth class or reassign an ID.
2. **Mask pixel values are {0, 64, 128, 192, 255}.** All polygon-fill operations target these values.
3. **Augmentation touches images only, never masks.** This constraint, introduced here, is absolute. It is repeated in Chapters 10 and 11 for emphasis.
4. **Era sampling governs aesthetics end-to-end.** A sample drawn as `soviet` uses Soviet wall styles (Ch. 5), Soviet opening symbols (Ch. 6), Soviet hand-drawn theme (Ch. 9), and Soviet heavy degradation (Ch. 10). The era token is the single parameter that binds all aesthetic subsystems.
5. **Monochrome ≥ 70%.** Any new visual element introduced in later chapters must be renderable in grayscale.

The build order that follows from these constraints — Chapters 2 → 3 → 5 → 4 → 6 → 7 → 8 → 9/10 → 11 — is documented in Chapter 12 along with the risks that accumulate at each stage.

#### 

#### Chapter 2 — Aggressive Footprint Generation

##### 

##### 2.1 Why Footprint Shape Is the Hardest Visual Change



Every downstream chapter — subdivision, wall placement, openings, icons — operates *inside* the outer boundary polygon. If that polygon is a rectangle, the entire pipeline collapses toward the behaviour of v6.0. The footprint is therefore the root of all visual variety, and generating footprints that look like real residential buildings rather than video-game placeholder boxes is the single most consequential change this generator makes relative to its predecessor.

The problem is well-established in procedural generation research. There are two fundamentally different approaches to building footprint generation: an additive/growth-based approach, which is good for houses or old buildings that frequently have extensions added onto the side, and a subtractive approach, which is better for very large commercial buildings that often fill most of their lot. This generator uses a **hybrid**: it begins subtractively — starting from a seed rectangle and cutting away corner and edge regions — then optionally applies additive extrusions to produce courtyard wings and staircase projections. The key innovation over naive random cutting is the `aggression` scalar, which controls both the number of operations and their depth, giving a single continuous knob from almost-rectangular to highly complex.

The production of virtual buildings with both interiors and exteriors composed by non-rectangular shapes — convex or concave n-gons — at the floor-plan level is still seldomly addressed in the literature. This chapter fills that gap for the specific case of residential floor plans.

##### 

##### 2.2 The Polygon Primitive Library



The generator maintains a library of **named base shapes**. These are not the final footprints — they are the *seeds* from which the subtractive/additive pipeline begins. Each primitive is defined parametrically in a normalised 1×1 unit square and is then scaled to the target canvas before any operations are applied.

###### 

###### 2.2.1 Primitive Catalogue



|ID|Name|Vertex count|Description|
|-|-|-|-|
|`RECT`|Rectangle|4|The degenerate base case; used at `aggression = 0`|
|`L`|L-shape|6|One corner quadrant removed|
|`T`|T-shape|8|One edge-centre rectangle subtracted, leaving two wings|
|`U`|U-shape|8|Two symmetric corner quadrants removed on same side, producing a courtyard|
|`Z`|Z-shape|6|Diagonal offset; two opposing corner quadrants removed|
|`STAIR`|Staircase|8–12|Step-profile shape, 2–3 steps; models panel-block stairwell projections|
|`BEVEL`|Bevelled rectangle|8|All four corners clipped at 45°; common in Khrushchev-era plan shapes|



Each non-rectangular primitive has one or more **aspect parameters** that control its proportions independently of the overall bounding box. For example, the `L` shape carries `arm\_width\_x` and `arm\_width\_y` (both in \[0.20, 0.55]) controlling how thick each remaining arm is. These are sampled per image.

If all of these basic shapes are constrained to axis-aligned polygons with an arbitrary rotation in reference to the building's orientation, they can be hierarchically subdivided into separate areas. Visual comparison with many real-world floor plans shows that they typically match well.

###### 

###### 2.2.2 Primitive Selection Probabilities



The selection probability of each primitive is a function of `aggression` (§2.3). At `aggression = 0`, `RECT` is drawn with probability 1.0. At `aggression = 1.0`, the full distribution is active:

```python
PRIMITIVE\_WEIGHTS = {
    "RECT":  0.05,
    "L":     0.28,
    "T":     0.18,
    "U":     0.17,
    "Z":     0.10,
    "STAIR": 0.12,
    "BEVEL": 0.10,
}
```

These weights reflect the approximate prevalence of each shape in the Soviet-era and post-Soviet residential corpus. Panel-block apartment buildings are overwhelmingly L, T, and U in their footprint when seen in plan, with staircase projections as secondary features.

###### 

###### 2.3 The `aggression` Scalar



`aggression` is a float in \[0.0, 1.0], passed directly from the top-level generator call (Chapter 1). It acts as a **gain control** over every stochastic decision in this chapter:

```
aggression = 0.0  →  rectangular seed, zero cutouts, no extrusions
aggression = 0.5  →  non-rectangular seed, 1–2 cutouts, minor extrusions  
aggression = 1.0  →  complex seed, 3–5 cutouts, multi-step extrusions, deep notches
```

Formally, `aggression` modulates two independent quantities:

**Cutout count** `n\_cuts`:

```python
n\_cuts = round(aggression \* np.random.triangular(0, 3, 5))
```

The triangular distribution peaks at 3 for maximum aggression, avoiding the pathological all-or-nothing behaviour of a uniform sample.

**Cutout depth** `d\_max`: each rectangular cutout is bounded in its maximum side length by:

```python
d\_max = aggression \* np.random.uniform(0.15, 0.45)
```

at fractions of the axis-aligned bounding box dimension. This prevents cuts that consume more than 45% of either dimension, which would make the resulting polygon unsubdivide-able in Chapter 3.

The immediately obvious problem with a subtractive pipeline is area. Most functions remove some space from the shape, which means that if too many are chained together we could end up with a tiny little footprint that can't have a reasonable floor plan. For this reason there are also conditional steps in the pipeline which perform an action, check if the condition is ok, and execute a fallback if not. The generator implements this via the validator (§2.5), which halts the cutout loop and triggers a fallback as soon as any guard condition fails.

##### 

##### 2.4 The Generation Pipeline



The pipeline runs in five sequential stages, all operating on Shapely `Polygon` objects:

###### 

###### Stage 1 — Seed



1. Sample the aspect ratio `ar \~ Uniform(0.55, 1.82)`. This bounds the bounding box to avoid extremely narrow buildings.
2. Select a primitive by weighted sample (§2.2.2, modulated by `aggression`).
3. Instantiate the primitive as a Shapely `Polygon` at the target canvas scale.



###### Stage 2 — Rectangular Cutouts



Iterate `n\_cuts` times. Each iteration:

1. Choose a **target edge**: one of the four axis-aligned bounding-box sides, selected with weights that favour exterior-facing long edges.
2. Sample a **notch rectangle**: width `w \~ Uniform(0.10, d\_max)` and depth `d \~ Uniform(0.10, d\_max)` along the chosen side, at a random position that keeps at least `min\_interior\_width` clearance on both flanks.
3. Compute `candidate = current\_polygon.difference(notch\_rectangle)`.
4. If the candidate passes the validator (§2.5), commit it. Otherwise, discard the notch and continue to the next iteration.

Subtractive generation works by starting with the lot shape and slicing bits off. This is best for big buildings which almost completely fill their lot. For residential floor plans, which always fill the canvas completely (the footprint *is* the drawing, not a building within a larger lot), the subtractive approach maps cleanly: every cut is a recess in the building facade.

###### 

###### Stage 3 — Additive Extrusions (conditional)



If `aggression > 0.4` and the number of committed cuts is less than `round(aggression \* 2)`, the pipeline may add one or two **extrusion rectangles** — small rectangular annexes attached to an existing edge. These model stairwell projections and lift shafts common in Soviet panel blocks.

Extrusion parameters:

* Width along the edge: `Uniform(0.06, 0.18)` × edge length
* Depth perpendicular to the edge: `Uniform(0.04, 0.12)` × bounding box dimension
* Applied via `current\_polygon.union(extrusion\_rect)`, immediately followed by `make\_valid()` and the validator

###### 

###### Stage 4 — Corner Bevelling (conditional)



If the primitive is `BEVEL` or if `aggression > 0.65` and at least one 90° convex corner remains, sample 1–4 convex corners and apply a chamfer. The chamfer is implemented as:

```python
chamfer = Polygon(\[
    corner - t \* edge1\_unit,
    corner,
    corner - t \* edge2\_unit,
])
footprint = footprint.difference(chamfer)
```

where `t \~ Uniform(8, 22)` pixels. This produces the truncated corners common in Soviet-era apartment block plan shapes. Bevelled corners also serve Chapter 4: they are the primary attachment points for external diagonal walls.

###### 

###### Stage 5 — Normalisation



1. Call `shapely.make\_valid()` to repair any degenerate geometry introduced by floating-point accumulation. Shapely does not check the topological simplicity or validity of instances when they are constructed, as the cost is unwarranted in most cases. Repair must therefore be applied explicitly at the end of every pipeline stage that modifies the polygon.
2. Translate and scale the validated polygon to fill the canvas with a margin of `canvas\_margin` pixels (default: 24px on each side).
3. Round all vertex coordinates to the nearest integer pixel. This avoids sub-pixel rendering artefacts when the polygon is later rasterised for the mask.

##### 

##### 2.5 The Validator



The validator is called after every polygon-modifying operation (Stages 2, 3, and 4). It returns `True` if the candidate polygon is acceptable, `False` if the operation that produced it must be rolled back. All checks operate on the candidate polygon after `make\_valid()` has been applied.

###### 

###### Check 1 — Shapely Validity



```python
assert candidate.is\_valid
assert not candidate.is\_empty
assert isinstance(candidate, Polygon)  # not MultiPolygon
```

A valid Polygon may not possess any overlapping exterior or interior rings. A valid MultiPolygon may not collect any overlapping polygons. The `isinstance` guard is critical: a difference operation that severs a narrow isthmus will silently return a `MultiPolygon` rather than raising an exception. Rings of a valid Polygon may not cross each other, but may touch at a single point only. Shapely will not prevent the creation of invalid features, but when they are operated on the results might be wrong or exceptions might be raised. Catching `MultiPolygon` output here prevents a cascade of downstream failures in Chapter 3's BSP.

###### 

###### Check 2 — Single-Connected Topology



```python
assert len(candidate.interiors) == 0
```

The footprint must have no holes. A polygon with an interior ring would represent a courtyard building — plausible architecturally, but not supported by Chapter 3's subdivision algorithm and excluded from this generator's scope.

###### 

###### Check 3 — Minimum Interior Width



The **minimum interior width** is approximated by computing the minimum width of the Shapely `minimum\_rotated\_rectangle` of the polygon:

```python
mrr = candidate.minimum\_rotated\_rectangle
edge\_lengths = sorted(\[mrr.exterior.length / 4 ...])  # approximate side lengths
assert edge\_lengths\[0] >= MIN\_INTERIOR\_WIDTH\_PX  # default: 80px at 512×512
```

This is a proxy measure. It fails conservatively: some valid narrow wings may be rejected if their minimum rotated rectangle is narrow. The cost (a slightly lower acceptance rate for extreme L and U shapes) is acceptable. The alternative — computing true medial axis width — is too slow for a per-operation guard.

A more direct check is applied to every **straight corridor** in the polygon: for each pair of parallel edges closer than `MIN\_INTERIOR\_WIDTH\_PX`, the check fails. This catches thin-arm shapes that the MRR check misses.

###### 

###### Check 4 — Angle Guard



```python
for i, vertex in enumerate(exterior\_coords\[:-1]):
    angle = compute\_interior\_angle(prev, vertex, next)
    assert angle >= MIN\_ANGLE\_DEG  # default: 60°
```

No interior angle may be less than 60°. This rules out extremely acute re-entrant corners that would produce degenerate wall junctions in Chapter 5 and prevent opening placement in Chapter 6.

###### 

###### Check 5 — Minimum Area



```python
assert candidate.area >= MIN\_AREA\_FRAC \* seed\_polygon.area  # default: 0.35
```

The candidate must retain at least 35% of the seed polygon's area. This prevents a sequence of aggressive cuts from reducing the floor plan to an unusably small space.

###### 

###### Fallback Behaviour



If any check fails, the offending operation is discarded and the loop moves to the next iteration. If all `n\_cuts` attempts fail validation, the pipeline exits with the current (partially cut) polygon rather than retrying indefinitely. This guarantees bounded generation time regardless of `aggression` value.

A conditional pipeline step can say something like: try to shrink the footprint by a given amount; if that results in an area below a threshold, instead bevel the corners by a smaller amount. The validator here implements exactly this pattern, but generalised to all five guards simultaneously.

##### 

##### 2.6 Coordinate System and Canvas Conventions



All footprint coordinates are maintained in **pixel space** throughout. The canvas origin `(0, 0)` is the top-left corner, consistent with NumPy's array indexing order. The y-axis points downward. Shapely operates with a mathematical (y-up) convention; all coordinates are passed as-is, which has no effect on 2D polygon geometry but must be remembered when converting to image-row/column indices during rasterisation.

The footprint polygon is maintained at the resolution of the output image (`image\_size` from Chapter 1). It is **not** stored in metres or any architectural unit. This eliminates a class of precision bugs that arise when converting between world coordinates and pixel coordinates during mask writing.

##### 

##### 2.7 Output Contract



The footprint generator returns a single dataclass:

```python
@dataclass
class Footprint:
    polygon: shapely.Polygon          # validated, pixel-space, integer vertices
    primitive\_id: str                 # "L", "T", "U", etc.
    n\_cuts\_applied: int               # actual cuts committed (≤ n\_cuts attempted)
    n\_extrusions\_applied: int
    bevel\_corners: list\[int]          # vertex indices that were bevelled
    aggression: float                 # the value used, for metadata JSON
    bounding\_box: tuple\[int,int,int,int]  # xmin, ymin, xmax, ymax
```

The `polygon` field is the sole input to Chapter 3's subdivision algorithm, Chapter 4's diagonal wall placement, and Chapter 5's wall graph initialisation. All downstream chapters receive a `Footprint` instance and are permitted to read any field, but **must not modify the polygon in-place**. Any modification must produce a new `Shapely` object.

&#x20;

##### 2.8 Implementation Notes and Known Edge Cases



**Z-shape topology.** The Z primitive is the most prone to validator rejection at high aggression because its offset geometry can create very narrow connecting bridges. If the Z shape fails validation more than twice in a row within a single `generate\_sample()` call, the primitive selector falls back to `L` for that sample. This is logged in `metadata\["primitive\_fallback"]`.

**STAIR extrusion accumulation.** The `STAIR` primitive already contains step-profile vertices. Applying additive extrusions (Stage 3) on top of it can produce very high vertex counts (up to 20+). This does not affect Shapely performance but *does* affect Chapter 3's PCA-based split-axis selection, which must not be confused by the near-collinear step vertices. Chapter 3 handles this by simplifying the input polygon before PCA.

**`make\_valid()` geometry changes.** Invalid geometric objects may result from simplification that does not preserve topology, and simplification may be sensitive to the order of coordinates: two geometries differing only in order of coordinates may be simplified differently. After `make\_valid()`, always re-check `isinstance(result, Polygon)` — in rare cases involving coincident edges, `make\_valid()` returns a `GeometryCollection` containing a polygon and degenerate lower-dimensional components. Extract the polygon component explicitly:

```python
from shapely.ops import make\_valid
from shapely import get\_parts

result = make\_valid(candidate)
if result.geom\_type == "GeometryCollection":
    polygons = \[g for g in get\_parts(result) if g.geom\_type == "Polygon"]
    result = max(polygons, key=lambda p: p.area)  # keep largest
```

**Pixel rounding introduces vertex drift.** Rounding vertices to integer pixels in Stage 5 can invalidate a polygon that passed all validator checks in floating-point space if two nearby vertices snap to the same pixel. Apply `polygon.simplify(0.5, preserve\_topology=True)` *after* pixel rounding, then re-run `is\_valid`.

**Performance budget.** The full footprint pipeline (all five stages including validator) runs in under 4 ms per sample on a single CPU core at 512×512 for `aggression ≤ 0.8`. At `aggression = 1.0` with a `STAIR` seed, worst-case is approximately 18 ms due to repeated `difference()` calls on high-vertex polygons. This is within the Chapter 12 performance budget.

###### &#x20;

##### 2.9 Relationship to Downstream Chapters



The `Footprint.polygon` object is the most critical dependency in the entire pipeline:



|Chapter|Dependency on footprint|
|-|-|
|**3**|BSP subdivision operates entirely inside `polygon`|
|**4**|Diagonal walls attach to edges or corners of `polygon`|
|**5**|The wall graph's exterior ring is initialised from `polygon.exterior`|
|**6**|Window/door eligibility is determined by which edges of `polygon` are exterior|
|**7**|Icon placement is clipped to `polygon` interior|
|**11**|Mask background (class 0) is the bitwise complement of the rasterised `polygon`|



The proposed method handles a wide variety of input image styles and building shapes, including non-convex polygons. It presents a method for automated reconstruction of building interiors from hand-drawn building sketches. That same requirement — handling non-convex polygons gracefully throughout the pipeline — is the reason every downstream chapter that touches geometry must be tested against the full primitive library, not just rectangles. Chapter 12's test matrix specifies one integration test per primitive type at three aggression levels as a minimum acceptance criterion.





&#x20;

#### Chapter 3 — Subdivision on Arbitrary Polygons

###### 

##### 3.1 Why Rectangle-Only BSP Fails



The classical BSP subdivision used for floor plan generation assumes a rectangular container. Binary space partitioning is a spatial subdivision algorithm that divides a rectangular dungeon area into increasingly smaller regions by making binary splits along horizontal or vertical lines. This assumption propagates cleanly through the recursion because both children of every split are themselves rectangles, and rectangles have well-defined axis-aligned widths and heights against which aspect ratio guards can be evaluated analytically.

Chapter 2 breaks this assumption entirely. Its output polygon is an L, T, U, Z, STAIR, or BEVEL shape — non-rectangular, possibly with acute re-entrant vertices, and with edges at arbitrary orientations. A naive axis-aligned split on such a shape produces children that may be degenerate slivers, disconnected fragments, or topologically invalid geometries. Shapely's split function encounters precision issues on U-shaped polygons specifically: while it may successfully split one leg of such a polygon, it can fail to split the other, suggesting fundamental precision issues within the split function itself.

This chapter defines a BSP + treemap hybrid that operates correctly on the full Chapter 2 primitive library. Every algorithmic decision — the split primitive, the split-axis selector, the treemap layout phase, the aspect guard, the fallback — is motivated by the specific failure modes that arise when the input geometry is non-rectangular.

###### &#x20;

##### 3.2 Algorithm Overview



The subdivision algorithm runs in two sequential phases:

**Phase 1 — BSP partitioning.** The footprint polygon is recursively bisected by half-plane operations into a binary tree of leaf cells. The split axis at each node is chosen by PCA on the cell's vertex coordinates (§3.5). Splitting is performed by the half-plane intersection/difference fallback (§3.4), not by `shapely.ops.split()`. Recursion terminates when a cell fails the minimum-area or aspect guard (§3.6).

**Phase 2 — Treemap layout.** Each BSP leaf cell is assigned a room type drawn from a weighted distribution. The squarified treemap algorithm (§3.7) then packs a set of target-area rectangles into each cell, using the cell's own polygon as the container rather than a bounding rectangle. Rooms that cannot be filled squarely remain as single-room cells.

The result is a flat list of `Room` polygons covering the footprint without overlap, each tagged with a room type. This list is the input to Chapter 5's wall graph.

##### &#x20;

##### 3.3 The Room Type Distribution



Before any geometry is computed, the algorithm samples a **room programme**: a list of (room\_type, relative\_area) pairs that must sum to 1.0 and tile the footprint. The distribution is drawn from:

```python
ROOM\_TYPE\_WEIGHTS = {
    "living":    0.22,
    "bedroom":   0.28,   # may appear 1–3 times
    "kitchen":   0.12,
    "bathroom":  0.08,   # may appear 1–2 times
    "corridor":  0.14,
    "balcony":   0.06,
    "utility":   0.06,
    "stairwell": 0.04,
}
```

A programme is sampled by drawing room types with replacement until their accumulated relative areas sum to approximately 1.0, with a jitter of ±15 % applied to each room's target area after sampling. Programmes with fewer than three rooms or more than nine rooms are resampled. The room count is stored in `metadata\["room\_count"]`.

Room type matters for downstream chapters:

* **Window placement** (Chapter 6) uses room type to determine whether exterior-wall openings are required.
* **Icon placement** (Chapter 7) uses room type to select which icon subcategory to draw.
* **Corridor and stairwell rooms** receive no icons and have their wall thickness set to the thinner partition variant (Chapter 5).

##### &#x20;

##### 3.4 The Split Primitive: Half-Plane Intersection/Difference

###### 

###### 3.4.1 Why `shapely.ops.split()` Cannot Be Used



The chapter outline flags `shapely.ops.split()`'s silent failure on polygons with holes. The underlying cause is structural. When `split()` is applied to a polygon with a hole using a line that crosses only some of its boundaries, the extra coordinates go into the existing exterior and hole boundaries but do not connect the hole edges to the exterior edges into a single polygon. The result is that `split()` simply returns the original polygon unchanged. This is not an exception — it is a silent no-op, which is the most dangerous possible failure mode in a recursive algorithm: the recursion terminates immediately because the cell appears unsplit, and the entire subtree is discarded.

The problem is more general than holes. If the splitter does not split the geometry, a collection with a single geometry equal to the input geometry is returned. For non-rectangular polygons, a split line that enters and exits through the same boundary edge (which can happen with re-entrant L and U shapes) produces exactly this outcome.

Additionally, when splitting a U-shaped polygon, Shapely may successfully split one leg but fail to split the other — and users working around this issue have had to implement their own splitting algorithms to obtain correct results.

The conclusion is unambiguous: `shapely.ops.split()` must not be used anywhere in this pipeline. The half-plane method, described below, is used exclusively.

###### 

###### 3.4.2 The Half-Plane Intersection/Difference Method



Given a cell polygon `P` and a split line defined by point `p0` and direction vector `d`:

1. Construct a **half-plane polygon** `H+` that is a very large rectangle (10× the canvas bounding box) aligned so that one of its edges lies exactly on the split line, covering all points on one side.
2. Compute `child\_A = P.intersection(H+)`.
3. Compute `child\_B = P.difference(H+)`.
4. Call `make\_valid()` on both children.
5. Assert both are non-empty simple `Polygon` instances (not `MultiPolygon` or `GeometryCollection`).

```python
def half\_plane\_split(polygon: Polygon, p0: np.ndarray, normal: np.ndarray,
                     canvas\_size: int = 10000) -> tuple\[Polygon, Polygon]:
    """
    Split polygon by the line through p0 with the given normal.
    Returns (child\_A, child\_B) or raises SplitFailure.
    """
    d = np.array(\[-normal\[1], normal\[0]])  # line direction perpendicular to normal
    far = canvas\_size
    # Build a large rectangle on the positive-normal side
    h\_plus = Polygon(\[
        p0 + far \* d + far \* normal,
        p0 - far \* d + far \* normal,
        p0 - far \* d - 0.001 \* normal,
        p0 + far \* d - 0.001 \* normal,
    ])
    child\_a = make\_valid(polygon.intersection(h\_plus))
    child\_b = make\_valid(polygon.difference(h\_plus))
    
    for child in (child\_a, child\_b):
        if child.is\_empty or child.geom\_type != "Polygon":
            raise SplitFailure("half\_plane\_split produced non-Polygon child")
    return child\_a, child\_b
```

The `0.001 \* normal` offset on the negative side of the half-plane prevents the split line itself from lying exactly on the rectangle boundary, which would cause floating-point coincidence issues in GEOS (Shapely's underlying geometry engine). If any rings cross each other, the feature is invalid and operations on it may fail. The offset ensures the half-plane boundary strictly crosses the polygon interior.

###### 

###### 3.4.3 MultiPolygon Output Handling



Even with the half-plane method, a concave polygon can yield a `MultiPolygon` child if the split line enters and exits the polygon boundary more than once. This happens with U shapes split by a horizontal line that crosses both arms without spanning the connecting base.

The handling protocol:

```python
from shapely import get\_parts

def extract\_largest\_polygon(geom) -> Polygon:
    if geom.geom\_type == "Polygon":
        return geom
    parts = \[g for g in get\_parts(geom) if g.geom\_type == "Polygon"]
    if not parts:
        raise SplitFailure("No polygon component in geometry")
    return max(parts, key=lambda p: p.area)
```

After extracting the largest polygon from each child, re-validate against the minimum area guard. Area lost to the extraction is acceptable — it represents the thin connecting strip between the two largest sub-regions, which would have generated a degenerate room in any case.

###### &#x20;

##### 3.5 Split-Axis Selection via PCA

###### 

###### 3.5.1 The Rectangular Case



For axis-aligned rectangular cells, the correct split axis is trivially the longer of the two bounding-box dimensions. The squarified treemap algorithm chooses a horizontal subdivision when the rectangle is wider than it is tall. This is the degenerate case of a more general principle: split perpendicular to the longest extent.

For rectangular cells in the generator, the PCA method (below) correctly recovers this axis-aligned choice because the vertex distribution of a rectangle has its principal axis aligned with the longer bounding-box dimension. The implementation therefore uses PCA uniformly for all cell geometries, including rectangles.

###### 

###### 3.5.2 PCA on Polygon Vertices



For a non-rectangular cell polygon, the "longest extent" is not the bounding-box axis — it is the direction of maximum spatial spread of the polygon's vertices. PCA is a linear dimensionality reduction technique that linearly transforms data onto a new coordinate system such that the directions capturing the largest variation in the data can be easily identified. Applied to 2D polygon vertices, the first principal component identifies the axis of maximum vertex spread, which is the correct split axis for producing compact, roughly-square children.

The computation:

```python
import numpy as np
from shapely.geometry import Polygon

def pca\_split\_axis(polygon: Polygon) -> np.ndarray:
    """
    Return the unit normal to the recommended split plane.
    The split is perpendicular to PC1 (i.e., along PC2).
    """
    coords = np.array(polygon.exterior.coords\[:-1])  # exclude closing vertex
    
    # Densify for high-vertex-count STAIR primitives: take every 3rd vertex
    if len(coords) > 16:
        coords = coords\[::3]
    
    # Centre and compute covariance
    centred = coords - coords.mean(axis=0)
    cov = np.cov(centred.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # PC1 is the eigenvector with the largest eigenvalue
    pc1 = eigenvectors\[:, np.argmax(eigenvalues)]
    
    # The split line runs along PC1; the split normal is perpendicular
    normal = np.array(\[-pc1\[1], pc1\[0]])
    return normal / np.linalg.norm(normal)
```

PCA is defined as an orthogonal linear transformation that transforms data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate. By splitting *perpendicular* to PC1, we cut the polygon across its widest extent, producing two children that are each narrower than the parent — exactly the behaviour needed to drive aspect ratio toward 1.0.

###### 

###### 3.5.3 Split Position Sampling



The split position along the normal axis is not the centroid. The generator samples:

```python
split\_t = np.random.uniform(0.35, 0.65)
```

where `split\_t = 0.5` is the midpoint along the bounding extent in the normal direction. This range \[0.35, 0.65] produces rooms of varied size while preventing extremely unequal children. An additional **area-target bias** is applied: if the programme requires a next room of relative area `r`, the split position is nudged toward `r` by:

```python
split\_t = np.clip(split\_t \* 0.7 + r \* 0.3, 0.30, 0.70)
```

This creates a soft correspondence between programme areas and cell areas without hard-constraining the geometry.

###### 

###### 3.5.4 Axis Jitter for Visual Variety



Real floor plans rarely have walls that are perfectly axis-aligned across the entire plan. After PCA selects the principal axis, a small angular jitter is applied:

```python
jitter\_deg = np.random.normal(0, 3.5)  # σ = 3.5°
normal = rotate\_vector(normal, np.deg2rad(jitter\_deg))
```

This jitter is independent of the diagonal wall system in Chapter 4. It operates within BSP subdivision and produces very slightly non-orthogonal room boundaries that are still rendered as axis-aligned walls in Chapter 5 (the wall graph snaps to the nearest axis). The jitter's visual effect is in room proportions, not wall angles.

&#x20;

##### 3.6 Recursion Guards



The BSP recursion terminates at any node where any of the following conditions fail. These are evaluated on the **candidate child**, not the parent, immediately after the split.

###### 

###### Guard 1 — Minimum Area



```python
MIN\_ROOM\_AREA\_PX2 = (MIN\_INTERIOR\_WIDTH\_PX \*\* 2) \* 1.5  # default: \~9600 px² at 512×512
assert child.area >= MIN\_ROOM\_AREA\_PX2
```

A cell that is too small to contain a valid room (Chapter 6's opening placement requires at least `MIN\_INTERIOR\_WIDTH\_PX` clearance on both axes) is terminated immediately. This is the most frequently triggered guard at high room counts.

###### 

###### Guard 2 — Aspect Ratio



The aspect ratio of a polygon is approximated by the ratio of the two eigenvalues from the PCA computation (reused from the axis-selection step):

```python
aspect = eigenvalues\[np.argmax(eigenvalues)] / (eigenvalues\[np.argmin(eigenvalues)] + 1e-6)
MAX\_ASPECT\_RATIO = 4.5
assert aspect <= MAX\_ASPECT\_RATIO
```

Rectangular treemaps have the disadvantage that their aspect ratio might be arbitrarily high in the worst case. To cope with this problem, several algorithms have been proposed that use regions that are general convex polygons, not necessarily rectangular. The eigenvalue ratio is a geometry-native aspect measure that correctly identifies elongated cells even when they are non-rectangular (e.g., a thin diagonal parallelogram).

###### 

###### Guard 3 — Minimum Interior Width



The minimum interior width check from Chapter 2's validator is reused verbatim. A cell that passes the area guard but is extremely narrow (e.g., a thin L-arm) will fail this check. The threshold is the same `MIN\_INTERIOR\_WIDTH\_PX`.

###### 

###### Guard 4 — Target Room Count Reached



```python
if len(leaf\_cells) >= target\_room\_count:
    break  # do not recurse further, regardless of cell size
```

`target\_room\_count` is sampled once per image from the programme (§3.3). This prevents overly deep trees from being generated for images where the programme calls for few rooms.

###### 

###### Fallback Behaviour



When a split fails all attempted axis/position combinations (the algorithm tries up to three PCA-derived axes before giving up), the cell is marked as a **terminal leaf** and becomes a room at whatever size it currently has. It is not discarded. This guarantees that the tree always produces a valid covering of the footprint, even if some rooms are larger than the programme target.

&#x20;

##### 3.7 Treemap Layout Within BSP Leaf Cells



After BSP partitioning completes, each leaf cell contains one room polygon. For large leaf cells (area > 2 × the programme's target room area), a second subdivision pass using a simplified qualified treemap is applied to fill the cell with multiple rooms.

###### 

###### 3.7.1 Squarified Treemap Adaptation



The main shortcoming of the classic treemap is that it produces thin, elongated rectangles. The squarified treemap algorithm attempts to produce more square-like rectangles by ensuring that the aspect ratios of the rectangles are as close to 1 as possible. In the original algorithm, this is achieved by greedily adding items to a row along the container's shorter side and committing the row when adding the next item would worsen the maximum aspect ratio. Instead of looking for the optimal solution, which would require finding all possible tessellations, the algorithm returns a good solution that can be computed in a short amount of time.

The adaptation from rectangles to arbitrary polygons proceeds as follows:

1. Compute the **oriented bounding box** of the leaf cell (the minimum-area rectangle enclosing it).
2. Run the squarified treemap algorithm on the oriented bounding box, treating it as the container.
3. **Intersect** each squarified sub-rectangle with the original leaf cell polygon: `room\_poly = sub\_rect.intersection(leaf\_cell)`.
4. Discard any intersection result with area < `MIN\_ROOM\_AREA\_PX2`.
5. The residual area (parts of the leaf cell not covered by any sub-rectangle after step 4) is merged with the largest adjacent room.

This approach avoids directly solving the hard problem of squarified treemapping on non-rectangular containers. Alternative treemap methods have been developed using circular or polygonal shapes, achieving a fit for non-rectangular canvases and providing a more pleasing way to present data, but these introduce significant implementation complexity. The intersection approach is a practical approximation: it produces somewhat non-square rooms near the cell boundary but is O(n) in the number of rooms and requires no specialised geometry library.

###### 

###### 3.7.2 Area Normalisation



The order in which rectangles are processed is important in the squarified algorithm. A decreasing order usually gives the best results. Room areas are sorted in descending order before being fed to the treemap. This ensures that the largest room (typically the living room) is placed first and occupies the most favourable region of the cell, with smaller rooms (bathrooms, utilities) filling the residual space.

###### 

###### 3.7.3 Treemap vs. BSP Selection



Not all leaf cells receive the treemap pass. The decision is made per-cell:

```python
USE\_TREEMAP = (
    cell.area > 2.0 \* target\_room\_area
    and remaining\_rooms\_to\_place >= 2
    and cell\_aspect\_ratio <= 3.0
)
```

Cells with aspect ratio above 3.0 receive additional BSP splits rather than treemap packing, because the squarified algorithm degenerates on highly elongated containers (it produces the thin-rectangle problem it was designed to prevent). The squarified algorithm is successful in the sense that rectangles are far less elongated and the black areas with cluttered rectangles disappear — but only when the container itself is reasonably proportioned.

&#x20;

##### 3.8 Vertex Simplification Before PCA



Chapter 2's STAIR primitive can produce cells with 16–24 vertices after BSP splits, because each split adds intersection vertices along the split line. High vertex counts do not harm correctness but do skew the PCA computation: the near-collinear step vertices of a staircase shape have a strongly anisotropic distribution that makes the PCA axis appear more diagonal than it is.

Before each PCA computation, apply:

```python
simplified = polygon.simplify(SIMPLIFY\_TOLERANCE, preserve\_topology=True)
```

where `SIMPLIFY\_TOLERANCE = 2.0` pixels. This removes near-collinear vertices while preserving the topology of the cell. After simplification, re-validate with `is\_valid` — as noted in Chapter 2, simplification can invalidate polygons with holes, though in the no-hole context of BSP leaf cells this is rare. Nevertheless, apply `make\_valid()` defensively.

**Critical**: simplification is applied only for PCA input, never to the cell polygon itself. The unsimplified polygon is always used for intersection/difference operations to avoid the Shapely issue where simplifying a Shapely polygon may leave holes outside the polygon, making the object invalid — even with `preserve\_topology=True`.

&#x20;

##### 3.9 The Room Dataclass



Each BSP/treemap leaf produces one `Room` instance:

```python
@dataclass
class Room:
    polygon: shapely.Polygon        # final room shape, pixel-space, integer vertices
    room\_type: str                  # from ROOM\_TYPE\_WEIGHTS keys
    target\_area: float              # programme target, for metadata
    actual\_area: float              # polygon.area
    area\_error: float               # abs(actual - target) / target
    cell\_depth: int                 # depth in BSP tree (0 = root)
    is\_treemap\_child: bool          # True if produced by treemap pass
    adjacency: list\[int]            # indices of adjacent rooms (shared wall edge)
    exterior\_edges: list\[LineString] # portions of polygon boundary that touch footprint exterior
```

The `exterior\_edges` field is pre-computed by intersecting the room polygon boundary with the footprint exterior ring:

```python
room\_boundary = room.polygon.boundary
footprint\_boundary = footprint.polygon.exterior
exterior\_edges = room\_boundary.intersection(footprint\_boundary)
```

This is used by Chapter 6 to determine which room walls are eligible for windows and doors. Computing it here, at subdivision time, avoids redundant geometry operations later.

The `adjacency` list is built after all rooms are finalised, by testing every pair of rooms for shared boundary length above a minimum threshold:

```python
MIN\_SHARED\_EDGE\_PX = 12  # shared boundary must be at least 12 px to count as adjacent
for i, room\_a in enumerate(rooms):
    for j, room\_b in enumerate(rooms\[i+1:], i+1):
        shared = room\_a.polygon.boundary.intersection(room\_b.polygon.boundary)
        if shared.length >= MIN\_SHARED\_EDGE\_PX:
            room\_a.adjacency.append(j)
            room\_b.adjacency.append(i)
```

Adjacency is required by Chapter 6's door placement algorithm, which places doors only on shared walls between adjacent rooms.

&#x20;

##### 3.10 Output Contract



The subdivider returns:

```python
@dataclass
class SubdivisionResult:
    rooms: list\[Room]               # complete list, covers footprint without overlap
    bsp\_tree: BSPNode               # root node, for debugging and metadata
    room\_count: int                 # len(rooms)
    treemap\_room\_count: int         # rooms produced by treemap pass
    max\_area\_error: float           # worst-case area\_error across all rooms
    coverage\_fraction: float        # sum(room.area) / footprint.area (should be ≥ 0.97)
```

The `coverage\_fraction` should be ≥ 0.97. Values below 0.97 indicate that residual slivers between rooms were too small to merge and were discarded. If `coverage\_fraction < 0.90`, the subdivision is flagged in metadata and the sample is regenerated with a different seed. This threshold is intentionally permissive: a 3–10% coverage gap produces thin hairline voids at room junctions, which are invisible after wall rendering (Chapter 5) and do not affect mask quality (Chapter 11).

All `Room.polygon` objects are non-overlapping by construction — intersection and difference operations are exact in GEOS, so two children of the same BSP split share exactly zero interior area. Floating-point snapping at pixel rounding (§2.4, Stage 5) can introduce 1-pixel overlaps at room boundaries; Chapter 11's write-order protocol resolves these by rendering walls last, overwriting any such artefact.

&#x20;

##### 3.11 Relationship to Downstream Chapters



|Chapter|Dependency on subdivision output|
|-|-|
|**4**|Diagonal walls attach to room edges; `exterior\_edges` identifies which are on the building facade|
|**5**|Wall graph is initialised from all shared room boundaries in `adjacency` list|
|**6**|Window/door placement uses `room\_type`, `exterior\_edges`, and `adjacency`|
|**7**|Icon subcategory selection and placement bounding box both use `room\_type` and `Room.polygon`|
|**8**|Dimension arrows span the longest interior dimension of each room polygon|
|**11**|Room polygons define the interior of each class 0 region; wall polygons are derived from shared edges|



The `SubdivisionResult` object is passed by reference through the entire pipeline. No downstream chapter modifies room polygons in place. Any chapter that requires a modified version of a room boundary (e.g., the wall-inset boundary used for icon placement in Chapter 7) computes it locally from `Room.polygon` using `polygon.buffer(-WALL\_HALF\_THICKNESS)`.



##### &#x20;

#### Chapter 4 — Diagonal Walls

##### 

##### 4.1 Why Diagonal Walls Matter



Axis-aligned floor plans are visually homogeneous. Every wall runs at 0° or 90°, every room is a rectangle or near-rectangle, and the resulting corpus produces a model with a strong inductive bias toward orthogonality. Real residential floor plans — particularly Soviet-era panel blocks, late-modernist designs, and irregular infill plots — regularly include diagonal walls at facade corners, chamfered internal junctions, and occasionally as full interior partition walls. A model trained without any diagonal geometry will misclassify these features, typically merging them with the background class or confusing them with furniture.

Diagonal walls are also the single most technically demanding wall feature to generate correctly. Three problems compound:

1. **Angle sampling** must produce a distribution that matches real-world prevalence — 30–60° is far more common than near-0° or near-90°.
2. **Thickness** cannot be computed from axis-aligned bounding boxes; it requires the perpendicular-normal method.
3. **Miter junctions** — where a diagonal wall meets an axis-aligned wall — cannot be drawn as intersecting stroked lines without producing gaps or overlaps at the join. They require explicit polygon union.

This chapter addresses all three problems. The output is a set of wall polygons (thick filled bands) that integrate cleanly with the wall graph in Chapter 5 and are rasterised directly to the mask in Chapter 11 without any line-drawing step.

&#x20;

##### 4.2 Coordinate and Angular Conventions



All angles in this chapter are measured as **deviation from the horizontal axis** in pixel space (y-downward), in degrees, in the range \[0°, 90°). A wall at 0° is horizontal. A wall at 90° is vertical. A wall at 45° runs from top-left to bottom-right.

Angles outside \[15°, 75°] are excluded entirely. A wall at 5° is nearly horizontal and visually indistinguishable from a slightly skewed axis-aligned wall after scan degradation (Chapter 10). A wall at 88° is nearly vertical and has the same problem. Including such angles would add near-duplicate geometry that does not contribute to model generalisation and introduces miter junction edge cases that are not worth solving for marginal visual diversity.

The effective range is therefore \[15°, 75°], with a peaked distribution toward \[30°, 60°].

&#x20;

##### 4.3 Angle Sampling

###### 

###### 4.3.1 The Sampling Distribution



The angle θ for a diagonal wall is drawn from a **mixture of a truncated normal and a uniform component**:

```python
def sample\_diagonal\_angle(rng: np.random.Generator) -> float:
    """
    Returns angle in degrees in \[15, 75].
    80% of mass from truncated normal peaked at 45°, σ=12°.
    20% of mass uniform across \[15, 75].
    """
    if rng.uniform() < 0.80:
        while True:
            θ = rng.normal(loc=45.0, scale=12.0)
            if 15.0 <= θ <= 75.0:
                return θ
    else:
        return rng.uniform(15.0, 75.0)
```

The truncated normal peaked at 45° reflects the real-world prevalence of 45° chamfers and corner bevels in architectural practice. The 20% uniform component prevents the model from learning that diagonal walls are always near-45°, which would cause it to miss shallower or steeper diagonals in real drawings.

### 

###### 4.3.2 Angle Snapping for Soviet-Era Samples



When the era is `soviet`, a **snap grid** of 15° is applied after sampling:

```python
if era == "soviet":
    θ = round(θ / 15.0) \* 15.0
    θ = np.clip(θ, 15.0, 75.0)
```

Soviet technical drafting conventions (BTI standards) favoured clean angular increments — 30°, 45°, 60° — because hand-drawn diagonals were typically constructed with a set square. Free-angle diagonals, common in CAD-era drawings, are era-inappropriate for Soviet samples and would introduce anachronistic visual signals.

For `digital` and `scan` era samples, no snapping is applied. The sampled angle is used directly, potentially producing non-standard values like 37° or 52°.

&#x20;

##### 4.4 Placement Strategy



Diagonal walls are placed according to a **80/20 split** between external and internal placement. This ratio reflects real-world distribution: the overwhelming majority of diagonal walls in residential floor plans occur at exterior corners and chamfered facades, with internal diagonal partitions being comparatively rare.

###### 

###### 4.4.1 External Placement (80%)



External diagonal walls are attached to the **building footprint boundary** from Chapter 2. There are two external placement sub-modes:

**Sub-mode A — Corner Bevel (60% of external placements)**

A corner bevel replaces a convex exterior corner of the footprint polygon with a diagonal cut. The footprint's bevelled-corner vertices (recorded in `Footprint.bevel\_corners` from Chapter 2) are the primary candidates. If `Footprint.bevel\_corners` is empty (the primitive was not `BEVEL` and aggression did not trigger Stage 4 chamfering), corner bevel placement samples a random convex exterior corner and applies a fresh chamfer.

The bevel diagonal is defined by two points `p1` and `p2` along the two edges adjacent to the target corner:

```python
def compute\_bevel\_endpoints(corner: np.ndarray,
                             prev\_vtx: np.ndarray,
                             next\_vtx: np.ndarray,
                             bevel\_depth: float) -> tuple\[np.ndarray, np.ndarray]:
    """
    bevel\_depth: distance from corner along each adjacent edge, in pixels.
    """
    edge1\_unit = normalise(prev\_vtx - corner)
    edge2\_unit = normalise(next\_vtx - corner)
    p1 = corner + bevel\_depth \* edge1\_unit
    p2 = corner + bevel\_depth \* edge2\_unit
    return p1, p2
```

`bevel\_depth` is sampled from `Uniform(12, 40)` pixels, keeping bevels architecturally plausible in size relative to a typical room. The wall centreline runs from `p1` to `p2`. Wall thickness is computed by the perpendicular-normal method (§4.5).

**Sub-mode B — Chamfered Facade Segment (40% of external placements)**

Rather than replacing a corner, a chamfered facade segment replaces a short section of an exterior wall edge with a diagonal inset. This models angled bay windows, recessed entrances, and irregular facade treatments common in modernist Soviet blocks.

The target edge is selected from exterior edges longer than `3 × MIN\_INTERIOR\_WIDTH\_PX`. A random sub-segment of length `Uniform(0.15, 0.35)` × edge length is chosen, and its midpoint is offset inward by `Uniform(8, 24)` pixels, producing a three-segment replacement: two short connecting segments and one diagonal centre segment. Only the diagonal centre segment is rendered as a diagonal wall; the connecting segments are folded into the main wall system as short axis-aligned walls in Chapter 5.

This sub-mode modifies the footprint polygon:

```python
updated\_footprint = footprint.polygon.difference(inset\_triangle)
updated\_footprint = make\_valid(updated\_footprint)
```

The updated footprint is stored back into the `Footprint` object. All downstream chapters receive the updated version. This is the **only chapter** permitted to modify the `Footprint.polygon` after Chapter 2 produces it — and only for chamfered facade segments, not for corner bevels (which were already baked into the footprint in Chapter 2).

### 

###### 4.4.2 Internal Placement (20%)



Internal diagonal walls run through the interior of the floor plan, forming partition walls between rooms at a non-orthogonal angle. They are placed entirely within the `SubdivisionResult` from Chapter 3.

Internal diagonal placement algorithm:

1. Select a **target room** from `SubdivisionResult.rooms` with area above `2.5 × MIN\_ROOM\_AREA\_PX2`. Small rooms cannot accommodate a diagonal partition without violating the minimum interior width guard.
2. Sample an angle θ (§4.3).
3. Sample a **placement line**: pass the diagonal through a point `q` sampled uniformly within the target room's `polygon.buffer(-MIN\_INTERIOR\_WIDTH\_PX)` (the inset polygon, ensuring clearance from room walls).
4. Clip the diagonal line to the room polygon boundary using `room.polygon.boundary.intersection(diagonal\_line)` to obtain endpoints `p1`, `p2`.
5. Validate that the resulting diagonal segment has length ≥ `2 × MIN\_INTERIOR\_WIDTH\_PX`. Reject and resample if not.
6. Check that neither sub-room produced by the diagonal falls below `MIN\_ROOM\_AREA\_PX2`. Reject if either does.
7. Commit the diagonal and update the room's polygon by splitting it with the half-plane method from Chapter 3 §3.4.2, producing two child rooms. Both children inherit the parent's `room\_type`.

At most **one internal diagonal wall** is placed per floor plan sample, regardless of room count. Multiple internal diagonals create compound miter junctions (diagonal meets diagonal) whose geometry is beyond the scope of this chapter and whose real-world prevalence does not justify the implementation cost.

###### 

###### 4.4.3 Per-Image Diagonal Wall Count



The total number of diagonal walls per image is sampled as:

```python
n\_diagonals = rng.choice(\[0, 1, 2, 3], p=\[0.30, 0.40, 0.20, 0.10])
```

30% of images have no diagonal walls at all. This is intentional: the model must not learn that diagonal geometry is always present, and a training set with 100% diagonal wall prevalence would cause the model to hallucinate diagonal walls in clean orthogonal real drawings.

When `n\_diagonals > 1`, at most one may be internal (§4.4.2). The remainder are external.

&#x20;

##### 4.5 Perpendicular-Normal Method for Wall Thickness

###### 

###### 4.5.1 Why Bounding-Box Thickness Fails



The naive approach to producing a thick wall polygon from a centreline segment is to offset the segment's bounding box — expand it by half the wall thickness in the x and y directions. This works for axis-aligned walls because their edges are parallel to the coordinate axes. For a diagonal wall, it fails catastrophically: the bounding-box expansion produces a rectangular region that is neither centred on the wall nor of the correct width, with the error growing as the wall angle approaches 45°.

At 45°, a bounding-box expansion by `t` pixels produces a wall polygon that is `t√2` pixels wide measured perpendicular to the wall centreline — 41% too thick. The mask would be wrong and wall junctions would not align.

###### 

###### 4.5.2 The Perpendicular-Normal Method



The correct approach is to offset the centreline segment perpendicular to its own direction. Given centreline endpoints `p1` and `p2`:

```python
def wall\_polygon\_from\_centreline(p1: np.ndarray,
                                  p2: np.ndarray,
                                  thickness: float) -> Polygon:
    """
    Returns a parallelogram-shaped wall polygon of the specified
    perpendicular thickness around the centreline p1→p2.
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        raise ValueError("Degenerate centreline: p1 == p2")

    unit = direction / length
    # Perpendicular normal (rotated 90° CCW in y-down pixel space)
    normal = np.array(\[-unit\[1], unit\[0]])

    half\_t = thickness / 2.0
    # Four corners of the wall polygon
    c1 = p1 + half\_t \* normal
    c2 = p1 - half\_t \* normal
    c3 = p2 - half\_t \* normal
    c4 = p2 + half\_t \* normal

    return Polygon(\[c1, c2, c3, c4])
```

This produces a **parallelogram** (in general) or rectangle (when the wall is axis-aligned) whose width measured perpendicular to its own centreline is exactly `thickness`, regardless of angle. The `normal` vector here is the perpendicular to the wall direction in y-downward pixel space.

###### 

###### 4.5.3 End-Cap Treatment



The raw perpendicular-normal polygon has blunt (90°) ends. For most wall types this is correct — the end is covered by the junction polygon (§4.6). For **isolated diagonal wall segments** that terminate at the footprint boundary without meeting another wall, a small end-cap extension is applied:

```python
cap\_extension = thickness \* 0.5
p1\_extended = p1 - cap\_extension \* unit
p2\_extended = p2 + cap\_extension \* unit
```

The extension ensures that the diagonal wall polygon fully covers any sub-pixel gap between the wall end and the footprint boundary when both are rasterised independently. Without this, a 1–2 pixel crack can appear at the wall terminus in the mask, which would be classified as background (class 0) by the mask writer.

###### 

###### 4.5.4 Thickness Sampling for Diagonal Walls



Diagonal wall thickness is drawn from the same distribution as exterior structural walls in Chapter 5 — typically `Uniform(6, 14)` pixels at 512×512 — but is **not jittered independently along the wall length**. Per-segment thickness jitter (Chapter 5's hollow-wall and hand-drawn effects) does not apply to diagonal walls because their short lengths make segment-level jitter visually implausible. A single thickness value is sampled once per diagonal wall and applied uniformly along the entire centreline.

&#x20;

##### 4.6 Miter Handling at Junctions

###### 

###### 4.6.1 The Line-Drawing Problem



If walls were drawn as stroked lines rather than filled polygons, a miter junction — where a diagonal wall centreline meets an axis-aligned wall centreline — would be handled by the stroke renderer's miter join mode. SVG, Cairo, and most 2D graphics libraries support this natively.

This generator does not draw stroked lines. All walls are filled polygons written directly to the mask canvas (Chapter 11's write-order protocol requires polygon-fill, never stroke-drawing, for mask integrity). The miter must therefore be resolved geometrically, as an explicit polygon operation, before anything is rasterised.

###### 

###### 4.6.2 Why `unary\_union` Is the Correct Primitive



The naive polygon approach — place the diagonal wall polygon and the axis-aligned wall polygon as separate filled regions — leaves one of three artefacts at the junction:

* **Gap**: if the two polygons do not overlap, a gap of background pixels appears at the join point.
* **Class bleed**: if the two polygons overlap but are drawn independently, the overlap region is written twice, with the second write overwriting the first. If the two writes are of the same class this is harmless — but any intermediate mask state is incorrect.
* **Incorrect miter geometry**: even if the two polygons touch exactly at a shared edge, the corner region between them is not filled, producing a triangular void at the junction.

The solution is to compute the **union of all wall polygons that meet at a junction before any rasterisation**:

```python
from shapely.ops import unary\_union

junction\_walls = \[diagonal\_wall\_polygon, axial\_wall\_polygon\_A, axial\_wall\_polygon\_B]
merged\_junction = unary\_union(junction\_walls)
merged\_junction = make\_valid(merged\_junction)
```

`unary\_union` computes the boolean union of an arbitrary collection of polygons in a single GEOS operation. The result is a single `Polygon` (or `MultiPolygon` if the inputs are not connected) whose boundary correctly traces the outer edge of all input polygons, including the filled-in junction region. No gaps, no overlaps, no class bleed.

The result is what is rasterised to the mask. The original per-wall polygons are retained in the wall graph (Chapter 5) for metadata and adjacency purposes but are **never rasterised individually**.

###### 

###### 4.6.3 Junction Classification



Every junction between a diagonal wall and one or more axis-aligned walls is classified into one of four types before `unary\_union` is applied. This classification matches the junction typing in Chapter 5's wall graph and ensures that the merged polygon receives the correct entry in the wall graph's `junction\_type` field:



|Type|Description|Geometry|
|-|-|-|
|`DIAG\_T`|Diagonal wall meets one axis-aligned wall mid-segment|T-junction; diagonal terminates, axial continues|
|`DIAG\_L`|Diagonal wall meets one axis-aligned wall at its end|L-junction; both walls terminate at the meet point|
|`DIAG\_X`|Diagonal wall crosses an axis-aligned wall|X-junction; both walls continue past the intersection|
|`DIAG\_CORNER`|Diagonal bevel between two axis-aligned walls|Corner replacement; diagonal terminates at both ends into axials|



`DIAG\_CORNER` is by far the most common type (it covers all corner-bevel external placements from §4.4.1 Sub-mode A). The `unary\_union` of a `DIAG\_CORNER` junction produces a clean filled polygon with no interior voids, because the three input polygons — the diagonal and the two adjacent wall stubs — form a simply-connected region.

`DIAG\_X` is the most geometrically complex. The merged polygon has a cruciform shape. Before passing it to `unary\_union`, validate that all four arms of the cross extend at least `MIN\_INTERIOR\_WIDTH\_PX / 2` pixels beyond the intersection point, otherwise the junction degenerates into a near-point and the cross arms are effectively invisible.

###### 

###### 4.6.4 Miter Angle Limit



When a diagonal wall meets an axis-aligned wall at an acute angle — specifically when the miter angle (the angle between the two wall centrelines) is less than 20° — the miter polygon becomes an extreme sliver with a very long pointed tip. This tip will:

* Extend far beyond the visual extent of the wall
* Produce single-pixel mask artefacts at the miter tip that the antialiasing step cannot fix
* Confuse Chapter 11's polygon-fill rasteriser

The miter angle limit is enforced in the junction validator:

```python
miter\_angle = angle\_between(diagonal\_centreline\_direction, axial\_wall\_direction)
miter\_angle = min(miter\_angle, 180.0 - miter\_angle)  # take the acute version
if miter\_angle < 20.0:
    raise DiagonalPlacementFailure("Miter angle too acute")
```

This failure propagates to the placement loop, which resamples the diagonal angle (§4.3) and retries. Since angles are drawn from \[15°, 75°] and most axis-aligned walls are at 0° or 90°, miter angles below 20° can only arise when a 15°–20° diagonal meets a 0° wall or a 70°–75° diagonal meets a 90° wall. The truncated normal distribution (§4.3.1) makes this rare but not impossible; the validator catches the cases that slip through.

&#x20;

##### 4.7 Interaction With the No-Opening Constraint



Chapter 6 places windows and doors in axis-aligned walls only. Diagonal walls receive no openings. This constraint is enforced at the wall graph level in Chapter 5 (each wall segment carries a `is\_diagonal` flag that Chapter 6 checks before placement), but its architectural motivation belongs here.

The constraint exists for two reasons:

**Geometric reason**: opening placement in Chapter 6 is implemented using axis-aligned bounding-box operations — a window is a rectangle cut out of a wall segment's bounding box. For a diagonal wall, this approach produces incorrectly oriented openings whose mask representation would require the same perpendicular-normal method used for the wall itself. The additional complexity is not justified by the training value.

**Prevalence reason**: diagonal wall openings are uncommon in the real-world corpus this model targets. Soviet-era panel blocks with chamfered facade corners typically have solid diagonal surfaces at those corners — the openings are in the adjacent axis-aligned facades. Including diagonal-wall openings in the synthetic corpus would teach the model to expect features that rarely appear in real drawings, harming generalisation.

The `is\_diagonal` flag is set to `True` for all wall polygons produced by this chapter, including the merged `unary\_union` junction polygons. Chapter 6 will not place any opening whose host wall segment has `is\_diagonal = True`.

&#x20;

##### 4.8 Diagonal Wall Dataclass



Each committed diagonal wall produces one `DiagonalWall` instance, which is added to the wall graph in Chapter 5:

```python
@dataclass
class DiagonalWall:
    centreline: shapely.LineString      # p1 → p2, pixel space
    wall\_polygon: shapely.Polygon       # perpendicular-normal polygon, pre-union
    merged\_polygon: shapely.Polygon     # unary\_union with junction neighbours
    angle\_deg: float                    # sampled angle in \[15, 75]
    thickness: float                    # pixels
    placement\_mode: str                 # "corner\_bevel" | "facade\_chamfer" | "internal"
    junction\_types: list\[str]           # one entry per endpoint: DIAG\_T/L/X/CORNER
    is\_diagonal: bool = True            # always True; consumed by Ch. 6
    era: str                            # inherited from generator era token
```

The `merged\_polygon` field is what gets written to the mask in Chapter 11. The `wall\_polygon` field is retained for wall graph construction in Chapter 5 and for any per-wall style decisions in Chapter 9.

&#x20;

##### 4.9 Implementation Notes and Known Edge Cases



**Near-collinear diagonal and footprint edge.** When a diagonal wall centreline is nearly parallel to an exterior footprint edge (angular difference < 5°), the half-plane intersection used to clip the diagonal to the room interior can produce a very thin sliver rather than a clean endpoint. Detect this before clipping:

```python
edge\_angle = angle\_of\_footprint\_edge(nearest\_exterior\_edge)
if abs(θ - edge\_angle) < 5.0 or abs((θ + 90) % 180 - edge\_angle) < 5.0:
    raise DiagonalPlacementFailure("Diagonal nearly parallel to footprint edge")
```

**STAIR primitive with step-profile edges.** The STAIR footprint has multiple short horizontal and vertical edges forming step risers and treads. A corner-bevel diagonal applied to a step corner produces a very short diagonal (sometimes < 15 px) that fails the minimum length check. Suppress corner-bevel placement on STAIR step corners; only chamfered-facade placement is permitted on STAIR primitives.

**`unary\_union` returning `MultiPolygon` for `DIAG\_X` junctions.** If the four arms of a cross junction are not all topologically connected — which can happen if the diagonal wall polygon and the axis-aligned wall polygon share only a single point rather than an overlapping region — `unary\_union` returns a `MultiPolygon`. Force an overlap by expanding each wall polygon by 0.5 pixels (`polygon.buffer(0.5)`) before the union, then contracting the result by 0.5 pixels afterwards. The 0.5-pixel buffer guarantees at least a shared area rather than a shared point:

```python
buffered = \[p.buffer(0.5) for p in junction\_walls]
merged = unary\_union(buffered).buffer(-0.5)
merged = make\_valid(merged)
```

**Floating-point vertex accumulation.** Each chapter that calls `difference()`, `intersection()`, or `union()` on a polygon adds new vertices at intersection points. By the time a diagonal wall polygon has been through the placement clip, the bevel endpoint computation, and the `unary\_union` merge, it may have accumulated 30+ vertices. This does not affect correctness but does slow Chapter 11's rasteriser. Apply `polygon.simplify(0.5, preserve\_topology=True)` to `merged\_polygon` before storing it in the `DiagonalWall` dataclass, and always after `make\_valid()`.

**Performance budget.** The diagonal wall placement pipeline (angle sampling + perpendicular-normal construction + junction classification + `unary\_union`) runs in under 2 ms per diagonal wall on a single core. At `n\_diagonals = 3`, the worst-case contribution to per-sample latency is approximately 6 ms, within the Chapter 12 budget.

&#x20;

##### 4.10 Relationship to Downstream Chapters

&#x20;

|Chapter|Dependency on diagonal wall output|
|-|-|
|**5**|`DiagonalWall` instances are inserted into the wall graph; `is\_diagonal` flag disables opening eligibility|
|**6**|Reads `is\_diagonal`; skips all opening placement for diagonal segments|
|**9**|`DiagonalWall.era` controls hand-drawn jitter style; Soviet samples apply heavier line imperfection to `wall\_polygon.exterior`|
|**10**|No special handling; diagonal walls are already rasterised polygons before degradation is applied|
|**11**|`merged\_polygon` is the rasterisation target; polygon-fill to class 1, write-order places it after room interiors and before openings|



I now have comprehensive research across all the necessary dimensions. Let me write the full chapter.

&#x20;



#### Chapter 5 — Wall Type \& Thickness System

##### 

##### 5.1 The Wall Graph: Central Data Structure



Every chapter from 2 onward has produced geometry — a footprint polygon, a set of room polygons, a set of diagonal wall objects. Chapter 5's first act is to collapse all of that geometry into a single unified data structure: the **wall graph**. Everything downstream — opening placement (Chapter 6), aesthetic rendering (Chapter 9), and mask writing (Chapter 11) — reads from the wall graph. Nothing writes to room polygons or footprint edges directly after this chapter.

Vector floor plans have a multi-level structure: spatially, 0D wall junctions form 1D wall segments, which close into 2D rooms. The wall graph formalises exactly this hierarchy. Wall junctions are abstracted as nodes, and wall segments as edges. In this generator, every node carries a position and a junction type; every edge carries thickness, draw style, and a flag indicating whether it is interior or exterior.

The wall graph is an **undirected planar graph** with the following properties:

* **Nodes** are wall junction points — places where two or more wall centrelines meet, or where a wall terminates at an exterior boundary.
* **Edges** are wall segment centrelines connecting two nodes. Each edge corresponds to one rendered wall polygon.
* The graph must be connected and planar (no two edges cross except at shared nodes).
* The graph must not contain isolated nodes (a junction with degree zero).

Wall structure is represented by a set of junctions where wall segments meet. There are four wall junction types: I-, L-, T-, and X-shaped, depending on the degrees of incident wall segments. This generator uses these same four types as the primary junction classification, extended with the diagonal-specific types defined in Chapter 4.

###### &#x20;

##### 5.2 Building the Wall Graph from Prior Chapters



The wall graph is constructed in four sequential passes over the outputs of Chapters 2, 3, and 4.

###### 

###### Pass 1 — Exterior Ring



The footprint's exterior ring (a `LinearRing` from `Footprint.polygon.exterior`) is walked vertex by vertex. Each vertex becomes a wall graph node. Each consecutive pair of vertices becomes a wall graph edge. All edges added in this pass are flagged `is\_exterior = True`.

```python
ring\_coords = list(footprint.polygon.exterior.coords\[:-1])  # drop closing duplicate
for i, coord in enumerate(ring\_coords):
    next\_coord = ring\_coords\[(i + 1) % len(ring\_coords)]
    node\_a = graph.add\_node(coord, on\_exterior=True)
    node\_b = graph.add\_node(next\_coord, on\_exterior=True)
    graph.add\_edge(node\_a, node\_b,
                   is\_exterior=True,
                   is\_diagonal=False,
                   thickness=None)  # assigned in §5.4
```

###### 

###### Pass 2 — Interior Room Boundaries



For each pair of adjacent rooms in `SubdivisionResult` (i.e., pairs with a shared edge length ≥ `MIN\_SHARED\_EDGE\_PX` from Chapter 3 §3.9), the shared boundary `LineString` is added as one or more wall graph edges. Shared boundaries may contain intermediate vertices (introduced by the BSP split operations); each vertex becomes a node.

```python
for room\_a, room\_b in adjacency\_pairs:
    shared = room\_a.polygon.boundary.intersection(room\_b.polygon.boundary)
    coords = list(shared.coords)
    for i in range(len(coords) - 1):
        node\_a = graph.add\_node(coords\[i], on\_exterior=False)
        node\_b = graph.add\_node(coords\[i+1], on\_exterior=False)
        graph.add\_edge(node\_a, node\_b,
                       is\_exterior=False,
                       is\_diagonal=False,
                       thickness=None)
```

###### 

###### Pass 3 — Diagonal Wall Insertion



Each `DiagonalWall` from Chapter 4 is inserted as one edge. The diagonal's two centreline endpoints are snapped to the nearest existing graph node if within `SNAP\_TOLERANCE = 3.0` pixels; otherwise a new node is created. If a diagonal endpoint falls on an existing edge (rather than an existing node), that edge is **split** at the intersection point: the old edge is removed, two new edges are added from the split point to the original endpoints, and the split point becomes a new T-junction node.

```python
for diag in diagonal\_walls:
    p1\_node = graph.snap\_or\_add(diag.centreline.coords\[0])
    p2\_node = graph.snap\_or\_add(diag.centreline.coords\[-1])
    graph.add\_edge(p1\_node, p2\_node,
                   is\_exterior=diag.placement\_mode in ("corner\_bevel", "facade\_chamfer"),
                   is\_diagonal=True,
                   thickness=diag.thickness,
                   merged\_polygon=diag.merged\_polygon)
```

###### 

###### Pass 4 — Junction Typing



After all three passes, every node's degree is known. Junction types are assigned:



|Degree|Junction Type|
|-|-|
|1|`I` (terminal / dead-end)|
|2|`I` (pass-through, straight or bent)|
|3|`T`|
|4|`X`|
|≥5|`X` (treated as X with extra incident edges; rare)|

The distinction between a degree-2 straight pass-through and a degree-2 corner (L-junction) is made by computing the angle between the two incident edges:

```python
if degree == 2:
    angle = angle\_between(edge\_a.direction, edge\_b.direction)
    junction\_type = "I" if abs(angle) < 10.0 or abs(angle - 180.0) < 10.0 else "L"
```

There are four wall junction types: I-, L-, T-, and X-shaped, depending on the degrees of incident wall segments. Considering orientations, there are in total 13 (= 4 + 4 + 4 + 1) types. For this generator, the 13-type orientation-aware classification is not used — the four base types are sufficient for the wall rendering and edge-clipping operations in §5.6.

&#x20;

##### 5.3 Wall Graph Edge Attributes



Every edge in the wall graph carries a `WallSegment` attribute bundle:

```python
@dataclass
class WallSegment:
    # Geometry
    node\_a: int                      # graph node index
    node\_b: int                      # graph node index
    centreline: shapely.LineString   # pixel space

    # Classification
    is\_exterior: bool                # True if on building boundary
    is\_diagonal: bool                # True if produced by Ch. 4
    room\_type\_a: str | None          # room type on left side of segment
    room\_type\_b: str | None          # room type on right side (None for exterior)

    # Thickness and style
    thickness: float                 # pixels; assigned by §5.4
    draw\_style: str                  # "solid" | "hollow" | "hatch"
    hollow\_gap: float                # pixels; inner gap for hollow style (0 if solid)

    # Junction
    junction\_type\_a: str             # "I" | "L" | "T" | "X" at node\_a
    junction\_type\_b: str             # "I" | "L" | "T" | "X" at node\_b

    # Rendering
    wall\_polygon: shapely.Polygon    # perpendicular-normal polygon (Ch. 4 §4.5.2)
    merged\_polygon: shapely.Polygon  # after junction union (Ch. 4 §4.6.2)

    # Flags
    opening\_eligible: bool           # False if is\_diagonal or too short
    era: str                         # inherited from generator era token
```

The `wall\_polygon` and `merged\_polygon` fields are computed during graph construction for diagonal walls (Chapter 4) and computed fresh here for axis-aligned walls using the same perpendicular-normal method (Chapter 4 §4.5.2). All walls — diagonal and axis-aligned — use the same polygon-based representation. There are no stroked-line walls anywhere in the pipeline.

&#x20;

##### 5.4 Thickness Assignment



###### 5.4.1 Thickness Classes



Wall thickness in real floor plans is not uniform. In architecture floor plans, there are different line thicknesses used for different elements. This is called line weights, used to make the plan easier to understand and help make the link between the 2D and the 3D. The generator models three wall classes, each with a distinct thickness range at the base resolution of 512×512 pixels:



|Class|Description|Thickness range (px)|Pixel equivalent|
|-|-|-|-|
|`EXTERIOR`|Outer building walls|10–16 px|\~200–320 mm at 1:100|
|`STRUCTURAL`|Load-bearing interior walls|7–11 px|\~140–220 mm|
|`PARTITION`|Non-structural interior divisions|4–7 px|\~80–140 mm|



Interior walls are usually about 4½ inches thick and exterior walls around 6½ inches. This real-world ratio of approximately 1:1.5 (partition:exterior) is preserved in the pixel ranges above. Khrushchevka partitions used thin gypsum-concrete, gypsum-sawdust, or reinforced concrete panels, with dual 80 mm partitions and a 40 mm gap between apartments. This informs the lower bound of the `PARTITION` class — the thinnest partition visible as a distinct wall at typical floor plan scales.

Wall class assignment rules:

```python
def assign\_thickness\_class(segment: WallSegment,
                            rooms: list\[Room]) -> str:
    if segment.is\_exterior:
        return "EXTERIOR"
    # Interior: check room types on both sides
    types = {segment.room\_type\_a, segment.room\_type\_b}
    if "stairwell" in types or "corridor" in types:
        return "STRUCTURAL"   # stair/corridor walls are typically load-bearing
    if rooms\_share\_apartment\_boundary(segment, rooms):
        return "STRUCTURAL"   # cross-apartment walls are heavier
    return "PARTITION"
```



###### 5.4.2 Base Thickness Sampling



Once the class is determined, the base thickness is drawn uniformly from the class range:

```python
BASE\_THICKNESS = {
    "EXTERIOR":   lambda rng: rng.uniform(10, 16),
    "STRUCTURAL": lambda rng: rng.uniform(7, 11),
    "PARTITION":  lambda rng: rng.uniform(4, 7),
}
thickness = BASE\_THICKNESS\[wall\_class](rng)
```

##### 

###### 5.4.3 Per-Image Thickness Consistency



Within a single image, all walls of the same class use the **same sampled base thickness**, plus a small per-segment jitter. This models the real-world convention where a draughtsman draws all exterior walls at the same nominal width. The per-image base thickness is sampled once:

```python
image\_thickness = {
    cls: BASE\_THICKNESS\[cls](image\_rng) for cls in ("EXTERIOR", "STRUCTURAL", "PARTITION")
}
```

Then per-segment jitter is applied:

```python
segment.thickness = image\_thickness\[wall\_class] + segment\_rng.normal(0, JITTER\_SIGMA\[era])
```

The jitter sigma is era-dependent:

|Era|`JITTER\_SIGMA`|
|-|-|
|`digital`|0.3 px (near-zero; CAD precision)|
|`scan`|0.8 px (slight scan/print variation)|
|`soviet`|1.8 px (hand-drawing pressure variation)|

The Soviet sigma of 1.8 px produces visibly uneven wall widths along a single long exterior wall segment — a characteristic feature of hand-drawn BTI plans that the model must learn to handle.



###### 5.4.4 Thickness Scaling for Non-512 Outputs



All thickness values are computed at 512×512. For other output sizes, scale linearly:

```python
thickness\_scaled = thickness \* (image\_size\[0] / 512.0)
thickness\_scaled = max(thickness\_scaled, MIN\_THICKNESS\_PX)  # floor: 3px at any size
```

&#x20;

##### 5.5 Draw Style Assignment



The `draw\_style` attribute governs how the wall polygon is rendered in the image (not in the mask — mask behaviour is covered in §5.8 and Chapter 11). Three styles are defined:

###### 

###### Style 1 — Solid (`"solid"`)



The wall polygon is filled with a single colour (black or near-black, era-dependent). This is the dominant style in post-2000 scan and digital era plans. Architects use hatches to shade in exterior walls — what architects call poche — typically as an 80% screen on a solid hatch on the hatch layer. In the scan and digital eras, this translates to a pure black or very dark grey fill with no internal structure.

###### 

###### Style 2 — Hollow (`"hollow"`)



The wall is rendered as two parallel outline strokes with the interior left at paper colour. This is the dominant style in Soviet-era BTI plans. The `hollow\_gap` parameter controls the interior width:

```python
hollow\_gap = thickness \* HOLLOW\_GAP\_FRAC\[era]

HOLLOW\_GAP\_FRAC = {
    "soviet":  rng.uniform(0.35, 0.55),   # thick contours, wide gap
    "scan":    rng.uniform(0.25, 0.40),   # thinner contours
    "digital": rng.uniform(0.20, 0.30),   # fine-line technical drawing look
}
```

Geometrically, the hollow wall is rendered as two filled rectangles: an outer rectangle of the full wall polygon, and an inner rectangle inset by `(thickness - hollow\_gap) / 2` on each side, filled with the paper colour. This is equivalent to an `outline` draw, but is implemented as two polygon fills to ensure clean corners at junctions (see §5.6).

###### 

###### Style 3 — Hatch (`"hatch"`)



The wall polygon is filled with a diagonal hatch pattern. Hatching is used very differently in design drawings versus working drawings. Hatching is used to communicate information about materials, textures and finishes, usually done in one of the thinnest line weights, and is secondary and supporting information to the outline of elements. In Soviet BTI plans, brick masonry walls are often cross-hatched or diagonal-hatched at 45°.

The hatch renderer:

1. Fills the wall polygon with paper colour.
2. Clips a grid of parallel lines (spacing = `thickness \* 0.35`, angle = 45°) to the wall polygon boundary using `shapely.ops.split` of the hatch-line against the polygon. Lines are drawn at the `solid\_line\_width` appropriate to the era.

###### 

###### 5.5.1 Per-Image Style Mix



Draw styles are assigned **per image**, not per segment. A single image uses one dominant style for all walls, with a possible minority variant:

```python
dominant\_style, minority\_style, minority\_prob = STYLE\_MIX\[era]

STYLE\_MIX = {
    "digital": ("solid",  None,     0.00),
    "scan":    ("solid",  "hollow", 0.10),
    "soviet":  ("hollow", "hatch",  0.20),
}
```

For each segment, the style is drawn from:

```python
if segment\_rng.uniform() < minority\_prob:
    segment.draw\_style = minority\_style
else:
    segment.draw\_style = dominant\_style
```

This produces images that are predominantly one style (as real plans are) but occasionally contain mixed conventions — as old plans revised by different hands often do.

&#x20;

##### 5.6 Junction Geometry and Edge Clipping

###### 

###### 5.6.1 The Clipping Problem



When two wall polygons of different thicknesses meet at a T- or X-junction, their raw polygons overlap. Rasterising both independently would produce a double-painted region, which for solid walls is harmless (same class twice) but for hollow walls introduces an artefact: the inner paper-coloured gap of one wall is painted over the solid outer boundary of the other, leaving a visible notch in the heavier wall.

The solution is **edge clipping**: before a wall polygon is rasterised, it is trimmed so that it does not extend beyond the boundary of any heavier wall it meets at a junction.

### 

###### 5.6.2 Edge-Clipping Logic



This extends v6.0's existing edge-clipping logic, which operated on axis-aligned wall rectangles, to the wall-graph representation:

```python
def clip\_segment\_at\_junctions(segment: WallSegment,
                               graph: WallGraph) -> shapely.Polygon:
    """
    Returns segment.wall\_polygon clipped at both endpoints by any heavier
    walls that dominate at that junction.
    """
    clipped = segment.wall\_polygon

    for node\_id in (segment.node\_a, segment.node\_b):
        node = graph.nodes\[node\_id]
        incident = graph.edges\_at(node\_id)

        for other\_seg in incident:
            if other\_seg.segment\_id == segment.segment\_id:
                continue
            if other\_seg.thickness <= segment.thickness:
                continue  # only clip against heavier segments

            # Clip: remove the region of `clipped` that falls inside `other\_seg.wall\_polygon`
            clipped = clipped.difference(other\_seg.wall\_polygon)
            clipped = make\_valid(clipped)

            if clipped.is\_empty:
                break  # fully consumed; stop
        
        if clipped.is\_empty:
            break

    return clipped
```

The rule is simple: **lighter walls yield to heavier walls at junctions**. An exterior wall (10–16 px) clips partition walls (4–7 px) at every junction they share. A structural wall clips a partition wall. Two walls of identical thickness at a T-junction are handled by the alternative approach: their polygons are `unary\_union`-ed at the junction point (identical to the diagonal wall miter handling in Chapter 4 §4.6.2), producing a single merged polygon that is rasterised once.

###### 

###### 5.6.3 L-Junction Geometry



At an L-junction (degree-2 corner node), two wall segments meet at a bend. The raw wall polygons of the two segments do not naturally fill the corner region — they leave a triangular gap at the inside of the bend. This is analogous to the diagonal miter problem in Chapter 4.

The fill is computed as:

```python
corner\_fill = segment\_a.wall\_polygon.union(segment\_b.wall\_polygon)
corner\_fill = make\_valid(corner\_fill)
# corner\_fill is the merged polygon used for rasterisation at this junction
```

`unary\_union` of the two raw wall polygons fills the corner gap automatically. The result is stored as the `merged\_polygon` of the junction node and is rasterised in place of the two individual segment polygons at the junction overlap region.

###### 

###### 5.6.4 T-Junction Geometry



At a T-junction (degree-3 node), one wall segment continues through and two branch off (or one branches). The through-wall has higher priority and is not clipped. The branch walls are clipped against the through-wall polygon exactly as in §5.6.2.

The T-junction additionally requires that the through-wall's polygon be extended to fill the stub region created by the branch wall's thickness:

```python
# Extend through-wall to cover the branch stub
branch\_stub = branch\_segment.wall\_polygon.intersection(
    through\_segment.wall\_polygon.buffer(through\_segment.thickness \* 0.5)
)
through\_extended = through\_segment.wall\_polygon.union(branch\_stub)
through\_segment.merged\_polygon = make\_valid(through\_extended)
```

Without this extension, a thin white line can appear across the through-wall at the T-junction point — particularly visible in hollow-wall style where the gap extends through the junction.

###### 

###### 5.6.5 X-Junction Geometry



At an X-junction (degree-4 node), four wall segments meet. The two thicker segments (typically the exterior or structural walls forming the dominant axes) are unioned first; the two thinner segments are clipped against the union result:

```python
heavy\_pair = sorted(incident\_segments, key=lambda s: -s.thickness)\[:2]
light\_pair  = sorted(incident\_segments, key=lambda s: -s.thickness)\[2:]

junction\_core = unary\_union(\[s.wall\_polygon for s in heavy\_pair])
junction\_core = make\_valid(junction\_core)

for seg in light\_pair:
    seg.merged\_polygon = make\_valid(seg.wall\_polygon.difference(junction\_core))
```

&#x20;

##### 5.7 Era-Specific Rendering Modifiers



The wall graph produces geometry (polygons) but not pixels. Pixel rendering happens in Chapter 9's era-theme system. However, Chapter 5 is responsible for setting the **rendering metadata** on each segment that Chapter 9 will consume. Three era-specific modifications are applied at this stage:

###### 

###### 5.7.1 Soviet: Line-End Imperfection



For `era == "soviet"`, axis-aligned wall segments receive a small **overshoot parameter**: each endpoint is extended or retracted by `rng.normal(0, OVERSHOOT\_SIGMA)` pixels, where `OVERSHOOT\_SIGMA = 1.5`. This is stored as `segment.endpoint\_jitter = (jitter\_a, jitter\_b)` and consumed by Chapter 9's Soviet renderer, which shifts the endpoint of the wall polygon accordingly.

Overshoot — where a draughtsman's pen continues slightly past the junction — is a defining visual characteristic of hand-drawn BTI plans. Soviet building at the time lacked standardized sizes, clear work organization, and efficient task distribution, relying on semi-handcrafted methods — a fact reflected directly in the visual imprecision of their technical drawings.

###### 

###### 5.7.2 Soviet: Ink Density Variation



For `era == "soviet"`, a **density map** is generated per wall polygon: a 1D array of per-pixel opacity multipliers along the wall's length, drawn from a slow random walk:

```python
n\_steps = int(segment.centreline.length)
walk = np.cumsum(rng.normal(0, 0.02, n\_steps))
walk = walk - walk.mean()
density\_profile = np.clip(1.0 + walk, 0.60, 1.05)
segment.ink\_density\_profile = density\_profile
```

The density profile is consumed by Chapter 9 to modulate the fill darkness along the wall, producing the characteristic faded-ink look of Soviet-era plans where the draughtsman's ink ran low mid-stroke.

###### 

###### 5.7.3 Digital: Sub-Pixel Line Precision



For `era == "digital"`, all wall polygon vertices are kept at their exact floating-point coordinates (not rounded to integer pixels) until the final rasterisation step in Chapter 11. This preserves the sharp, precise line quality of CAD-exported plans. For `scan` and `soviet`, vertices are rounded to integer pixels before rendering, which introduces the slight geometric imprecision characteristic of those eras.

&#x20;

##### 5.8 The Hollow-Wall Mask Rule



This is the most important constraint in the entire chapter, and it is repeated here in full to ensure it is not missed during implementation.

**When a wall is drawn in hollow style, both the contour polygons AND the interior paper-coloured gap are labelled class 1 (wall) in the mask.**

The image will show a white gap between two dark lines. The mask will show solid class 1 across the full wall band. This is not an error — it is a deliberate training signal.

The rationale: if the interior gap were labelled class 0 (background), the model would see:

* Image: dark | white | dark (hollow wall)
* Mask: wall | background | wall

This pattern is structurally identical to two thin parallel walls with a background gap, which is a legitimate floor plan feature (a narrow corridor). Training on this ambiguous signal would cause the model to confuse narrow corridors with hollow walls. By labelling the entire hollow wall band as class 1, the model learns that the visual pattern of two parallel close-spaced lines is always wall, regardless of what fills the gap.

Implementation:

```python
def hollow\_wall\_mask\_polygon(segment: WallSegment) -> shapely.Polygon:
    """
    Returns the full wall band polygon for mask writing.
    Does NOT subtract the inner gap — the full band is class 1.
    """
    return segment.wall\_polygon  # the full perpendicular-normal polygon, pre-gap
```

The image renderer (Chapter 9) uses `segment.hollow\_gap` to draw the visual gap. The mask writer (Chapter 11) uses `segment.wall\_polygon` directly, ignoring `hollow\_gap` entirely.

&#x20;

##### 5.9 Wall Graph Validation



After full construction and all edge-clipping operations, the wall graph is validated before being passed downstream. Failures trigger a regeneration of the current sample.

###### 

###### Validation Checks



**Check 1 — Connectivity:**

```python
assert nx.is\_connected(graph.to\_networkx())
```

A disconnected wall graph indicates a floating wall segment with no junction connections — a pathological geometry that would produce isolated wall polygons in the mask.

**Check 2 — No Degenerate Segments:**

```python
for seg in graph.edges():
    assert seg.centreline.length >= MIN\_SEGMENT\_LENGTH\_PX  # default: 4px
    assert not seg.wall\_polygon.is\_empty
    assert seg.wall\_polygon.is\_valid
```

**Check 3 — No Invalid Merged Polygons:**

```python
for seg in graph.edges():
    if seg.merged\_polygon is not None:
        assert seg.merged\_polygon.is\_valid
        assert not seg.merged\_polygon.is\_empty
        assert seg.merged\_polygon.geom\_type == "Polygon"
```

The `MultiPolygon` guard in Check 3 catches the case where `unary\_union` at a junction returned a disconnected result due to near-zero overlap. If any check fails, the detailed failure reason is written to `metadata\["wall\_graph\_validation\_failure"]` before regeneration is triggered.

**Check 4 — Opening Eligibility Audit:**

```python
eligible\_count = sum(1 for seg in graph.edges()
                     if seg.opening\_eligible)
assert eligible\_count >= MIN\_ELIGIBLE\_SEGMENTS  # default: 3
```

If fewer than 3 segments are eligible for openings, Chapter 6 cannot place the minimum required windows and doors. This check catches degenerate footprints where nearly all exterior walls are either too short or diagonal.

&#x20;

##### 5.10 The Wall Graph Dataclass



```python
@dataclass
class WallGraph:
    nodes: dict\[int, WallNode]           # node\_id → WallNode
    edges: dict\[int, WallSegment]        # edge\_id → WallSegment
    exterior\_edge\_ids: list\[int]         # edges with is\_exterior=True
    interior\_edge\_ids: list\[int]         # edges with is\_exterior=False
    diagonal\_edge\_ids: list\[int]         # edges with is\_diagonal=True
    junction\_nodes: dict\[str, list\[int]] # "I"/"L"/"T"/"X" → \[node\_ids]
    era: str                             # propagated from generator input
    image\_thickness: dict\[str, float]   # per-class base thickness for this image
    dominant\_draw\_style: str            # "solid" | "hollow" | "hatch"
    is\_valid: bool                       # True after all validation checks pass
```

```python
@dataclass
class WallNode:
    node\_id: int
    position: tuple\[float, float]        # pixel coordinates
    on\_exterior: bool
    junction\_type: str                   # "I" | "L" | "T" | "X"
    degree: int                          # number of incident edges
    incident\_edge\_ids: list\[int]
```

&#x20;

##### 5.11 Backwards Compatibility



Chapter 5's wall graph is a new data structure that did not exist in v6.0. v6.0 used a direct polygon-to-renderer pipeline with no intermediate graph representation. Backward compatibility is maintained as follows:

* The v6.0 plugin strategy system is retained. The wall graph builder is registered as a new strategy named `"wall\_graph\_v7"`. A shim strategy `"wall\_legacy\_v6"` wraps the old direct-polygon path for callers that pass `wall\_strategy="v6"` explicitly.
* YOLO bounding box generation reads from the wall graph's `exterior\_edge\_ids` and `interior\_edge\_ids` lists. The output format is identical to v6.0.
* The metadata JSON gains new keys: `wall\_graph.node\_count`, `wall\_graph.edge\_count`, `wall\_graph.dominant\_draw\_style`, `wall\_graph.image\_thickness`. These are additive; existing metadata consumers see no breaking change.
* The v6.0 edge-clipping logic for axis-aligned rectangular walls is preserved as a special case within §5.6.2: when both segments are axis-aligned and of equal thickness, the v6.0 rectangular-clip path is called directly, saving the overhead of the polygon-difference operation for the common case.

&#x20;

##### 5.12 Relationship to Downstream Chapters



|Chapter|Dependency on wall graph|
|-|-|
|**6**|Reads `opening\_eligible`, `is\_exterior`, `room\_type\_a/b`, and `junction\_type\_a/b` to select opening host segments|
|**7**|Reads `is\_exterior` and `room\_type\_a` to determine wall-snap positions for icon placement|
|**9**|Reads `draw\_style`, `hollow\_gap`, `ink\_density\_profile`, `endpoint\_jitter`, and `era` to render each segment in the correct aesthetic|
|**10**|No direct dependency; degradation is applied to the fully-rendered image after all wall polygons have been rasterised|
|**11**|Reads `wall\_polygon` (never `merged\_polygon`) for mask writing; reads `draw\_style` to apply the hollow-wall mask rule (§5.8); write-order places wall polygons after room fills and before opening cutouts|



The wall graph is passed by reference through the entire remaining pipeline. Existing methods face challenges in design complexity and constrained generation with extensive post-processing, and tend to obvious geometric inconsistencies such as misalignment, overlap, and gaps. The graph-based representation eliminates these inconsistencies at generation time by resolving all junction geometry before any rasterisation occurs — a systematic approach to a problem that has caused persistent artefacts in simpler procedural generators.





&#x20;

#### Chapter 6 — Openings (Windows + Doors)



##### 6.1 The Role of Openings in the Five-Class Schema



Floor plan symbols represent the size and location of structural elements like walls, doors, windows, and stairs, as well as mechanical elements like plumbing and HVAC systems. In the five-class schema, both windows (class 2) and doors (class 3) are carved directly out of wall polygons. They are not drawn *on top* of walls — they replace a rectangular section of the wall band entirely, and the mask reflects this: the pixels of a window or door opening carry class 2 or class 3, not class 1. No pixel in the output mask may belong simultaneously to class 1 and either class 2 or 3.

This chapter defines: the geometry of each opening type; the rules governing which wall segments are eligible to receive openings; the room-type light rules that drive window placement priority; the door placement algorithm driven by the adjacency graph from Chapter 3; and the BTI-specific symbol variants inherited from v6.0. The chapter closes with the opening carve-out protocol — the exact sequence in which openings are subtracted from wall polygons before any rasterisation occurs.

&#x20;

##### 6.2 The Axis-Aligned Constraint



**Diagonal walls receive no openings.** This constraint is encoded in the `WallSegment.opening\_eligible` flag set in Chapter 5: any segment with `is\_diagonal = True` has `opening\_eligible = False`, and the placement algorithms in this chapter never attempt to place an opening on such a segment.

The constraint exists for two reasons established in Chapter 4. Geometrically, the opening carve-out operation (§6.8) uses axis-aligned rectangular cutouts; applying these to a diagonal wall produces an incorrectly oriented gap whose mask label would be inconsistent. From a prevalence standpoint, diagonal-wall openings are uncommon in the real-world corpus — Soviet-era panel blocks with chamfered facade corners carry solid diagonal surfaces, not windows.

All opening placement logic below therefore operates exclusively on axis-aligned `WallSegment` instances.

&#x20;

##### 6.3 Wall Segment Eligibility



Beyond the diagonal constraint, a wall segment must pass three additional eligibility checks before it can receive any opening.

###### 

###### Check 1 — Minimum Segment Length



```python
MIN\_OPENING\_HOST\_LENGTH\_PX = 3 \* MIN\_INTERIOR\_WIDTH\_PX  # default: 240px at 512×512
assert segment.centreline.length >= MIN\_OPENING\_HOST\_LENGTH\_PX
```

A segment shorter than three times the minimum interior width cannot accommodate even one standard opening with adequate clearance on both flanks. This is the most common eligibility failure for corridor and utility room walls.

###### 

###### Check 2 — Exterior vs. Interior Eligibility



Windows may only be placed on **exterior** wall segments (`is\_exterior = True`). One square foot of natural light is needed for every 10 square feet of floor space — this real-world constraint is the architectural rationale: windows are always exterior-facing. Interior walls do not receive windows.

Doors may be placed on both exterior and interior walls. Exterior doors are the main entrance; interior doors connect adjacent rooms. The placement algorithm handles both sub-types (§6.6).

###### 

###### Check 3 — Room-Type Light Rule Override



Some room types have special eligibility rules that override the general exterior check:

```python
WINDOW\_ELIGIBILITY\_OVERRIDE = {
    "corridor":   False,   # corridors receive no windows
    "utility":    False,   # utility rooms receive no windows
    "stairwell":  False,   # stairwells receive no windows
    "bathroom":   True,    # bathrooms may receive a small exterior window
    "kitchen":    True,    # kitchens require at least one exterior window
    "living":     True,    # living rooms require at least one exterior window
    "bedroom":    True,    # bedrooms require at least one exterior window
    "balcony":    False,   # balconies are already open; no window symbol
}
```

Bathrooms, hallways, closets, and garages are exempt from natural light requirements in real buildings. In the generator, corridors and utility rooms are similarly suppressed, keeping the synthetic corpus consistent with real-world expectations.

The `opening\_eligible` flag on each `WallSegment` is set as the AND of all three checks. Once set, it does not change. The placement algorithms read this flag and skip ineligible segments without attempting placement.

&#x20;

##### 6.4 Window Geometry

##### 

###### 6.4.1 The Standard Window Symbol



Window floor plan symbols are represented as breaks in the wall, typically shown with three parallel lines that distinguish them from solid walls. In this generator, the "three-line" convention is rendered as a rectangular opening cut into the wall, with three parallel horizontal strokes across the opening width: one at the centre (the glazing midline) and two at the inner and outer wall faces (the sill/reveal lines). This is the post-2000 scan and digital era convention.

The geometry of a single standard window:

```
Wall outer face:   ───────────┬──────────────┬───────────
                              │  outer sill  │
                              │──────────────│  ← sill line (1px)
                              │              │
                              │──────────────│  ← glazing midline (1px)
                              │              │
                              │──────────────│  ← inner sill (1px)
                              │  inner sill  │
Wall inner face:   ───────────┴──────────────┴───────────
```

Formally, the window opening is a rectangle of width `w\_window` along the wall centreline and depth equal to the full wall thickness `t`. The three sill lines are drawn at `y = 0`, `y = t/2`, and `y = t` within the opening rectangle's local coordinate frame (y measured perpendicular to the wall).

###### 

###### 6.4.2 Window Size Sampling



A normal window typically ranges from 24 to 48 inches wide, with 36 inches being the most common width for standard residential applications. Mapped to the generator's pixel scale at 1:100 (100 mm/px at 512×512), these translate to approximately 24–48 px wide. The generator uses:

```python
WINDOW\_WIDTH\_RANGE = {
    "living":   (36, 54),   # larger windows; living rooms benefit from large windows
    "bedroom":  (24, 40),
    "kitchen":  (20, 36),
    "bathroom": (14, 22),   # small, privacy-oriented
}
# Fallback for unlisted room types:
DEFAULT\_WINDOW\_WIDTH\_RANGE = (20, 36)
```

Especially in rooms where we spend most of our time — such as the living or dining room — natural light and the right window layout are extremely important. This justifies the wider range for `living` rooms relative to all others.

Window width is drawn from the appropriate range with `rng.uniform()`, then snapped to the nearest even integer pixel (a convention from real plan drawing, where window widths are always even multiples of the drawing unit).

###### 

###### 6.4.3 Window Count Per Room



The number of windows placed on a room's exterior edges is governed by:

```python
WINDOW\_COUNT = {
    "living":   rng.choice(\[1, 2, 3], p=\[0.20, 0.55, 0.25]),
    "bedroom":  rng.choice(\[1, 2],    p=\[0.65, 0.35]),
    "kitchen":  rng.choice(\[1, 2],    p=\[0.70, 0.30]),
    "bathroom": rng.choice(\[0, 1],    p=\[0.30, 0.70]),
}
```

Living rooms often benefit from large windows or even floor-to-ceiling glass panels that provide expansive views and ample sunlight. The living room distribution reflects this — it is the only room type with a non-trivial probability of three windows.

### 

###### 6.4.4 Window Placement Along a Segment



Once a target segment and window count are determined, windows are distributed along the segment by the following algorithm:

1. Compute the **usable length** of the segment: `L\_usable = segment.length - 2 × WINDOW\_MARGIN`, where `WINDOW\_MARGIN = 12px`. This reserves clearance at both ends to prevent windows from touching wall junctions.
2. Divide `L\_usable` into `n\_windows` equal slots of width `L\_usable / n\_windows`.
3. Within each slot, place the window centred with a random offset `rng.uniform(-slot\_width \* 0.2, slot\_width \* 0.2)`.
4. Validate that no two windows overlap and that each window has at least `MIN\_WINDOW\_CLEARANCE = 8px` of solid wall on each side.

Whenever possible, align all the windows on a wall both horizontally and vertically. The equal-slot distribution honours this convention by default; the jitter introduces enough variation to avoid a rigid, CAD-perfect look while maintaining approximate alignment.

###### 

###### 6.4.5 Soviet-Era Window Variant



For `era == "soviet"`, the three-line window symbol is replaced with a **two-line** variant: only the outer and inner sill lines are drawn, with no glazing midline. This matches the BTI convention visible in Soviet residential survey drawings, where the window reveal is shown as a simple rectangular gap with parallel edge lines and no internal structure.

&#x20;

##### 6.5 Door Geometry



###### 6.5.1 The Standard Swing-Arc Door



Door symbols in architectural drawings typically appear as a straight line that interrupts wall lines, followed by an arc indicating the direction in which the door swings. The standard door symbol in this generator consists of:

* A **door leaf line**: a thin rectangle of width `w\_door` and depth 1px, representing the door slab.
* A **swing arc**: a quarter-circle arc of radius `w\_door`, centred at the hinge point, swept 90° from the door-closed position to the door-open position.

```python
def draw\_swing\_door(opening\_rect: Polygon,
                    hinge\_side: str,          # "left" or "right" along wall direction
                    swing\_direction: str,      # "inward" or "outward" (into room)
                    w\_door: float,
                    era: str) -> tuple\[Polygon, Path]:
    """
    Returns (opening\_polygon, arc\_path).
    opening\_polygon: the rectangular cut in the wall (class 3 in mask).
    arc\_path: the arc drawn in the image only; NOT written to mask.
    """
    hinge\_point = ...  # computed from opening\_rect and hinge\_side
    arc = Arc(centre=hinge\_point, radius=w\_door, start\_angle=..., end\_angle=...)
    return opening\_rect, arc
```

The orientation of the arc in door symbols indicates whether a door swings inward or outward, critical for understanding the door's functionality in a space.

The swing arc is drawn in the **image only**. It is explicitly not written to the mask. The mask receives only the rectangular opening polygon (class 3). The arc is a decorative annotation that the model must learn to ignore when classifying the opening region.

Hinge side is sampled uniformly (`left` or `right` along the wall segment direction). Swing direction is sampled with `p(inward) = 0.6`, `p(outward) = 0.4`, matching the real-world prevalence of inward-swinging residential doors.

###### 

###### 6.5.2 Door Size Sampling



Standard interior doors in residential buildings typically measure 0.8 meters (or 80 centimeters) in width. For exterior doors, the standard width can range from 0.9 meters to 1.2 meters, depending on the design and purpose of the entrance.

At the generator's 1:100 pixel scale:

```python
DOOR\_WIDTH\_RANGE = {
    "exterior\_main":  (28, 36),   # 900–1200mm at 1:100 → 28–36px at 512×512 (scaled)
    "interior":       (22, 28),   # 700–900mm
    "bathroom":       (18, 22),   # 600–700mm; narrower for privacy rooms
}
```

Interior hinged doors (nominal widths): 24", 28", 30", 32", 34", 36" (610, 711, 762, 813, 864, 914 mm). The generator samples from a discrete set rather than a continuous range: `{18, 20, 22, 24, 26, 28, 30, 32, 34, 36}` px, weighted toward the centre of the appropriate class range.

###### 

###### 6.5.3 Door Type Distribution



Door symbols are drawn using simple lines and arcs to highlight important information such as door type, swing direction, and placement. The main types of door symbols include single, double, sliding, and door openings.

The generator produces three door types, with per-era probabilities:



|Type|Digital|Scan|Soviet|
|-|-|-|-|
|Single swing|0.65|0.70|0.80|
|Double swing|0.15|0.10|0.05|
|BTI cross-ridge|0.20|0.20|0.15|



Double swing doors are only placed on openings wide enough to accommodate two leaves: `w\_opening >= 2 × w\_door\_min`. They are rendered as two swing doors side by side, symbolized by two arcs next to each other on the floor plan.



###### 6.5.4 The BTI Cross-Ridge Door Symbol



The BTI cross-ridge symbol — inherited from v6.0 and retained unchanged — is the primary Soviet-era door representation. It consists of:

* A rectangular opening in the wall of width `w\_door` (class 3 in the mask).
* An X drawn across the full opening rectangle using two diagonal lines from corner to corner.
* No swing arc.

This symbol originated in Soviet technical inventory (BTI) conventions as a simplified door mark used when swing direction was not relevant to the survey purpose. The X pattern distinguishes it unambiguously from windows (which carry parallel horizontal lines) and from modern swing-arc door symbols. It is always drawn in the image only — the X diagonals are image-layer marks, not mask marks. The mask receives only the rectangular opening.

For `era == "soviet"`, the BTI cross-ridge symbol is the dominant door type (weight 0.80 as shown above for single swing; the table above assigns 0.15 to cross-ridge, but this is recalibrated in the Soviet era to `0.75` cross-ridge, `0.15` single swing, `0.10` double swing). This reflects that BTI survey drawings almost universally used the cross-ridge convention.

&#x20;

##### 6.6 Door Placement Algorithm



Door placement is driven by the room adjacency graph from Chapter 3. Every pair of adjacent rooms must have at least one connecting door; the placement algorithm guarantees this before placing any additional doors.

###### 

###### 6.6.1 Required Door Pass



```python
for room\_a, room\_b in adjacency\_pairs:
    shared\_segments = get\_shared\_wall\_segments(room\_a, room\_b, wall\_graph)
    eligible = \[s for s in shared\_segments if s.opening\_eligible and not s.is\_diagonal]
    
    if not eligible:
        # No eligible shared segment: log warning, skip this pair
        metadata\["missing\_door\_pairs"].append((room\_a.room\_id, room\_b.room\_id))
        continue
    
    host\_segment = max(eligible, key=lambda s: s.centreline.length)
    place\_door(host\_segment, door\_type="interior", rng=rng)
```

The longest eligible shared segment is chosen as the host for the required door. This models the architectural convention of placing doors in the widest available wall rather than cramming them into short partition stubs.



###### 6.6.2 Exterior Door Pass



At least one **exterior door** (main entrance) is placed per floor plan. The host segment is selected from exterior wall segments with the following priority:

1. Segments on the longest exterior edge of the footprint (the primary facade).
2. Segments adjacent to a `corridor` room.
3. Segments on any other exterior edge if 1 and 2 are unavailable.

The main entrance door always uses `door\_type = "exterior\_main"` (the wider size range) and is placed at the longitudinal centre of the host segment ± `rng.uniform(-0.15, 0.15) × segment.length`.



###### 6.6.3 Additional Interior Door Pass



After the required and exterior passes, additional doors are placed stochastically on eligible interior segments that do not yet have a door:

```python
additional\_door\_prob = {"living": 0.15, "bedroom": 0.10, "kitchen": 0.20, "bathroom": 0.05}
```

These model the real-world occurrence of secondary interior connections — a kitchen-to-dining room pass-through, a second bedroom door, etc.

###### 

###### 6.6.4 Door Position Along Segment



Within the host segment, the door is positioned using the same clearance algorithm as windows (§6.4.4), with `DOOR\_MARGIN = 14px` on each flank. Doors are never placed within `DOOR\_MARGIN` pixels of a wall junction, regardless of segment length.

&#x20;

##### 6.7 Windows-First Placement Order



Windows are always placed before doors. This ordering is not arbitrary — it reflects the structural logic of opening placement:

1. **Windows are exterior-only**. Placing them first claims the most valuable exterior wall real estate (long exterior segments with exterior-facing rooms) before door placement can monopolise those segments.
2. **Door-swing arcs must not overlap windows**. Placing windows first establishes no-go zones for the swing arc extents of any doors subsequently placed on the same or adjacent segments.
3. **Mask write-order consistency**. Chapter 11's write-order protocol requires that openings be written to the mask in a fixed order (class 2 before class 3). Generating them in the same order ensures no ambiguity about which class occupies a pixel at the boundary between an adjacent window and door.

The windows-first rule is enforced at the algorithm level: `place\_windows()` is called and fully completed before `place\_doors()` begins. After `place\_windows()`, each host segment's `occupied\_intervals` list is populated with the wall-coordinate ranges claimed by windows; `place\_doors()` reads this list and rejects positions that would produce overlaps.

&#x20;

##### 6.8 The Opening Carve-Out Protocol



After all openings are placed and their geometry finalised, the wall polygons in the wall graph must be updated to reflect the carved-out regions. This is the step that ensures zero overlap between class 1 (wall) and classes 2/3 (openings) in the mask.

###### 

###### 6.8.1 Carve-Out Operation



For each opening `O` with host segment `S`:

```python
opening\_polygon = Polygon(\[...])  # the axis-aligned opening rectangle
updated\_wall\_poly = S.wall\_polygon.difference(opening\_polygon)
updated\_wall\_poly = make\_valid(updated\_wall\_poly)

# Validate: result should be a Polygon (not MultiPolygon)
if updated\_wall\_poly.geom\_type != "Polygon":
    parts = \[g for g in get\_parts(updated\_wall\_poly) if g.geom\_type == "Polygon"]
    # Keep both parts: the wall segment is now split into two stubs
    S.wall\_stub\_a, S.wall\_stub\_b = sorted(parts, key=lambda p: p.bounds\[0])
    S.wall\_polygon = None  # signal to rasteriser to use stubs instead
else:
    S.wall\_polygon = updated\_wall\_poly
```

The `difference()` operation cleanly removes the opening rectangle from the wall polygon. For a wall segment with a single opening, the result is a `Polygon` with a rectangular notch. For a wall segment with two or more openings, repeated `difference()` calls may return a `MultiPolygon` (two wall stubs flanking the openings). The `MultiPolygon` handling above splits the result into stubs and stores them separately. The rasteriser in Chapter 11 checks for `wall\_polygon is None` and uses the stubs list instead.

###### 

###### 6.8.2 Opening Polygon Storage



Each opening is stored as an `Opening` dataclass:

```python
@dataclass
class Opening:
    opening\_id: int
    opening\_type: str                  # "window" | "door"
    host\_segment\_id: int               # wall graph edge ID
    opening\_polygon: shapely.Polygon   # the carved-out rectangle
    symbol\_geometry: dict              # image-only marks (arc, X, sill lines)
    door\_type: str | None              # "single\_swing" | "double\_swing" | "bti\_cross\_ridge"
    hinge\_side: str | None             # "left" | "right"
    swing\_direction: str | None        # "inward" | "outward"
    width\_px: float
    era: str
    mask\_class: int                    # 2 for window, 3 for door
```

The `symbol\_geometry` dictionary contains all drawing marks that go into the **image layer only**: swing arcs, BTI cross-ridge X diagonals, sill lines, glazing midlines. None of these marks are written to the mask. The mask receives only `opening\_polygon`, rasterised to `mask\_class`.

###### 

###### 6.8.3 Overlap Guard



After all carve-outs are complete, a global overlap check is run:

```python
all\_opening\_polygons = \[o.opening\_polygon for o in openings]
for i, oa in enumerate(all\_opening\_polygons):
    for ob in all\_opening\_polygons\[i+1:]:
        assert oa.intersection(ob).area < 0.5  # allow sub-pixel floating-point error
```

Any pair of openings with non-trivial overlap (> 0.5 px²) indicates a placement error. If this check fails, the offending opening is removed and logged in `metadata\["opening\_overlap\_removals"]`. This is expected to be rare (< 0.1% of samples) and is caused by the stochastic placement algorithm placing two openings whose clearance zones almost — but not quite — prevent overlap.

&#x20;

##### 6.9 Era-Specific Symbol Rendering



The `symbol\_geometry` field of each `Opening` is populated differently per era. Chapter 9 consumes these geometry descriptors to render the final image marks.

###### 

###### Soviet Era

* **Windows**: two-line sill convention (outer + inner only, no midline). Line weight heavier than digital era. Slight hand-drawn jitter on sill line length (±1.5px at each end).
* **Doors**: BTI cross-ridge dominant. X lines drawn with slight ink-pressure variation (thicker at centre, thinner at corners). No swing arc.

###### 

###### Scan Era

* **Windows**: three-line convention. Lines drawn precisely (no jitter). Slight scan blur applied in Chapter 10.
* **Doors**: swing-arc convention dominant. Arc drawn as a thin quarter-circle. Slight scan-induced discontinuity in the arc at probability 0.05 (models ink dropout on scan).

###### 

###### Digital Era

* **Windows**: three-line convention. Sub-pixel precise. Optional thin grey fill between sill lines (20% probability), modelling CAD-exported plans that shade the glass area.
* **Doors**: swing-arc convention. Perfect circular arc with no jitter. Hinge point marked with a small filled square (2×2px), modelling the common CAD door symbol convention.

&#x20;

##### 6.10 Mask Write Rules for Openings



Chapter 11 will document the full write-order protocol, but the opening-specific mask rules are defined here for completeness.

**Rule 1 — Polygon fill only.** Opening polygons are rasterised by polygon fill (`cv2.fillPoly` or equivalent). No stroke drawing. A stroked rectangle would paint the wall class (class 1) on the opening boundary pixels, contradicting the carve-out operation.

**Rule 2 — No symbol geometry in mask.** Swing arcs, BTI X lines, sill lines, and all other `symbol\_geometry` entries are image-only marks. They do not influence the mask.

**Rule 3 — Write-order: class 2 before class 3.** If a window and a door are adjacent on the same wall segment and their opening polygons share a boundary edge (zero-area intersection), writing class 2 first and class 3 second ensures that the shared boundary pixel is class 3 (door). This is consistent with the convention that a door jamb pixel, if ambiguous, is classified as door rather than window.

**Rule 4 — Openings override walls.** The mask write-order sequence is: room fills (class 0) → wall polygons (class 1) → window openings (class 2) → door openings (class 3). A pixel that belongs to both a wall polygon and an opening polygon receives the opening class. This is guaranteed by the carve-out (§6.8.1), but the write-order provides a second line of defence against sub-pixel polygon boundary drift.

&#x20;

##### 6.11 Validation and Output Contract



After all openings are placed and carved out, the opening system is validated:

```python
# Check 1: every habitable room has at least one window on an exterior wall
for room in rooms:
    if WINDOW\_ELIGIBILITY\_OVERRIDE.get(room.room\_type, True):
        exterior\_segs = \[s for s in room\_exterior\_segments(room) if s.is\_exterior]
        n\_windows\_on\_room = sum(
            1 for o in openings
            if o.opening\_type == "window"
            and o.host\_segment\_id in \[s.edge\_id for s in exterior\_segs]
        )
        if n\_windows\_on\_room == 0:
            metadata\["windowless\_habitable\_rooms"].append(room.room\_id)
            # Not a hard failure; logged only

# Check 2: every adjacent room pair has at least one connecting door
for room\_a, room\_b in adjacency\_pairs:
    connecting\_doors = \[
        o for o in openings
        if o.opening\_type == "door"
        and is\_on\_shared\_wall(o, room\_a, room\_b)
    ]
    if not connecting\_doors:
        raise OpeningValidationError(f"No door between rooms {room\_a.room\_id} and {room\_b.room\_id}")

# Check 3: no opening polygon is empty or invalid
for o in openings:
    assert o.opening\_polygon.is\_valid
    assert not o.opening\_polygon.is\_empty
    assert o.opening\_polygon.area > 0
```

Check 1 is a soft warning, not a hard failure. Windowless habitable rooms can occur when all exterior segments of a room are too short to receive a window — a geometric consequence of the BSP subdivision, not a logical error. The model must generalise to such cases (they exist in real drawings), so forcing every room to have a window would over-constrain the corpus.

Check 2 is a hard failure: a floor plan with disconnected rooms is not a valid residential layout and would produce confusing training samples.

The final output contract:

```python
@dataclass
class OpeningResult:
    openings: list\[Opening]             # all placed openings
    updated\_wall\_graph: WallGraph       # wall\_polygon fields updated by carve-outs
    window\_count: int
    door\_count: int
    windowless\_room\_ids: list\[int]      # soft warning list
    missing\_door\_pairs: list\[tuple]     # soft warning list (should be empty)
```

&#x20;

##### 6.12 Relationship to Downstream Chapters



|Chapter|Dependency on opening output|
|-|-|
|**7**|Icon placement avoids opening polygons; `Room.polygon.buffer(-wall\_thickness)` already excludes wall bands, but opening positions are additionally excluded from icon anchor candidates|
|**8**|Dimension arrows spanning rooms are not allowed to cross opening polygons; arrows are rerouted around openings|
|**9**|`Opening.symbol\_geometry` is consumed by the era renderer to draw arcs, X marks, and sill lines in the image|
|**10**|No special handling; opening symbols are already rasterised marks before degradation|
|**11**|`Opening.opening\_polygon` and `Opening.mask\_class` are the direct inputs to the mask polygon-fill step; write-order (class 2 before class 3, both after class 1) is enforced here|



Even standardized floor plan symbols can have slight variations from one architect or firm to another. This is precisely why the generator encodes three distinct symbol variants (§6.9) rather than a single canonical form — the downstream model must learn that a three-line rectangle in a wall, a two-line rectangle, and an X-filled rectangle all represent the same semantic class (window or door), regardless of which era-specific convention was used to draw them.





&#x20;

#### Chapter 7 — Icon Pack Integration



##### 7.1 The Role of Icons in the Five-Class Schema



Floor plan symbols are standardised icons and shapes used on architectural floor plans to represent structural elements, doors, windows, furniture, and mechanical systems. Common floor plan symbols include door symbols, window symbols, wall symbols, stair symbols, electrical symbols, plumbing symbols, HVAC symbols, and furniture symbols. In the five-class schema all of these non-structural marks — every drawn object that is neither a wall nor an opening — collapse into a single class: **class 4 (furniture / contents)**. The icon pack integration system is the mechanism by which those marks are introduced into the synthetic image in a way that is both visually realistic and mask-correct.

The icon system has three responsibilities that no previous chapter handles:

1. **Loading and caching** a directory tree of PNG assets at generator startup, validating each file and its optional sidecar metadata.
2. **Compositing** each icon onto the canvas using pre-multiplied alpha, so that icons drawn on coloured or off-white backgrounds do not produce halo artefacts.
3. **Writing the mask** correctly: every opaque or semi-opaque pixel of an icon's footprint must be written as class 4, regardless of what colour the icon pixel is in the image.

Floor plans might include outer and inner walls, windows, furniture, dimension lines, grids, text, or icons, alongside the constraints and relationships between them. The icon pack system is the mechanism that populates the furniture/icons layer in the generator's output, producing the visual complexity that differentiates real floor plans from bare room-outline diagrams.

&#x20;

##### 7.2 Folder Convention and Asset Structure



###### 7.2.1 Directory Layout



The icon pack is rooted at the path passed as `icon\_pack\_dir` in the top-level generator call (Chapter 1). The required layout is:

```
icons/
├── sanitary/
│   ├── bath\_left.png
│   ├── bath\_left.json          ← optional sidecar
│   ├── bath\_right.png
│   ├── toilet.png
│   ├── toilet.json
│   ├── sink\_round.png
│   └── sink\_double.png
├── kitchen/
│   ├── hob\_4burner.png
│   ├── hob\_4burner.json
│   ├── refrigerator.png
│   └── sink\_single.png
├── bedroom/
│   ├── bed\_single.png
│   ├── bed\_double.png
│   └── wardrobe.png
├── living/
│   ├── sofa\_2seat.png
│   ├── sofa\_3seat.png
│   ├── armchair.png
│   ├── table\_coffee.png
│   └── table\_dining.png
├── stair/
│   ├── stair\_straight.png
│   ├── stair\_straight.json
│   └── stair\_spiral.png
├── radiator/
│   ├── radiator\_short.png
│   └── radiator\_long.png
└── misc/
    ├── arrow\_up.png
    └── room\_number.png         ← template; numbers are rendered at runtime
```

The top-level subdirectory name is the **category key**. It maps directly to room types from Chapter 3's `ROOM\_TYPE\_WEIGHTS` (§3.3): a room of type `"bathroom"` draws from `sanitary/`; a room of type `"kitchen"` draws from both `kitchen/` and `radiator/`; a room of type `"living"` draws from `living/` and `radiator/`. The full mapping is defined in §7.5.

The category name is the only required organisational element. Filenames within a category are arbitrary but must be unique within that category. The generator references icons by `(category, stem)` pairs — e.g., `("sanitary", "bath\_left")` — not by full path.

###### 

###### 7.2.2 The PNG Asset Requirements



Every icon PNG must satisfy:

1. **RGBA format** (4 channels). A grayscale PNG with no alpha channel will be rejected at load time with a clear error message.
2. **Pre-multiplied alpha is not assumed in the stored file.** The PNG standard explicitly requires non-premultiplied alpha. Icons are stored in straight (unassociated) alpha. The generator converts to pre-multiplied alpha at load time (§7.3.1), not at compositing time.
3. **Consistent line style with the target era.** Icons for Soviet-era samples should have thicker, slightly irregular strokes; icons for digital-era samples should be pixel-clean. This is enforced by placing era-specific variants in subdirectories: `sanitary/soviet/bath\_left.png` vs. `sanitary/bath\_left.png`. If a Soviet-specific variant is absent, the base variant is used with a programmatic stroke-thickening step (§7.7.2).
4. **Transparent background.** All pixels not part of the icon drawing must have alpha = 0. A white or off-white background with alpha = 255 is not acceptable — it would occlude the room fill colour and produce a visible rectangular halo around every icon.

###### 

###### 7.2.3 The JSON Sidecar



Each PNG may have an accompanying `.json` sidecar with the same stem. The sidecar is optional; if absent, defaults are assumed for all fields.

```json
{
    "anchor": \[0.5, 1.0],
    "footprint": \[\[0.0, 0.8], \[1.0, 0.8], \[1.0, 1.0], \[0.0, 1.0]],
    "wall\_snap": "bottom",
    "min\_room\_area\_frac": 0.08,
    "allow\_rotation": \[0, 90, 180, 270],
    "era\_compatible": \["soviet", "scan", "digital"]
}
```



|Field|Type|Default|Description|
|-|-|-|-|
|`anchor`|`\[float, float]`|`\[0.5, 0.5]`|Normalised \[x, y] within the icon bounding box that is treated as the placement point. `\[0.5, 1.0]` = bottom-centre, used for wall-snapped items like baths and radiators.|
|`footprint`|`\[\[float, float], ...]`|Full bounding box|Normalised polygon defining the opaque footprint for collision detection and mask writing. If absent, the alpha channel's bounding box is used.|
|`wall\_snap`|`str|null`|`null`|
|`min\_room\_area\_frac`|`float`|`0.05`|Minimum room area as a fraction of canvas area for this icon to be eligible for placement. Large items like double beds are suppressed in small rooms.|
|`allow\_rotation`|`\[int, ...]`|`\[0, 90, 180, 270]`|Rotation angles in degrees that are valid for this icon. Asymmetric icons like baths have limited rotations; symmetric icons like radiators allow all four.|
|`era\_compatible`|`\[str, ...]`|All three eras|If an era is absent from this list, the icon is never placed in samples of that era.|

&#x20;

##### 7.3 Alpha Compositing with Pre-Multiplied Alpha

###### 

###### 7.3.1 Why Pre-Multiplication Is Required



In computer graphics, alpha compositing or alpha blending is the process of combining one image with a background to create the appearance of partial or full transparency. It is often useful to render picture elements (pixels) in separate passes or layers and then combine the resulting 2D images into a single, final image called the composite.

The halo problem arises specifically from straight-alpha compositing on non-white backgrounds. A floor plan generated in `soviet` or `scan` era has an off-white or yellowed background (Chapter 9). When a straight-alpha PNG icon is composited onto this background using the standard `OVER` operator without pre-multiplication, semi-transparent edge pixels blend between the icon's ink colour and the assumed white background of the icon file — not the actual canvas background. The result is a thin white fringe around the icon outline, which is visually incorrect and would cause the model to learn spurious high-frequency edge patterns around furniture icons.

The most significant advantage of premultiplied alpha is that it allows for correct blending, interpolation, and filtering. Knowing whether a file uses straight or premultiplied alpha is essential to correctly process or composite it, as a different calculation is required.

The fix is to convert straight-alpha PNGs to pre-multiplied alpha at load time and use the pre-multiplied `OVER` operator for all compositing.

###### 

###### 7.3.2 Load-Time Pre-Multiplication



```python
import numpy as np
from PIL import Image

def load\_icon\_premultiplied(path: Path) -> np.ndarray:
    """
    Loads a straight-alpha RGBA PNG and returns a float32 array
    with pre-multiplied alpha, shape (H, W, 4), values in \[0, 1].
    """
    img = Image.open(path).convert("RGBA")
    arr = np.array(img, dtype=np.float32) / 255.0   # straight alpha, \[0,1]

    # Pre-multiply: R, G, B channels are multiplied by alpha
    arr\[..., :3] \*= arr\[..., 3:4]

    return arr  # stored in icon cache; never written back to disk
```

With straight alpha, the RGB components represent the color of the object or pixel, disregarding its opacity. With premultiplied alpha, the RGB components represent the emission of the object or pixel, and the alpha represents the occlusion. After this conversion, every pixel in the cached array is in the pre-multiplied form `(R×α, G×α, B×α, α)`.

Another advantage of premultiplied alpha is performance; in certain situations, it can reduce the number of multiplication operations, for example if the image is used many times during later compositing. Since each icon may be placed multiple times per sample (e.g., multiple radiators), the load-time conversion amortises the multiplication cost over all placements.

###### 

###### 7.3.3 The OVER Compositing Operator



The pre-multiplied `OVER` operator composites a foreground layer `F` (the icon, pre-multiplied) over a background layer `B` (the canvas, in \[0, 1] float space):

```python
def composite\_over(canvas: np.ndarray,
                   icon\_pm: np.ndarray,
                   top\_left: tuple\[int, int]) -> np.ndarray:
    """
    canvas:   float32 (H, W, C) where C is 1 (monochrome) or 3 (RGB)
    icon\_pm:  float32 (ih, iw, 4) pre-multiplied RGBA
    top\_left: (row, col) pixel coordinate of icon top-left on canvas
    Returns:  updated canvas (same shape)
    """
    r0, c0 = top\_left
    ih, iw = icon\_pm.shape\[:2]

    # Clip to canvas bounds
    r1 = min(r0 + ih, canvas.shape\[0])
    c1 = min(c0 + iw, canvas.shape\[1])
    icon\_crop = icon\_pm\[:r1-r0, :c1-c0]

    alpha = icon\_crop\[..., 3:4]           # shape (h, w, 1)
    fg\_rgb = icon\_crop\[..., :3]           # already pre-multiplied

    if canvas.ndim == 2:
        # Monochrome canvas: convert fg to luminance
        fg\_rgb = 0.299\*fg\_rgb\[...,0] + 0.587\*fg\_rgb\[...,1] + 0.114\*fg\_rgb\[...,2]
        fg\_rgb = fg\_rgb\[..., np.newaxis]

    # OVER operator: result = fg + (1 - alpha) \* bg
    roi = canvas\[r0:r1, c0:c1]
    canvas\[r0:r1, c0:c1] = fg\_rgb + (1.0 - alpha) \* roi

    return canvas
```

The `OVER` operator in pre-multiplied form: `result = fg\_premult + (1 - alpha) × bg`. The Porter–Duff operations have a simple form only in premultiplied alpha. This is cleaner than the straight-alpha form which requires dividing out alpha to recover true colour, a division that is numerically unstable near `alpha = 0`.

###### 

###### 7.3.4 Monochrome Conversion for Non-Digital Eras



When the canvas is monochrome (the ≥70% case from Chapter 1), the icon's RGB channels must be collapsed to a single luminance value before compositing. The standard luminance formula above (0.299R + 0.587G + 0.114B) is used. For icons that are already monochrome line-drawings (black strokes on transparent background), all three channels are equal and the formula reduces to the channel value itself.

Icons drawn in colour (e.g., a red bathtub symbol from a digital-era pack) will appear as a grey smear on a monochrome canvas. This is intentional — it matches the appearance of colour-printed plans that have been scanned in grayscale, which is precisely the `scan` era condition. The resulting grey icons may be darker or lighter than the black ink lines of the walls, which is a realistic variation the model must handle.

&#x20;

##### 7.4 The Icon Cache



All icons are loaded at generator startup (not per-sample) and stored in a nested dictionary keyed by `(category, stem)`:

```python
@dataclass
class IconAsset:
    stem: str
    category: str
    image\_pm: np.ndarray           # pre-multiplied RGBA float32 (H, W, 4)
    alpha\_mask: np.ndarray         # uint8 (H, W), values 0 or 255 (binarised alpha)
    footprint\_polygon: shapely.Polygon  # from sidecar or derived from alpha\_mask
    anchor: tuple\[float, float]    # normalised
    wall\_snap: str | None
    min\_room\_area\_frac: float
    allow\_rotation: list\[int]
    era\_compatible: list\[str]
    natural\_size\_px: tuple\[int, int]  # (W, H) at 512×512 scale

class IconCache:
    assets: dict\[tuple\[str, str], IconAsset]  # (category, stem) → asset

    def get(self, category: str, era: str,
            rng: np.random.Generator) -> IconAsset | None:
        candidates = \[
            a for (cat, \_), a in self.assets.items()
            if cat == category and era in a.era\_compatible
        ]
        return rng.choice(candidates) if candidates else None
```

The `alpha\_mask` is a binarised version of the pre-multiplied alpha channel: pixels with `alpha > 0.05` are set to 255, all others to 0. This is used during mask writing (§7.8) and collision detection (§7.6.3). The threshold of 0.05 rather than 0.0 avoids marking near-invisible anti-aliasing fringe pixels as opaque.

The `footprint\_polygon` is computed from the sidecar's `footprint` field (de-normalised to pixel coordinates at the icon's `natural\_size\_px`) or, if no sidecar exists, from the bounding box of all `alpha > 0.05` pixels in the icon.

&#x20;

##### 7.5 Room-Type to Category Mapping



The room type from Chapter 3's subdivision result controls which icon categories are eligible for placement in each room. The mapping:

```python
ROOM\_ICON\_CATEGORIES = {
    "living":    \["living",  "radiator"],
    "bedroom":   \["bedroom", "radiator"],
    "kitchen":   \["kitchen", "radiator"],
    "bathroom":  \["sanitary"],
    "corridor":  \[],                      # no icons in corridors
    "utility":   \["misc"],
    "stairwell": \["stair"],
    "balcony":   \[],                      # no icons on balconies
}
```

Kitchen symbols represent important elements of a kitchen layout. They show items like wash basins, stoves, refrigerators, and cabinets. Bathroom symbols show the placement of fixtures like bathtubs, showers and toilets. Bedroom symbols mainly focus on furniture placement. The purpose of these symbols is to help visualize sleeping arrangements and storage solutions. Beds are drawn as simple rectangles to indicate their position and size within a room.

The `radiator` category appears in living rooms, bedrooms, and kitchens but not in bathrooms (where Soviet-era sanitary units include integrated heating) or corridors (where wall space is occupied by doors and no radiator symbol is expected).

The `stair` category is exclusively for rooms of type `stairwell`. Stair icons are composited once per stairwell room, occupying most of the room area. The stairs symbol on a floor plan is a series of parallel lines inside a rectangle, with an arrow pointing in the ascending direction (spiral stairs are shown as a circle with radiating lines). Stair symbols typically look like a series of attached rows or rectangles, usually with an arrow indicating which direction leads to a higher level. However, the shape may change depending on the type of staircase.

&#x20;

##### 7.6 Wall-Snap Placement Algorithm



Most floor plan icons are not placed arbitrarily within a room — they are aligned against walls. Interior design convention enforces that certain items rest flush against a wall; for freestanding objects such as dining tables the test is disabled. These three constraints collectively ensure that each furniture item remains legally within its designated room, allows for sufficient circulation space, and, where applicable, aligns neatly with a wall.

The wall-snap algorithm implements this convention. Icons with `wall\_snap != null` in their sidecar are placed by this algorithm; freestanding icons (dining tables, coffee tables) use the freestanding placement algorithm (§7.6.4).

###### 

###### 7.6.1 Candidate Wall Edge Enumeration



For a room polygon from `SubdivisionResult.rooms`, the eligible wall edges for snapping are:

```python
def get\_snap\_candidate\_edges(room: Room,
                              wall\_graph: WallGraph,
                              snap\_side: str,   # "bottom" = parallel to floor
                              icon: IconAsset) -> list\[LineString]:
    """
    Returns wall edges of the room that are:
    1. Long enough to accommodate the icon width + clearance
    2. Not occupied by an opening (door or window)
    3. Not already occupied by another wall-snapped icon
    4. On the correct side relative to the room interior (snap\_side)
    """
    min\_length = icon.natural\_size\_px\[0] + 2 \* ICON\_WALL\_CLEARANCE\_PX  # default: 8px
    candidates = \[]

    for edge\_id in wall\_graph.edges\_bounding\_room(room.room\_id):
        seg = wall\_graph.edges\[edge\_id]
        if seg.centreline.length < min\_length:
            continue
        if any(o.host\_segment\_id == edge\_id for o in openings):
            continue
        if edge\_is\_occupied\_by\_icon(edge\_id, placed\_icons):
            continue
        candidates.append(seg.centreline)

    return candidates
```

###### 

###### 7.6.2 Snap Position Computation



Once a candidate edge is selected (uniformly at random from the candidate list), the icon's anchor point is placed against the edge:

```python
def snap\_to\_wall(edge: LineString,
                 icon: IconAsset,
                 rotation\_deg: int,
                 rng: np.random.Generator) -> tuple\[int, int]:
    """
    Returns the (row, col) top-left pixel for placing the icon,
    with its anchor snapped to the wall edge.
    """
    # Rotate icon to the correct orientation
    rotated = rotate\_icon(icon, rotation\_deg)
    
    # Compute wall-normal direction (into room interior)
    wall\_dir = normalise(np.array(edge.coords\[-1]) - np.array(edge.coords\[0]))
    wall\_normal = np.array(\[-wall\_dir\[1], wall\_dir\[0]])  # 90° CCW
    
    # Sample a position along the edge (with clearance margins)
    t = rng.uniform(ICON\_EDGE\_MARGIN / edge.length,
                    1.0 - ICON\_EDGE\_MARGIN / edge.length)
    anchor\_world = np.array(edge.interpolate(t \* edge.length).coords\[0])
    
    # Offset anchor into the room by wall half-thickness + small gap
    wall\_seg = wall\_graph.edges\[edge\_id]
    offset = (wall\_seg.thickness / 2.0) + ICON\_WALL\_GAP\_PX  # default: 2px
    anchor\_world += offset \* wall\_normal
    
    # Convert anchor to top-left pixel using icon's normalised anchor field
    ah, aw = rotated.shape\[:2]
    ax\_frac, ay\_frac = icon.anchor
    top\_left = anchor\_world - np.array(\[ax\_frac \* aw, ay\_frac \* ah])
    
    return (int(round(top\_left\[1])), int(round(top\_left\[0])))  # (row, col)
```

The wall-normal direction is computed in pixel space (y-downward). The `offset` ensures that the icon body does not overlap the wall polygon — it sits flush against the wall's interior face, not embedded in the wall itself.

###### 

###### 7.6.3 Collision Detection



Before committing a placement, the icon's footprint polygon (transformed to canvas coordinates) is checked against:

1. **Room boundary**: `icon\_footprint.within(room.polygon.buffer(-WALL\_HALF\_THICKNESS))` — the icon must fit inside the inset room polygon.
2. **Previously placed icons**: `icon\_footprint.intersects(other\_icon\_footprint)` for each already-placed icon in this room. A collision-free buffer is guaranteed, where the obstruction set contains all walls, doors, windows and previously placed furniture.
3. **Opening clearance**: `icon\_footprint.distance(opening\_polygon) >= MIN\_OPENING\_CLEARANCE\_PX` (default: 10px). Icons must not block door swing arcs or window sill access.

If all three checks pass, the placement is committed. Otherwise, up to `MAX\_SNAP\_RETRIES = 5` alternate positions along the same edge are tried. If all retries fail, the algorithm moves to the next candidate edge. If all candidate edges fail, the icon is skipped for this room and the failure is logged in `metadata\["icon\_placement\_failures"]`.

Items that violate any criterion are incrementally nudged towards the nearest wall by the greedy algorithm described until feasibility is achieved. If a furniture item fails to meet any of these constraints at its initial position, a greedy wall-seeking algorithm is applied to incrementally adjust its placement until all conditions are satisfied. The generator uses a simplified version of this: rather than a continuous greedy nudge, it samples a discrete set of retry positions.

###### 

###### 7.6.4 Freestanding Placement Algorithm



Icons with `wall\_snap = null` (dining tables, coffee tables, spiral stairs) are placed at the room centroid with a random offset:

```python
def place\_freestanding(room: Room,
                        icon: IconAsset,
                        rng: np.random.Generator) -> tuple\[int, int] | None:
    centroid = np.array(room.polygon.centroid.coords\[0])
    
    for attempt in range(MAX\_FREESTANDING\_RETRIES):
        # Sample offset from centroid
        max\_offset = np.sqrt(room.polygon.area) \* 0.25
        offset = rng.uniform(-max\_offset, max\_offset, size=2)
        candidate\_anchor = centroid + offset
        
        # Compute top-left from anchor
        top\_left = candidate\_anchor - np.array(\[
            icon.anchor\[0] \* icon.natural\_size\_px\[0],
            icon.anchor\[1] \* icon.natural\_size\_px\[1]
        ])
        top\_left = (int(round(top\_left\[1])), int(round(top\_left\[0])))
        
        # Collision check
        icon\_fp = transform\_footprint(icon.footprint\_polygon, top\_left, rotation=0)
        if (icon\_fp.within(room.polygon.buffer(-WALL\_HALF\_THICKNESS))
                and not any(icon\_fp.intersects(p) for p in placed\_footprints)):
            return top\_left
    
    return None  # placement failed; icon skipped
```

&#x20;

##### 7.7 Icon Sizing and Rotation

###### 

###### 7.7.1 Scaling to Room Size



Icons are stored at a `natural\_size\_px` calibrated for a canonical room width of 200px at 512×512 output (corresponding to approximately 4 metres at 1:100 scale, a typical Soviet residential room). When the host room is larger or smaller, icons are scaled proportionally:

```python
room\_width\_approx = np.sqrt(room.polygon.area)  # rough linear dimension
scale = room\_width\_approx / CANONICAL\_ROOM\_WIDTH\_PX  # default: 200px
scale = np.clip(scale, MIN\_ICON\_SCALE, MAX\_ICON\_SCALE)  # default: \[0.5, 2.0]

new\_w = int(round(icon.natural\_size\_px\[0] \* scale))
new\_h = int(round(icon.natural\_size\_px\[1] \* scale))
scaled\_image = cv2.resize(icon.image\_pm, (new\_w, new\_h),
                           interpolation=cv2.INTER\_LINEAR)
# Re-binarise alpha after resize to avoid sub-pixel fringe
scaled\_image\[..., 3] = (scaled\_image\[..., 3] > 0.05).astype(np.float32)
```

The alpha channel is re-binarised after resize. Linear interpolation of pre-multiplied alpha introduces sub-pixel fringe values (e.g., `alpha = 0.03`) that would mark pixels as non-transparent but nearly invisible in the image, causing thin ghost rings in the mask. The re-binarisation step eliminates this by snapping all near-zero alpha values to exactly 0.

###### 

###### 7.7.2 Era-Specific Stroke Thickening



When no Soviet-specific icon variant exists in the pack, the base icon is programmatically thickened for `soviet` era samples:

```python
def thicken\_icon\_strokes(icon\_pm: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    Dilates the opaque regions of the icon by sigma pixels,
    simulating the heavier line weight of hand-drawn Soviet plan symbols.
    """
    alpha = icon\_pm\[..., 3]
    dilated\_alpha = cv2.dilate(alpha, 
                                cv2.getStructuringElement(cv2.MORPH\_ELLIPSE, (3, 3)),
                                iterations=1)
    result = icon\_pm.copy()
    result\[..., 3] = np.minimum(dilated\_alpha, 1.0)
    # Extend RGB channels to newly dilated pixels
    result\[..., :3] = np.where(result\[..., 3:4] > 0, 
                                result\[..., :3] / (result\[..., 3:4] + 1e-6),
                                0.0) \* result\[..., 3:4]
    return result
```

This is a morphological dilation of the alpha channel followed by re-multiplication of the RGB channels into the new alpha shape. The effect is a slightly thicker outline on the icon, consistent with the Soviet-era convention of heavier line weights for all drawn marks. Appliances and fixtures, such as toilets, sinks, and bathtubs, are drawn to scale with a thin line, and resemble the item they symbolize. Similarly, the furniture in floor plans is drawn with a light line weight so you can quickly tell that it is not integral to the building. In the Soviet era, the distinction between light and heavy line weight was less consistent — both wall lines and furniture marks tended toward the heavier end — and the dilation approximates this.

###### 

###### 7.7.3 Rotation



Icons are rotated from the `allow\_rotation` list in their sidecar. Rotation is applied to the pre-multiplied `image\_pm` array using `cv2.warpAffine` with `INTER\_LINEAR` interpolation and `BORDER\_CONSTANT` border mode (transparent padding):

```python
def rotate\_icon(icon: IconAsset, angle\_deg: int) -> np.ndarray:
    if angle\_deg == 0:
        return icon.image\_pm
    h, w = icon.image\_pm.shape\[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, -angle\_deg, 1.0)  # negative: CW in y-down
    # Compute new bounding box size
    cos, sin = abs(M\[0, 0]), abs(M\[0, 1])
    new\_w = int(h \* sin + w \* cos)
    new\_h = int(h \* cos + w \* sin)
    M\[0, 2] += (new\_w / 2.0) - centre\[0]
    M\[1, 2] += (new\_h / 2.0) - centre\[1]
    rotated = cv2.warpAffine(icon.image\_pm, M, (new\_w, new\_h),
                              flags=cv2.INTER\_LINEAR,
                              borderMode=cv2.BORDER\_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    return rotated
```

Only the four cardinal rotations (0°, 90°, 180°, 270°) are supported in this chapter. Non-cardinal rotations are not used because real floor plan icons are always axis-aligned; a rotated bath or toilet at 37° is never encountered in real drawings and would confuse the model.

&#x20;

##### 7.8 Mask-Writing Rule



The mask rule for icons is the simplest in the entire system, and its simplicity is deliberate:

**Every pixel of the icon's footprint that has `alpha > 0.05` after scaling and rotation is written as class 4 (value 255) in the mask, regardless of the pixel's colour in the image.**

This rule follows directly from the class 4 definition in Chapter 1: all non-structural interior content is class 4. It does not matter whether the pixel is a black ink line, a grey hatch mark, or a white interior fill — if it is part of an icon, it is class 4.

The implementation uses the binarised `alpha\_mask` from the icon cache (§7.4):

```python
def write\_icon\_to\_mask(mask: np.ndarray,
                        scaled\_rotated\_icon: IconAsset,
                        top\_left: tuple\[int, int]) -> None:
    """
    Writes class 4 (value 255) to all opaque icon pixels in the mask.
    """
    r0, c0 = top\_left
    alpha = scaled\_rotated\_icon\[..., 3]   # float32 \[0,1]
    opaque = (alpha > 0.05).astype(np.uint8) \* 255  # binarised

    ih, iw = opaque.shape
    r1 = min(r0 + ih, mask.shape\[0])
    c1 = min(c0 + iw, mask.shape\[1])

    icon\_crop = opaque\[:r1-r0, :c1-c0]
    # Only overwrite mask pixels where icon is opaque
    mask\_roi = mask\[r0:r1, c0:c1]
    mask\[r0:r1, c0:c1] = np.where(icon\_crop > 0, 255, mask\_roi)
```

The `np.where` condition ensures that **transparent icon pixels do not overwrite the underlying mask class**. A transparent pixel at the edge of a bath icon that happens to fall on a wall pixel (class 1) must remain class 1 in the mask. This is the critical difference from the image compositing step, where the `OVER` operator blends semi-transparent pixels — the mask uses a hard binary threshold.

###### 

###### 7.8.1 Non-Transparent Footprint Rule



The footprint polygon from the sidecar (or derived from the alpha mask) defines the full opaque body of the icon, which may be larger than the pixel-level alpha threshold would suggest. For example, a bath icon may have a narrow white gap at its centre (the bath interior) that has alpha = 0 in the image but should still be class 4 in the mask — because the bath occupies that floor space, even where it is visually white.

The resolution of this is through the `footprint` field in the sidecar:

```python
def write\_icon\_footprint\_to\_mask(mask: np.ndarray,
                                  footprint\_poly: shapely.Polygon,
                                  top\_left: tuple\[int, int]) -> None:
    """
    Fills the footprint polygon (not just opaque pixels) with class 4.
    Used when sidecar specifies an explicit footprint.
    """
    # Translate footprint to canvas coordinates
    translated = translate(footprint\_poly, 
                           xoff=top\_left\[1], yoff=top\_left\[0])
    # Rasterise polygon fill to mask
    pts = np.array(translated.exterior.coords, dtype=np.int32)
    cv2.fillPoly(mask, \[pts], color=255)
```

The sidecar's `footprint` is the authoritative mask-writing shape when present. The pixel-level alpha threshold is the fallback when no sidecar exists. The distinction matters for icons with white interiors (bath interior, toilet bowl interior) — the sidecar footprint ensures the full extent of the fixture is labelled class 4, not just its dark outline strokes.

&#x20;

##### 7.9 Per-Room Icon Count and Placement Order

###### 

###### 7.9.1 Icon Count Sampling



The number of icons placed per room is sampled once per room:

```python
ICON\_COUNT\_RANGE = {
    "living":   (2, 5),
    "bedroom":  (1, 3),
    "kitchen":  (2, 4),
    "bathroom": (1, 3),   # toilet + bath/shower + optional sink
    "stairwell":(1, 1),   # exactly one stair icon
    "utility":  (0, 1),
    "corridor": (0, 0),
    "balcony":  (0, 0),
}

n\_icons = rng.randint(low=count\_range\[0], high=count\_range\[1] + 1)
```

Synthetic floor plan datasets contain furniture classes as window, sofa, sink, table, door, tub, armchair, sink, and bed placed in various rooms, which helps in generating more realistic results. Datasets of 10,000 floor-plan images can contain around 300,000 furniture items of 16 classes. This density — roughly 30 items per image — is at the upper end of what the generator targets. The per-room count ranges above produce an average of approximately 8–12 icons per floor plan image at the default room count distribution, which is appropriate for the five-class schema where icon density need not be exhaustive.

###### 

###### 7.9.2 Placement Order Within a Room



Within each room, icons are placed in a fixed priority order to ensure that the most functionally important items get the best available wall space:

1. **Sanitary fixtures** (bath, shower, toilet) — placed first; always wall-snapped
2. **Kitchen appliances** (hob, sink, refrigerator) — placed second; always wall-snapped
3. **Beds** — placed third; always wall-snapped
4. **Radiators** — placed fourth on exterior walls preferentially
5. **Large furniture** (sofa, wardrobe, dining table) — placed fifth; may be freestanding
6. **Small furniture** (armchair, coffee table, side table) — placed last; freestanding

This order matches the architectural convention that fixed plumbing and appliance positions drive the room layout, with loose furniture filling in around them.

###### 

###### 7.9.3 Cross-Room Deduplication



A stair icon placed in a `stairwell` room occupies a large footprint. Its footprint polygon is added to a **global no-placement zone list** that is checked by all other rooms' placement algorithms. This prevents a large dining table from being placed in an adjacent room in a position that visually overlaps the stairwell icon when the two rooms share a wall with no depth between them.

&#x20;

##### 7.10 Radiator Placement



Radiators deserve separate treatment because their placement convention is highly era-specific and differs from all other icons.

Floor plan symbols include a wide range of symbols — from furniture to fixtures to HVAC. Radiators appear as the horizontal fin symbol — a rectangle with parallel interior lines — placed flat against exterior walls. They are the canonical HVAC mark in Soviet-era residential plans.

Radiator placement rules:

1. **Exterior walls only.** Radiators are placed exclusively on wall segments with `is\_exterior = True`. Interior partition walls never receive radiators.
2. **Below windows when present.** If an exterior wall segment has a window, the radiator is placed centred directly below the window opening:

```python
   radiator\_centre\_x = window\_opening\_centre\_x  # same horizontal position
   radiator\_centre\_y = window\_inner\_face\_y + RADIATOR\_GAP\_PX  # 4px below sill
   ```

3. **Standalone if no window.** If the exterior segment has no window, the radiator is placed at the segment midpoint with the standard wall-snap algorithm.
4. **One per exterior wall segment.** No segment receives more than one radiator.
5. **Era gating.** Radiators are placed in all three eras, but with different probabilities: `soviet = 0.95`, `scan = 0.75`, `digital = 0.50`. Soviet-era plans almost universally show radiators; digital-era exports often omit them.

&#x20;

##### 7.11 Recommended Open Packs for Bootstrapping



The generator requires an icon pack to be present at `icon\_pack\_dir`. For users building the corpus from scratch, the following publicly available sources provide suitable starting material. All require format conversion (SVG→PNG with transparent background at 512×512, then RGBA conversion) and in most cases need to be thematically filtered to remove icons that are not appropriate for residential floor plans (e.g., office chairs, server racks).



|Source|Format|Coverage|Notes|
|-|-|-|-|
|**Noun Project** (free tier)|SVG, PNG|Broad: sanitary, kitchen, furniture|Attribution required on free tier; varied styles need era-filtering|
|**Flaticon plan-furniture pack**|SVG, PNG|50 line icons covering beds, sofas, tables, appliances|Plan furniture free icon pack in lineal style. Available sources SVG, EPS, PSD, PNG files. Personal and commercial use. Lineal style is appropriate for digital and scan eras; needs stroke thickening for Soviet era|
|**Icons8 furniture floor plan set**|SVG, PNG|Comprehensive furniture symbols in multiple styles|Free furniture floor plan icons, logos, symbols in 50+ UI design styles. Available in PNG, SVG, GIF. Multi-style availability useful for era differentiation|
|**SESYD synthetic dataset symbols**|PNG|Furniture classes: sofa, sink, table, tub, armchair, bed|Contains 16 furniture classes including window, sofa, sink, table, door, tub, sink, armchair, and bed placed in various rooms. Already in floor plan top-down style; suitable for direct use|
|**Hand-drawn BTI symbols** (custom)|PNG|Radiators, Soviet-era sanitary marks|Must be hand-drawn or vectorised from real BTI scans; no public pack currently available. Minimum viable set: bath, toilet, sink, radiator, stair|



The minimum viable icon pack for producing visually convincing outputs consists of:

* 2 bath variants (left/right hand)
* 1 shower symbol
* 1 toilet
* 1–2 sink variants
* 1 hob
* 1 refrigerator
* 1 single bed + 1 double bed
* 1 wardrobe
* 1 sofa
* 1 radiator (short) + 1 radiator (long)
* 1 straight stair + 1 spiral stair

This set of 14–16 icons is sufficient to populate all room types in the `ROOM\_ICON\_CATEGORIES` mapping. A richer pack (50+ icons) is recommended for production use, as repeated icons in a large training corpus are a form of distribution collapse — the model sees the same radiator shape in every Soviet-era bedroom and may overfit to it.

&#x20;

##### 7.12 The Placed Icon Dataclass



Each successfully placed icon produces a `PlacedIcon` record:

```python
@dataclass
class PlacedIcon:
    icon\_stem: str                     # e.g. "bath\_left"
    category: str                      # e.g. "sanitary"
    room\_id: int                       # host room
    top\_left: tuple\[int, int]          # (row, col) on canvas
    rotation\_deg: int                  # 0 / 90 / 180 / 270
    scale: float                       # applied scale factor
    canvas\_footprint: shapely.Polygon  # footprint in canvas coordinates
    wall\_snap\_edge\_id: int | None      # wall graph edge ID if snapped
    era: str
    mask\_written: bool                 # True after mask write step
```

The `canvas\_footprint` field is the definitive mask region for this icon. It is stored in the `GeneratorOutput.metadata\["placed\_icons"]` list for debugging and for any downstream pipeline that needs to know the spatial extent of each icon.

&#x20;

##### 7.13 Relationship to Downstream Chapters



|Chapter|Dependency on icon output|
|-|-|
|**8**|Dimension arrows must not cross icon footprints; `PlacedIcon.canvas\_footprint` is added to the arrow routing exclusion set|
|**9**|Era-specific rendering is already applied to the icon image before compositing (stroke thickening in §7.7.2); Chapter 9 applies no additional style to icons|
|**10**|No special handling; icons are rasterised pixel values after compositing, indistinguishable from any other drawn mark for the augmentation pipeline|
|**11**|`PlacedIcon.canvas\_footprint` is the polygon-fill target for class 4; write-order places icon mask fills after wall class 1 fills but before any annotation marks|



The plan structure must satisfy high-level geometric, topologic, and semantic constraints; for example, doors are embedded within walls, generally composed of parallel lines, and walls define the perimeter of rooms, in which their label, furniture, and layout can define its usage. The icon placement system operationalises the last part of this observation: furniture and fixtures are placed within rooms in a room-type-appropriate manner, and their mask labels correctly reflect their class 4 status. The downstream model therefore learns both the spatial statistics of furniture placement — baths on exterior walls, sofas freestanding in living rooms — and the correct semantic class for all such objects.





#### Chapter 8 — Dimension Annotations

##### 8.1 Purpose and Scope

Dimension annotations are the numeric measurements and associated graphical apparatus that appear on every floor plan of every era. They are, visually, one of the most information-dense elements on the page — and, from the segmentation model's perspective, one of the most dangerous sources of false activation. A dimension line crosses a wall at a right angle; its arrowheads can resemble door swing arcs; its numeric label floats in what should be an empty background region. Without explicit treatment, a model trained on annotation-free synthetic data will fail systematically on real drawings that carry heavy dimensioning.

This chapter specifies the geometry, placement algorithm, per-era rendering, and — critically — the mask assignment for all dimension annotation elements. The governing rule, established in §1.2.1, is unambiguous: **all dimension annotation elements are class 0 (background), without exception.** This chapter confirms that rule, documents its implementation in the rasterisation pipeline, and explains the reasoning that makes it the only defensible choice.

The chapter covers the following annotation sub-types:

|Sub-type|Description|
|-|-|
|**Linear dimension**|A pair of extension lines, a dimension line with terminal arrowheads, and a numeric label giving the measured distance|
|**Running dimension**|A chain of consecutive linear dimensions sharing a common baseline|
|**Leader annotation**|An angled leader line with a single arrowhead pointing to a feature and a text label at its tail|
|**Stair direction arrow**|A single-headed arrow inside a stair polygon indicating the direction of travel; rendered as a furniture element (§1.2.4) and **not** covered by this chapter|

The stair direction arrow distinction is noted here because it is visually similar to a leader annotation but belongs to class 4, not class 0. The criterion is containment: any arrow whose head and tail are both strictly interior to the building footprint is class 4; any arrow that extends to or beyond a wall line, or that floats entirely in the paper margin, is class 0.

##### 8.2 Annotation Geometry Model

###### 8.2.1 The Linear Dimension Unit

A single linear dimension is the atomic unit from which all other annotation forms are composed. It consists of four geometric sub-elements:

1. **The measured segment** — the wall edge, opening span, or room dimension being annotated. This segment is always one of the edges produced by the wall graph (Chapter 5) or an interior room span produced by the BSP subdivision (Chapter 3). The generator never annotates diagonal walls (see §8.4.3).
2. **Extension lines** — two short line segments drawn perpendicular to the measured edge, one at each endpoint. Each extension line begins at an offset of `ext\_gap` pixels from the annotated feature and extends outward by `ext\_length` pixels. Default values: `ext\_gap = 4px`, `ext\_length = 14px` at 512×512.
3. **The dimension line** — a line segment parallel to the measured edge, drawn at a standoff distance of `standoff` pixels from the nearest wall face. The standoff is measured perpendicular to the annotated edge. The dimension line connects the outer tips of the two extension lines. Default `standoff = 18px`. Running dimensions share a single baseline at a fixed standoff; their individual dimension lines are the sub-segments between consecutive tick marks.
4. **Terminal symbols** — drawn at both ends of the dimension line. Three terminal styles are supported, selected per era (§8.5):

|Style ID|Description|Era(s)|
|-|-|-|
|`ARROW\_FILLED`|Solid filled triangular arrowhead, pointing inward along the dimension line|`scan`, `digital`|
|`ARROW\_OPEN`|Two-stroke open arrowhead (chevron), no fill|`scan`|
|`TICK\_OBLIQUE`|A short diagonal slash at ≈45° to the dimension line|`soviet`|
|`DOT`|A small filled circle of radius 2–3px|`digital` (rare, ≤10%)|

5. **Numeric label** — a text string of the form `N,NN m` (e.g., `4,82 m`) for metric plans or `N'-N"` for imperial. The generator produces metric labels exclusively. The label is centred on the dimension line, drawn directly over it, with a small white knockout rectangle behind the text to ensure legibility over crossed lines. Label font, size, and style are governed by §8.5.

###### 8.2.2 Arrow Geometry: Filled Arrowhead

The `ARROW\_FILLED` terminal is the most common style and receives the most complete specification because its triangular silhouette is the shape most likely to confuse the segmentation model if not correctly masked.

The arrowhead is defined as an isoceles triangle with:

* **Tip** at the exact terminus of the dimension line, pointing inward (toward the opposite terminal)
* **Base width** `w\_arrow` pixels, perpendicular to the line direction
* **Length** `l\_arrow` pixels along the line direction

```python
def filled\_arrowhead(tip: np.ndarray, direction: np.ndarray,
                     w\_arrow: float, l\_arrow: float) -> Polygon:
    """
    tip       : (2,) array, pixel coordinates of the arrowhead point
    direction : unit vector pointing FROM the tip TOWARD the line interior
    """
    perp = np.array(\[-direction\[1], direction\[0]])
    base\_centre = tip + direction \* l\_arrow
    left  = base\_centre + perp \* (w\_arrow / 2)
    right = base\_centre - perp \* (w\_arrow / 2)
    return Polygon(\[tip, left, right])
```

Default dimensions at 512×512: `w\_arrow = 5px`, `l\_arrow = 9px`. At 1024×1024 these scale linearly. The arrowhead polygon is rendered as a solid fill using the foreground ink colour (§8.5). Its rasterised pixels are masked as class 0.

###### 8.2.3 Arrow Geometry: Open Arrowhead

The `ARROW\_OPEN` terminal is drawn as two line segments meeting at the tip, each `l\_arrow` pixels long and diverging at half-angle `θ` from the line axis. Default `θ = 22°`, `l\_arrow = 8px`, stroke width 1–2px depending on era line weight. No fill; no closed polygon. Pixels are still masked as class 0.

###### 8.2.4 Arrow Geometry: Oblique Tick

The `TICK\_OBLIQUE` terminal — standard in Soviet and Central European hand-drawn conventions — is a single stroke of length `tick\_len` pixels, centred on the dimension line endpoint, drawn at a fixed angle of 45° relative to the line. Default `tick\_len = 10px`. At `soviet` era this is drawn with the same variable-width stroke engine used for wall lines (§8.5.3). Pixels are masked as class 0.

##### 8.3 Placement Algorithm

###### 8.3.1 Candidate Segment Selection

Dimension annotations are placed on a subset of wall segments and interior spans produced by the wall graph and BSP subdivision respectively. The placement algorithm operates in two phases:

**Phase 1 — exterior edge dimensioning.** All exterior edges of the footprint polygon (those whose outward normal points away from the building interior) are eligible for a linear dimension. Each exterior edge is processed independently. The probability of annotating a given exterior edge is:

```python
p\_annotate\_exterior = clip(0.35 + 0.45 \* annotation\_density, 0.10, 0.95)
```

where `annotation\_density` is a per-image scalar sampled from `Uniform(0.0, 1.0)` at generation time and stored in `metadata\["annotation\_density"]`.

**Phase 2 — interior span dimensioning.** For each room produced by the BSP, sample 0–2 interior spans along the room's major and minor axes. Interior spans are axis-aligned segments connecting opposite walls at the room centroid. These are placed at a standoff toward the room interior and annotated with the room's pixel-space width or height converted to the plan's declared scale.

Running dimensions are used when three or more collinear exterior edges share a common outer face (common in T and U primitives). The algorithm detects collinear edge sequences by testing parallelism within 1° and shared outward normal within 2°.

###### 8.3.2 Standoff Side Selection

For exterior dimensions, the dimension line is always placed *outside* the building footprint — on the paper margin side. The standoff is measured from the outer wall face. For interior dimensions, the dimension line is placed inside the room, offset from the wall face toward the centroid.

The exterior standoff must not cause the dimension line to exceed the canvas boundary. If `wall\_outer\_edge + standoff > canvas\_width - canvas\_margin`, the dimension is suppressed for that edge. This silently skips annotations on footprints that are placed very close to the canvas edge (can occur at low `canvas\_margin` settings).

###### 8.3.3 Label Value Computation

The numeric label value is derived from the pixel-space segment length and the plan scale stored in `metadata\["scale\_m\_per\_px"]`. The scale is set during the footprint generation (Chapter 2) by dividing an assumed real-world building width (sampled from `Uniform(8.0, 22.0)` metres) by the pixel width of the footprint bounding box:

```python
scale\_m\_per\_px = assumed\_width\_m / footprint.bounding\_box\[2]  # bb width in px
label\_value\_m  = segment\_length\_px \* scale\_m\_per\_px
label\_str      = f"{label\_value\_m:.2f}".replace(".", ",") + " m"
```

The comma-decimal format (`4,82 m`) is standard in Soviet and European floor plan conventions. Labels shorter than `0.20 m` or longer than `99.99 m` are replaced with a placeholder `· · ·` to prevent physically implausible annotations.

###### 8.3.4 Collision Avoidance

After all dimension elements for an image have been computed, a collision pass removes any label whose bounding box overlaps with any other label or with a placed icon (Chapter 7). The pass is greedy and processes labels in order of decreasing segment length, favouring the annotation of longer spans. Suppressed labels remove only their text; the extension lines and dimension line are retained.

##### 8.4 Constraints and Special Cases

###### 8.4.1 Minimum Annotatable Length

Segments shorter than `MIN\_ANNOTATION\_LENGTH\_PX = 30px` at 512×512 are never annotated. Below this threshold, the label cannot be legibly rendered at the minimum font size and the arrowheads would overlap. The threshold scales linearly with `image\_size`.

###### 8.4.2 Running Dimension Chain Interruption

A running dimension chain along a collinear edge sequence is interrupted if any segment in the chain is shorter than `MIN\_ANNOTATION\_LENGTH\_PX`. The chain is split at the short segment; each sub-chain of length ≥ 2 is rendered as an independent running dimension. Chains of length 1 fall back to a single linear dimension.

###### 8.4.3 No Annotations on Diagonal Walls

Diagonal wall segments (Chapter 4) are never annotated. The standoff direction for a diagonal edge is not axis-aligned, which makes label positioning ambiguous and visually inconsistent with the orthogonal annotation grid. This is a hard exclusion: the diagonal wall placement pass marks all diagonal edges with a `no\_annotate` flag before the annotation algorithm runs. Chapter 4 sets this flag.

###### 8.4.4 Opening Span Annotations

Door and window openings (Chapter 6) may be annotated with their clear opening width. This is controlled by `annotate\_openings: bool` (default `True` for `scan` era, `False` for `digital` and `soviet`). When enabled, the opening's clear width is annotated with a dimension line placed in the same standoff band as the containing wall's exterior dimension. The terminal style matches the era. Opening annotations always use `ARROW\_FILLED` regardless of the wall-level terminal style; this is a minor deliberate inconsistency that matches observed behaviour in real scan-era documents.

##### 8.5 Per-Era Rendering

###### 8.5.1 Era 1 — Post-2000 Scan

* **Terminal style**: `ARROW\_FILLED` (95%) or `ARROW\_OPEN` (5%)
* **Stroke width**: 1px for extension lines and dimension line; arrowhead fill solid
* **Font**: a condensed sans-serif (e.g., Liberation Sans Narrow or a metric-equivalent synthetic font); size 7–9px at 512×512
* **Ink colour**: same foreground ink as the wall lines (typically near-black, slightly brownish due to scan yellowing applied in Chapter 10)
* **Label knockout**: a filled white rectangle of `label\_width + 4px` × `label\_height + 2px`, drawn behind the label, centred on the dimension line
* **Annotation density**: sampled from the per-image scalar; typically moderate (0.3–0.7)

###### 8.5.2 Era 2 — Modern Digital

* **Terminal style**: `ARROW\_FILLED` (85%) or `DOT` (15%)
* **Stroke width**: 0.5–1px; arrowheads smaller and sharper than scan era
* **Font**: a pixel-perfect monospace or architectural lettering font; size 6–8px
* **Ink colour**: pure black (`#000000`) with no yellowing
* **Label knockout**: omitted; labels are rendered on a clean white background and do not need a separate knockout
* **Annotation density**: generally lower than scan era (0.1–0.5); `digital` plans frequently omit interior span dimensions

###### 8.5.3 Era 3 — Soviet Hand-Drawn

* **Terminal style**: `TICK\_OBLIQUE` exclusively
* **Stroke width**: 1.5–2.5px with variable-width modulation (see §9.4.2 for the stroke engine); the oblique tick uses the same ink-pressure model as wall lines
* **Font**: a hand-lettered technical font, either Cyrillic or Latin, with per-character baseline jitter `±1px` and size jitter `±1px`; size 8–11px
* **Ink colour**: slightly faded, drawn from the era's ink degradation palette (§9.4.3)
* **Label knockout**: none; Soviet drawings do not interrupt the dimension line for the label — the text floats above or below it
* **Annotation density**: high; Soviet BTI drawings are typically heavily dimensioned, with nearly every exterior edge and most interior room spans annotated; `annotation\_density` is biased toward `Uniform(0.6, 1.0)` for `soviet` era

##### 8.6 Mask Assignment

All dimension annotation elements — extension lines, dimension lines, arrowheads (in every terminal style), numeric labels, leader lines — are rasterised directly to **class 0 (background, pixel value `0`)** in the mask. This is the same class as the paper margin and all other non-plan elements.

The implementation enforces this via write order. The mask writing pipeline (Chapter 11) processes elements in a fixed priority stack:

```
Priority 1 (lowest): Background fill — class 0
Priority 2: Footprint interior fill — class 4 (later overwritten by rooms)
Priority 3: Room fills — class 4 → class 0 (background exterior)
Priority 4: Wall polygons — class 1
Priority 5: Opening carve-outs — class 2 and 3
Priority 6: Icons and furniture — class 4
Priority 7 (highest): Dimension annotations — class 0
```

By writing dimension annotations at the highest priority, they overwrite any underlying wall, room, or icon pixels in the mask. This is intentional and correct: a dimension line crossing a wall is annotated as background in both the image *and* the mask, teaching the model to treat annotation clutter as transparent.

**The critical implication**: dimension annotations are drawn *after* all structural elements in the mask pipeline, even though in the image they are drawn at a consistent layer that may be behind or in front of icons depending on era convention. The image layer ordering and the mask write order are permitted to differ, provided they produce consistent image–mask pairs. Chapter 11 documents the full write-order table and the layer separation mechanism.

##### 8.7 Implementation Notes

**Label bounding-box registration.** Every rendered label must register its bounding box in `metadata\["annotation\_bboxes"]` as a list of `\[x, y, w, h]` records. This enables the YOLO label generator (§1.3.2) to suppress detection annotations from overlapping regions and allows Chapter 10's degradation pass to apply blur and ink-spread consistently across text.

**Font availability at runtime.** The generator should not depend on system font installation. All era-appropriate fonts must be bundled in the `icon\_pack\_dir` tree under `fonts/<era>/`. The rendering path falls back to PIL's default font only if the bundled font is missing, and logs a warning; the fallback font is visually inconsistent and should not appear in production-quality outputs.

**Pixel-level arrowhead rendering.** At 512×512 the arrowhead polygons are small (≈5×9px). Rasterising them with `PIL.ImageDraw.polygon()` using integer-pixel vertices produces jagged edges consistent with scan-era quality. At `digital` era, anti-aliased rendering via a 4× supersample-and-downsample pass may be applied if `antialiased\_annotations: bool = True` (default `False`). Anti-aliasing is never applied at `soviet` era.

**Zero-annotation images.** Approximately 8% of generated images carry no dimension annotations at all (controlled by `p\_no\_annotations = 0.08`). This matches the real-world prevalence of floor plan images where annotations have been stripped for publishing. The `metadata\["annotation\_count"]` field will be `0` for these images.

\---

#### Chapter 9 — Era Aesthetic Themes

##### 9.1 Purpose of the Theme System

The three eras defined in §1.4 are not merely statistical sampling weights. Each era represents a distinct *visual regime* that governs how every drawn element in the image is rendered: the character of ink strokes, the quality of line junctions, the background paper texture, the typographic conventions, and the spatial density of drawn content. A generator that applies era labels only to the augmentation pass (Chapter 10) while rendering all structural elements identically will produce outputs that look superficially aged but are structurally uniform — and a model trained on them will learn to ignore degradation rather than generalise through it.

This chapter specifies the **era theme system**: the set of rendering parameters, stroke models, palette constraints, and typographic rules that are bound to each era token and applied during the geometry rendering passes of Chapters 5, 6, 7, and 8. The theme system is implemented as a plugin interface (§9.5) so that new eras or era sub-variants can be registered without modifying core rendering code.

The era token is set once per image during the top-level `generate\_sample()` call (§1.3.1) and propagates as a read-only parameter to every downstream rendering function. No rendering function selects its own era. The constraint from §1.7 is absolute: **era sampling governs aesthetics end-to-end.**

##### 9.2 The Era Theme Dataclass

Each era is represented as a populated instance of `EraTheme`:

```python
@dataclass
class EraTheme:
    era\_id: str                    # "scan" | "digital" | "soviet"

    # Line rendering
    line\_weight\_range: tuple\[float, float]  # (min\_px, max\_px) stroke width
    line\_weight\_jitter: float               # per-segment std dev, in px
    stroke\_model: Literal\["solid", "variable", "hand"]
    overshoot\_px: float                     # mean corner overshoot
    undershoot\_px: float                    # mean corner undershoot (gap)

    # Fill rendering
    wall\_fill\_style: Literal\["solid", "hollow", "hatch"]
    wall\_fill\_prob: dict\[str, float]        # {"solid": p, "hollow": p, "hatch": p}
    hatch\_angle\_deg: float                  # degrees, used when style == "hatch"
    hatch\_spacing\_px: float

    # Background
    bg\_colour\_range: tuple\[int, int]        # (min, max) grayscale value
    bg\_texture: Literal\["none", "grain", "laid", "wove"]
    bg\_texture\_strength: float              # 0.0–1.0

    # Ink
    ink\_colour\_range: tuple\[int, int]       # (min, max) grayscale; 0 = black
    ink\_fade\_prob: float                    # probability of faded ink on any segment
    ink\_bleed\_radius: float                 # gaussian blur radius for bleed, px

    # Typography
    font\_family: str                        # path relative to icon\_pack\_dir/fonts/
    font\_size\_range: tuple\[int, int]        # (min\_pt, max\_pt)
    char\_jitter\_px: float                   # per-character baseline jitter
    char\_angle\_jitter\_deg: float

    # Colour (only active when monochrome\_prob roll fails)
    colour\_mode: Literal\["none", "tint", "full"]
    room\_tint\_palette: list\[tuple\[int,int,int]]  # RGB; empty if colour\_mode == "none"

    # Annotation
    dimension\_terminal: str                 # see §8.5
    annotate\_openings: bool
    annotation\_density\_range: tuple\[float, float]
```

An `EraTheme` instance is immutable after construction. Downstream rendering functions receive it as a keyword argument `theme: EraTheme` and must not modify it.

##### 9.3 Era 1 — Post-2000 Scan Theme

The scan era represents the output of a modern CAD or vector drawing application, printed on a laser or inkjet printer and subsequently scanned at 200–400 dpi. The defining property is **structural cleanliness degraded by reproduction**: lines are originally perfect, but the print-scan round trip introduces noise that is spatially correlated at the scan resolution.

###### 9.3.1 Stroke Model: Solid

Scan-era lines are rendered as geometrically exact strokes of uniform width within a segment, with no within-segment weight variation. The `stroke\_model` is `"solid"`. Width is sampled once per wall segment from `Uniform(line\_weight\_range)`, typically `(1.5, 2.5)px` at 512×512, and held constant for that segment's entire length.

Corner treatment: exterior corners of wall polygons are rendered with a **butt join** — no overshoot, no rounding. Interior T-junctions between walls receive a **miter join** that is clipped at `miter\_limit = 2.5px` to prevent spikes at acute angles. These join styles match the default output of common CAD export pipelines.

###### 9.3.2 Wall Fill Style

```python
wall\_fill\_prob = {"solid": 0.78, "hollow": 0.20, "hatch": 0.02}
```

Solid fill — the wall polygon interior is painted with near-black ink — dominates in the post-2000 era. Hollow walls, which were common before solid-fill became the CAD default, persist in approximately 20% of samples to model legacy drawings that have been scanned. Hatched walls are rare and limited to structural notation conventions that occasionally appear in building permits.

When `hollow` is selected, the wall polygon is rendered as a closed outline of width `line\_weight + 0.5px` with no interior fill. The mask protocol from §1.2.2 applies: both the outline and the interior gap are labelled class 1. The hollow interior gap carries no foreground ink, but its mask pixels are class 1. See Chapter 11 for the polygon-fill implementation of this rule.

###### 9.3.3 Background

```python
bg\_colour\_range = (240, 255)       # near-white to pure white
bg\_texture      = "grain"          # ISO 400 film-grain model
bg\_texture\_strength = 0.08         # subtle; most noise is added by Ch. 10 degradation
```

The grain texture is applied as additive Gaussian noise with `σ = bg\_texture\_strength × 30` to the background canvas before any geometry is drawn. This ensures that the noise is present beneath wall fills, not layered on top — matching the physical reality of a textured paper surface.

###### 9.3.4 Ink Palette

```python
ink\_colour\_range = (0, 35)         # pure black to very dark grey
ink\_fade\_prob    = 0.03            # 3% of segments have slightly faded ink
ink\_bleed\_radius = 0.0             # no bleed; added by Ch. 10
```

At the scan theme level, ink is nearly perfect. The visible degradation — toner spread, scan blur, JPEG ringing — is entirely the responsibility of the Chapter 10 degradation pass. The theme's `ink\_bleed\_radius = 0.0` ensures no pre-degradation bleed.

###### 9.3.5 Typography and Annotations

```python
font\_family            = "fonts/scan/LiberationSansNarrow.ttf"
font\_size\_range        = (7, 9)
char\_jitter\_px         = 0.0
char\_angle\_jitter\_deg  = 0.0
dimension\_terminal     = "ARROW\_FILLED"
annotate\_openings      = True
annotation\_density\_range = (0.3, 0.7)
```

Text is computer-generated and geometrically perfect at the theme level. Post-rendering, Chapter 10 may apply scan-blur that softens the letterforms consistently with the rest of the image.

###### 9.3.6 Colour Mode

When the `monochrome\_prob` roll permits a colour image (≤30% of scan-era samples), the scan theme uses `colour\_mode = "tint"`: a single low-saturation tint colour is applied uniformly to the entire image to simulate yellowed or sepia-toned paper stock. This is distinct from the `digital` era's room-fill tints, which are per-room.

```python
colour\_mode        = "tint"
room\_tint\_palette  = \[]   # not used; tint is applied image-wide in Ch. 10
```

The tint transformation is: convert the grayscale image to RGB, then bias the red and green channels by `+r\_shift` and the blue channel by `-b\_shift`, where `r\_shift \~ Uniform(5, 18)` and `b\_shift \~ Uniform(3, 10)`. This produces the characteristic warm-yellow cast of aged paper.

##### 9.4 Era 2 — Modern Digital Theme

The digital era represents the direct export of a contemporary architectural CAD or BIM application — no print, no scan. The defining property is **pixel-level precision**: lines land on exact pixel boundaries, fills are uniform, and no degradation has occurred.

###### 9.4.1 Stroke Model: Solid (Thin)

```python
line\_weight\_range = (0.75, 1.5)   # thinner than scan era
line\_weight\_jitter = 0.0
stroke\_model      = "solid"
overshoot\_px      = 0.0
undershoot\_px     = 0.0
```

Digital lines are rendered at sub-pixel widths when anti-aliasing is enabled (`antialiased\_annotations = True`). At 512×512, walls in the digital era appear lighter than scan-era walls. Line joins are exact miter joins with no clipping required at the default `aggression` values.

###### 9.4.2 Wall Fill Style

```python
wall\_fill\_prob = {"solid": 0.60, "hollow": 0.38, "hatch": 0.02}
```

The hollow-wall proportion is higher in the digital era than in the scan era because contemporary CAD tools often export hollow (unfilled) wall representations as the default, relying on the enclosing polygon to imply solidity.

###### 9.4.3 Colour Mode

The digital era is the only era that produces per-room colour tints:

```python
colour\_mode = "tint"
room\_tint\_palette = \[
    (230, 240, 250),  # light blue — bedroom
    (250, 245, 225),  # warm cream — living room
    (220, 240, 225),  # light green — kitchen/utility
    (245, 235, 250),  # light lavender — bathroom
    (240, 240, 240),  # neutral grey — corridor
]
```

When colour mode is active, each room produced by the BSP subdivision is assigned a palette entry at random (with replacement). The tint is applied as a flood fill inside the room polygon before wall lines are drawn. When `monochrome\_prob` forces grayscale output, the palette is ignored and rooms receive a white fill.

###### 9.4.4 Background

```python
bg\_colour\_range     = (255, 255)   # pure white, no variation
bg\_texture          = "none"
bg\_texture\_strength = 0.0
```

Digital exports have a perfectly uniform white background. Chapter 10's degradation pass is configured to apply near-zero noise to digital-era images (`augmentation\_preset` defaults to `"clean"` for this era).

##### 9.5 Era 3 — Soviet Hand-Drawn Theme

The Soviet hand-drawn theme is the most technically complex of the three eras. It models manual drafting with technical pens or ruling pens on cartographic or architectural drafting paper, produced according to the conventions of the BTI and related Soviet technical drawing standards. The visual result is distinctive and requires a dedicated stroke rendering engine.

###### 9.5.1 Stroke Model: Variable-Width Hand Engine

```python
stroke\_model       = "hand"
line\_weight\_range  = (1.8, 3.5)
line\_weight\_jitter = 0.4
overshoot\_px       = 2.5
undershoot\_px      = 0.8
```

The `"hand"` stroke model renders each line segment not as a single constant-width rectangle but as a **variable-width ribbon** whose width oscillates along the segment length according to:

```python
def hand\_width\_profile(length\_px: int, base\_weight: float,
                        jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Returns a width value at each pixel along the segment."""
    noise = rng.normal(0, jitter, size=length\_px)
    # Low-frequency drift: slow pressure variation over the stroke
    drift\_freq = rng.uniform(0.005, 0.02)  # cycles per pixel
    drift\_amp  = rng.uniform(0.0, jitter \* 1.5)
    drift      = drift\_amp \* np.sin(2 \* np.pi \* drift\_freq \* np.arange(length\_px))
    return np.clip(base\_weight + noise + drift, 0.5, base\_weight \* 2.2)
```

The noise and drift components model two distinct physical phenomena: high-frequency noise models hand tremor, and low-frequency drift models pen pressure variation as the drafter repositions their hand along a ruler. The result is a stroke that visibly varies in weight along its length, consistent with observed Soviet BTI drawings.

###### 9.5.2 Corner Treatment: Overshoot and Undershoot

```python
overshoot\_px  = 2.5   # mean; sampled from Truncated Normal(μ=2.5, σ=1.0, min=0)
undershoot\_px = 0.8
```

At each wall junction, the generator independently samples whether the arriving stroke overshoots (extends past the corner point) or undershoots (stops short of it). The probability of overshoot is `0.65`; undershoot `0.20`; exact meeting `0.15`. These probabilities are applied per-stroke-end, so a single junction may have one overshooting stroke and one undershooting stroke — as is frequently observed in real BTI drawings.

Overshoot and undershoot apply only to the image rendering; the wall polygon geometry and therefore the mask are unaffected. The mask always represents the idealised wall boundary regardless of ink overshoot.

###### 9.5.3 Wall Fill Style

```python
wall\_fill\_prob = {"solid": 0.30, "hollow": 0.40, "hatch": 0.30}
```

Soviet-era wall fills show the highest diversity of the three eras. Solid fill (inked solid using a brush or wide ruling pen) is common in later BTI documents; hollow walls (outline only) are common in earlier and more technical drawings; and diagonal hatch fill is common in Stalinist-era structural drawings and in drawings that distinguish load-bearing from partition walls.

Hatch fill parameters for Soviet era:

```python
hatch\_angle\_deg   = 45.0   # standard cross-hatch is 45° and 135°
hatch\_spacing\_px  = 4.0    # at 512×512; scales with image\_size
```

Cross-hatch (both 45° and 135° sets of lines) is applied to 40% of hatch-fill walls; single-direction hatch is applied to the remaining 60%. Hatch lines are drawn using the hand stroke engine with reduced jitter (`line\_weight\_jitter \* 0.5`) to match the finer nib typically used for hatching.

###### 9.5.4 Background

```python
bg\_colour\_range     = (195, 245)   # yellowed to near-white
bg\_texture          = "laid"       # laid paper (chain lines visible)
bg\_texture\_strength = 0.18
```

Laid paper has a characteristic grid of fine parallel lines (chain lines and wire lines) caused by the wire mesh of the paper mould. The texture is synthesised as:

```python
def laid\_texture(h: int, w: int, strength: float,
                  rng: np.random.Generator) -> np.ndarray:
    texture = np.zeros((h, w), dtype=np.float32)
    # Chain lines: widely spaced vertical stripes
    chain\_spacing = rng.integers(18, 28)
    for x in range(0, w, chain\_spacing):
        texture\[:, x] -= strength \* rng.uniform(0.4, 0.8)
    # Wire lines: fine horizontal stripes
    wire\_spacing = rng.integers(3, 6)
    for y in range(0, h, wire\_spacing):
        texture\[y, :] -= strength \* rng.uniform(0.15, 0.35)
    return texture
```

The texture is added to the background before any ink is applied. Chapter 10 adds a separate, independent yellowing and foxing pass on top of the theme-level background; the two are not equivalent and must not be collapsed.

###### 9.5.5 Ink Palette

```python
ink\_colour\_range = (0, 55)         # black to dark grey, with visible fading
ink\_fade\_prob    = 0.18            # 18% of segments have visibly faded ink
ink\_bleed\_radius = 0.6             # mild bleed applied at theme level
```

The theme-level `ink\_bleed\_radius` applies a mild Gaussian blur (`σ = ink\_bleed\_radius`) to each stroke *before* compositing onto the canvas. This pre-compositing blur models ink soaking into the paper fibres. Chapter 10's post-compositing blur models the optical spread in the scan process. The two are physically distinct and are applied at different pipeline stages.

###### 9.5.6 Typography: Cyrillic Technical Lettering

```python
font\_family            = "fonts/soviet/BTITechnical.ttf"
font\_size\_range        = (8, 11)
char\_jitter\_px         = 1.2
char\_angle\_jitter\_deg  = 1.5
dimension\_terminal     = "TICK\_OBLIQUE"
annotate\_openings      = False
annotation\_density\_range = (0.6, 1.0)
```

Soviet BTI drawings use a highly distinctive hand-lettered technical typeface. Room labels and dimension numbers are in Cyrillic. The generator includes a synthetic Cyrillic technical font that matches the stroke weight and letterform proportions of authentic BTI lettering. The `char\_jitter\_px` and `char\_angle\_jitter\_deg` parameters model the per-character baseline and rotation variation characteristic of hand lettering.

When `monochrome\_prob` forces grayscale (always true for Soviet era, since `colour\_mode = "none"`), all lettering is rendered in the era's ink palette. Colour is never applied to Soviet-era outputs.

##### 9.6 The Theme Plugin Interface

Themes register with the generator via a lightweight plugin interface. The interface allows new era variants to be added — for example, a `"gdr"` (East German) sub-variant of the Soviet era, or a `"blueprint"` theme for Diazo-printed plans — without modifying the core rendering pipeline.

```python
class EraThemePlugin(Protocol):
    era\_id: str

    def build\_theme(self, rng: np.random.Generator,
                    image\_size: tuple\[int, int]) -> EraTheme:
        """
        Construct and return a fully populated EraTheme instance.
        May use rng to sample any parameters that vary per image
        within the era's allowed range.
        image\_size is provided so that pixel-space parameters
        (line weights, spacings) can be scaled correctly.
        """
        ...
```

Registration and lookup:

```python
\_THEME\_REGISTRY: dict\[str, EraThemePlugin] = {}

def register\_theme(plugin: EraThemePlugin) -> None:
    \_THEME\_REGISTRY\[plugin.era\_id] = plugin

def get\_theme(era\_id: str, rng: np.random.Generator,
              image\_size: tuple\[int, int]) -> EraTheme:
    if era\_id not in \_THEME\_REGISTRY:
        raise KeyError(f"Unknown era: {era\_id!r}. "
                       f"Registered eras: {list(\_THEME\_REGISTRY)}")
    return \_THEME\_REGISTRY\[era\_id].build\_theme(rng, image\_size)
```

The three built-in themes (`scan`, `digital`, `soviet`) are registered at module import time. The `era\_mix` parameter from §1.3.1 governs sampling weights but does not interact with the registry; any registered era can be sampled regardless of whether it appears in `era\_mix`.

##### 9.7 Cross-Era Consistency Rules

Regardless of era, the following rendering rules are invariant. These are reproduced here as a checklist for implementers adding new era plugins:

1. **Mask independence.** No theme parameter affects mask output. Themes control image appearance only; the mask is derived solely from polygon geometry and the class write-order table in Chapter 11.
2. **Monochrome enforceability.** Every theme must produce a valid grayscale output when `monochrome\_prob` forces it. Themes with `colour\_mode = "tint"` apply a grayscale-equivalent of their tint (i.e., the per-channel average) when monochrome is forced; this ensures a smooth mono distribution without hard-clipping colour logic.
3. **Line weight scaling.** All pixel-space parameters in `EraTheme` — `line\_weight\_range`, `overshoot\_px`, `hatch\_spacing\_px`, etc. — are specified at the reference resolution of 512×512 and must be multiplied by `min(image\_size) / 512` in `build\_theme()` before populating the dataclass. Failure to scale produces visually incorrect outputs at non-standard resolutions.
4. **Stroke model availability.** The `"hand"` stroke model is computationally more expensive than `"solid"`. At 1024×1024 it adds approximately 35 ms per image (single CPU core). New themes should use `"solid"` unless the era genuinely requires variable-width strokes. The performance budget in Chapter 12 reserves a 60 ms allowance for the stroke rendering pass at 1024×1024.
5. **Font bundling.** All font files referenced by `font\_family` must be present in `icon\_pack\_dir/fonts/<era\_id>/` before the theme is registered. The registration call does not validate font availability; missing fonts are detected only at render time and will raise `FileNotFoundError`. Chapter 12's integration test suite includes a font availability check for all registered themes.

##### 9.8 Relationship to Downstream Chapters

|Chapter|Dependency on era theme|
|-|-|
|**5**|Wall stroke model, fill style, and line weight are read directly from `EraTheme`|
|**6**|Opening symbols (door swing radius, window sill width) scale with `line\_weight\_range`|
|**7**|Icon compositing uses `ink\_colour\_range` to tint icons to match the plan's ink|
|**8**|Dimension terminal style, font, annotation density, and label knockout rules are all era-governed|
|**10**|The `augmentation\_preset` string is derived from `era\_id`: `"heavy"` for `soviet`, `"medium"` for `scan`, `"clean"` for `digital`; Chapter 10 reads `era\_id` directly but honours the theme system's palette constraints|
|**11**|Mask writing is era-independent; Chapter 11 receives only polygon geometry|

The era theme system is the single most important consistency mechanism in the generator. It ensures that every visual element — from the width of a wall line to the font of a room number to the texture of the paper — is drawn from a coherent, physically motivated distribution. A model trained on outputs from this generator should not encounter any combination of visual features that the generator would not itself produce.

#### Chapter 10 — Augmentation and Degradation

##### 10.1 The Image-Only Constraint

The constraint introduced in §1.7 and repeated here for emphasis: **augmentation and degradation operations touch the image only. They must never modify the mask.**

This is not merely a convenience rule. It is a statement about what the mask represents. The mask encodes the ideal geometric annotation — the Platonic version of the drawing, free of reproduction artefacts. A model trained on this data must learn to read through degradation, not to incorporate it into its understanding of class boundaries. If the mask were degraded in step with the image, the model would be supervised on blurred, skewed, JPEG-corrupted class boundaries and would learn those as ground truth. It would then fail on clean images, on heavily degraded images whose degradation pattern differs from training, and on any image whose degradation is inconsistent with its spatial class structure.

Every operation in this chapter is therefore applied to `GeneratorOutput.image` only. `GeneratorOutput.mask` and `GeneratorOutput.class\_mask` are written once by Chapter 11 and are not touched thereafter.

##### 10.2 The Three Augmentation Presets

The `augmentation\_preset` parameter (§1.3.1) selects one of three named profiles. The mapping from era to preset is:

|Era|Default preset|
|-|-|
|`digital`|`"clean"`|
|`scan`|`"medium"`|
|`soviet`|`"heavy"`|

This mapping is a default, not a constraint. A caller may pass `augmentation\_preset="heavy"` for a `digital`-era image to simulate an aggressively degraded export; the augmentation pipeline will apply the `heavy` profile regardless. However, the resulting image will be visually inconsistent (pixel-perfect wall lines beneath heavy scan noise) and should be used only for stress-testing model robustness, not for standard training data.

Each preset is an instance of `AugmentationPreset`:

```python
@dataclass
class AugmentationPreset:
    name: str

    # Geometric
    skew\_angle\_deg\_max: float        # maximum rotation applied to canvas
    perspective\_warp\_strength: float # 0.0 = none; 1.0 = strong keystone
    barrel\_distortion\_k: float       # radial distortion coefficient; 0.0 = none

    # Blur
    gaussian\_blur\_sigma\_range: tuple\[float, float]  # (min, max); 0.0 = skip
    motion\_blur\_length\_range:  tuple\[int,   int]    # px; 0 = skip
    motion\_blur\_prob:          float

    # Noise
    gaussian\_noise\_sigma\_range: tuple\[float, float]
    salt\_pepper\_prob:           float    # fraction of pixels affected; 0.0 = skip

    # Compression
    jpeg\_quality\_range: tuple\[int, int]  # (min, max); None = skip entirely
    jpeg\_prob:          float

    # Paper and scanner effects
    yellowing\_strength\_range: tuple\[float, float]  # (min, max); 0.0 = skip
    foxing\_prob:              float                # probability of foxing spots
    fold\_line\_prob:           float                # probability of fold crease
    roller\_mark\_prob:         float                # probability of scan roller band
    vignette\_strength\_range:  tuple\[float, float]  # 0.0 = none

    # Uneven illumination
    illumination\_gradient\_prob:     float
    illumination\_gradient\_strength: float
```

The three presets are populated as follows.

###### 10.2.1 Preset: `"clean"`

Applied to `digital`-era images. The intent is near-zero degradation — the output should look like a direct CAD export viewed on screen.

```python
AugmentationPreset(
    name                        = "clean",
    skew\_angle\_deg\_max          = 0.0,
    perspective\_warp\_strength   = 0.0,
    barrel\_distortion\_k         = 0.0,
    gaussian\_blur\_sigma\_range   = (0.0, 0.0),
    motion\_blur\_length\_range    = (0, 0),
    motion\_blur\_prob            = 0.0,
    gaussian\_noise\_sigma\_range  = (0.0, 2.0),   # barely perceptible
    salt\_pepper\_prob            = 0.0,
    jpeg\_quality\_range          = (92, 99),
    jpeg\_prob                   = 0.3,           # occasional mild compression
    yellowing\_strength\_range    = (0.0, 0.0),
    foxing\_prob                 = 0.0,
    fold\_line\_prob              = 0.0,
    roller\_mark\_prob            = 0.0,
    vignette\_strength\_range     = (0.0, 0.0),
    illumination\_gradient\_prob  = 0.0,
    illumination\_gradient\_strength = 0.0,
)
```

###### 10.2.2 Preset: `"medium"`

Applied to `scan`-era images. Models the physical round-trip: printed on a laser or inkjet printer, placed on a flatbed scanner, and exported as JPEG or PNG.

```python
AugmentationPreset(
    name                        = "medium",
    skew\_angle\_deg\_max          = 3.5,           # scanner bed misalignment
    perspective\_warp\_strength   = 0.008,
    barrel\_distortion\_k         = 0.012,
    gaussian\_blur\_sigma\_range   = (0.3, 1.2),
    motion\_blur\_length\_range    = (0, 3),
    motion\_blur\_prob            = 0.08,
    gaussian\_noise\_sigma\_range  = (1.0, 8.0),
    salt\_pepper\_prob            = 0.0005,
    jpeg\_quality\_range          = (55, 85),
    jpeg\_prob                   = 0.75,
    yellowing\_strength\_range    = (0.02, 0.14),
    foxing\_prob                 = 0.12,
    fold\_line\_prob              = 0.10,
    roller\_mark\_prob            = 0.18,
    vignette\_strength\_range     = (0.0, 0.12),
    illumination\_gradient\_prob  = 0.20,
    illumination\_gradient\_strength = 0.08,
)
```

###### 10.2.3 Preset: `"heavy"`

Applied to `soviet`-era images. Models hand-drafted originals that have been physically aged, folded, stored in a damp archive, and then scanned on low-quality equipment with uneven illumination.

```python
AugmentationPreset(
    name                        = "heavy",
    skew\_angle\_deg\_max          = 5.0,
    perspective\_warp\_strength   = 0.018,
    barrel\_distortion\_k         = 0.025,
    gaussian\_blur\_sigma\_range   = (0.6, 2.0),
    motion\_blur\_length\_range    = (2, 6),
    motion\_blur\_prob            = 0.22,
    gaussian\_noise\_sigma\_range  = (4.0, 18.0),
    salt\_pepper\_prob            = 0.002,
    jpeg\_quality\_range          = (40, 72),
    jpeg\_prob                   = 0.85,
    yellowing\_strength\_range    = (0.10, 0.38),
    foxing\_prob                 = 0.55,
    fold\_line\_prob              = 0.45,
    roller\_mark\_prob            = 0.30,
    vignette\_strength\_range     = (0.05, 0.28),
    illumination\_gradient\_prob  = 0.60,
    illumination\_gradient\_strength = 0.20,
)
```

##### 10.3 Operation Catalogue

Operations are applied in a fixed sequence. The sequence is not arbitrary: each operation must be applied at the stage where it is physically meaningful, and later operations must not undo the effects of earlier ones in ways that corrupt the model of the degradation process. The ordering is:

```
Stage A — Geometric distortions
Stage B — Paper and ink pre-effects
Stage C — Blur
Stage D — Noise
Stage E — Scanner surface effects
Stage F — Illumination
Stage G — Compression
```

###### Stage A — Geometric Distortions

**Skew (rotation).** A rotation of `θ \~ Uniform(-max, +max)` degrees is applied to the image canvas, with `max = skew\_angle\_deg\_max`. The fill colour for pixels introduced at the canvas boundary by rotation is sampled from the background palette of the active era theme (`EraTheme.bg\_colour\_range`), not hardcoded to white. The mask is **not** rotated. The image–mask pair is now misaligned in pixel space, which is intentional: the model must learn to be robust to a few degrees of scan skew.

The skew is stored in `metadata\["skew\_deg"]` so that callers who require aligned image–mask pairs can undo it.

**Perspective warp.** A random homography is applied to model the slight keystone distortion introduced when a document is placed slightly off-axis on a flatbed scanner, or photographed rather than scanned. The four corner displacements are sampled independently from `Uniform(0, perspective\_warp\_strength × image\_size)`. Skipped if `perspective\_warp\_strength == 0.0`.

**Barrel distortion.** Applied as a radial distortion `r' = r(1 + k·r²)` where `r` is the normalised distance from the image centre and `k = barrel\_distortion\_k`. This models the mild lens barrel distortion of flatbed scanner optics and phone-camera captures. Skipped if `k == 0.0`.

Geometric distortions are applied first because all subsequent effects (blur, noise, JPEG artefacts) must be applied to the already-distorted geometry, not to the pre-distortion geometry — matching the physical process in which the degraded paper is what the scanner optic processes.

###### Stage B — Paper and Ink Pre-Effects

**Yellowing.** The grayscale image is converted to a three-channel float array and a warm-toned bias is applied channel-wise:

```python
def apply\_yellowing(img\_f32: np.ndarray, strength: float,
                     rng: np.random.Generator) -> np.ndarray:
    """img\_f32: (H, W, 3) float32 in \[0, 1]."""
    r\_shift = strength \* rng.uniform(0.55, 1.00)
    g\_shift = strength \* rng.uniform(0.20, 0.50)
    b\_shift = strength \* rng.uniform(0.10, 0.30)
    img\_f32\[..., 0] = np.clip(img\_f32\[..., 0] + r\_shift \* (1 - img\_f32\[..., 0]), 0, 1)
    img\_f32\[..., 1] = np.clip(img\_f32\[..., 1] + g\_shift \* (1 - img\_f32\[..., 1]), 0, 1)
    img\_f32\[..., 2] = np.clip(img\_f32\[..., 2] - b\_shift \* img\_f32\[..., 2],      0, 1)
    return img\_f32
```

The multiplicative form `shift × (1 - channel)` ensures that already-dark pixels (ink) are yellowed less than light pixels (paper), matching the physical effect where ink resists yellowing more than cellulose fibre. The transformation is applied to the paper background selectively: it affects only pixels brighter than the era's `ink\_colour\_range` upper bound.

This yellowing pass is physically and computationally distinct from the theme-level `bg\_colour\_range` bias in Chapter 9. The theme-level parameter sets the base paper tone before any drawing is composited. This pass adds spatially uniform yellowing *after* compositing, modelling the ageing of the complete assembled drawing. They must not be collapsed into a single operation.

**Foxing.** Foxing spots are small, roughly circular patches of brown–orange discolouration caused by fungal or oxidative degradation of paper. When active (sampled with `foxing\_prob`), a Poisson-distributed number of spots (mean 12, capped at 40) are placed at random positions within the canvas. Each spot is an ellipse with:

* Semi-axes `a, b \~ Uniform(3, 14)px`, random orientation
* Colour: a warm brown sampled from `HSV(25°–40°, 0.4–0.7, 0.55–0.80)` converted to the output colour space
* Edge softening: a Gaussian feathering of radius `min(a, b) \* 0.4`

Foxing spots are placed at Stage B because they are a property of the paper, not of the scanner, and should be subject to the Stage C blur that follows.

**Fold lines.** A fold crease is a thin, low-contrast line running across the image, typically horizontal or vertical, modelling a document that was folded for storage. When active (sampled with `fold\_line\_prob`), one to three fold lines are drawn as:

* Orientation: horizontal (60%) or vertical (40%)
* Position: `Uniform(0.15, 0.85)` × image dimension
* Width: 1–3px
* Tone: slightly darker than the local background — a fold creates a shadow crease, not an ink mark
* Opacity: `Uniform(0.3, 0.7)` — subtle

Fold lines cross all drawn elements and are added directly to the image array. They are class 0 in the mask, but no explicit mask operation is required: because they are applied post-composition, the mask already exists and is not modified.

###### Stage C — Blur

Blur is applied as a single composed pass. If both Gaussian blur and motion blur are active, the Gaussian blur is applied first and the motion blur second.

**Gaussian blur.** `scipy.ndimage.gaussian\_filter` with `σ` sampled from the preset range. For monochrome images this operates on the single channel; for colour images on each channel independently. Values of `σ < 0.3` are treated as zero (no blur applied).

**Motion blur.** A directional kernel of `length` pixels is constructed at a random angle `\~ Uniform(0°, 180°)` and convolved with the image. This models a scanner whose document feed moves slightly unsteadily. Applied with probability `motion\_blur\_prob`.

###### Stage D — Noise

**Gaussian noise.** Additive Gaussian noise with `σ` sampled from the preset range, clipped to `\[0, 255]` after addition. Applied as `uint8` addition with overflow protection.

**Salt-and-pepper noise.** A fraction `salt\_pepper\_prob` of pixels are set to 0 (pepper) or 255 (salt) with equal probability. This models dust on the scanner glass or individual sensor failures in low-quality scan hardware.

###### Stage E — Scanner Surface Effects

**Roller marks.** A flatbed scanner's document-feed mechanism can leave a faint repeating horizontal band across the image at intervals corresponding to the roller spacing. When active, 1–4 bands are placed at `Uniform(0.0, 1.0)` × image height, each with:

* Height: 2–6px
* Tone shift: `Uniform(-12, +8)` on the grayscale value (rollers can leave both dark smears and light cleaning streaks)
* Softened edges: Gaussian feathering of 1px

**Vignette.** A radially symmetric darkening is applied to the canvas corners using a smooth mask:

```python
def vignette\_mask(h: int, w: int, strength: float) -> np.ndarray:
    cy, cx = h / 2, w / 2
    y, x   = np.mgrid\[0:h, 0:w]
    r      = np.sqrt(((y - cy) / cy) \*\* 2 + ((x - cx) / cx) \*\* 2)
    return 1.0 - strength \* np.clip(r - 0.5, 0.0, 1.0)
```

The image is multiplied element-wise by this mask. Vignette models the light fall-off at the edges of a flatbed scanner bed and is particularly common in older scan hardware.

###### Stage F — Uneven Illumination

A smooth low-frequency illumination gradient is generated as a bicubic-upsampled random 4×4 grid of values in `\[1 - s, 1 + s]` where `s = illumination\_gradient\_strength`. The image is multiplied by this field, modelling a light source that is brighter at one side of the scanner bed. Applied with probability `illumination\_gradient\_prob`.

###### Stage G — Compression

**JPEG compression.** Applied by encoding the image to a JPEG byte buffer at quality `q \~ Uniform(jpeg\_quality\_range)` and decoding back to a NumPy array. This introduces the characteristic blocking and ringing artefacts at quality settings below 75. Applied with probability `jpeg\_prob`. Skipped entirely if `jpeg\_quality\_range is None`.

JPEG is applied last because it should encode all previously applied degradations, not be followed by additional operations that would add structure JPEG cannot represent.

##### 10.4 Colour Image Handling

The image entering the augmentation pipeline may be either single-channel (grayscale) or three-channel (colour), depending on the `monochrome\_prob` roll and the era's `colour\_mode`. All operations above support both modes. The following rules apply to colour images specifically:

* Geometric distortions, blur, noise, and compression operate identically on all channels.
* Yellowing operates on channel-specific biases as specified in §10.3 Stage B.
* Foxing spot colour is rendered in RGB for three-channel images; for grayscale images the spot is rendered as a darkening of the background value (no colour, just tone).
* Monochrome enforcement: if `monochrome\_prob` forced the image to grayscale at generation time, it arrives here as a single-channel array and **must leave as a single-channel array**. No augmentation operation may introduce a colour dimension. This is checked by assertion at the end of the pipeline.

##### 10.5 Augmentation Metadata

The following keys are added to `metadata` by the augmentation pass:

```python
metadata\["augmentation\_preset"]  = preset.name
metadata\["skew\_deg"]             = applied\_skew
metadata\["jpeg\_quality"]         = applied\_quality   # None if not applied
metadata\["yellowing\_strength"]   = applied\_yellowing  # 0.0 if not applied
metadata\["foxing\_applied"]       = bool
metadata\["fold\_lines\_applied"]   = int   # count of fold lines placed
```

These fields enable training-time curriculum strategies (e.g., filtering out high-degradation samples for early epochs) and post-hoc analysis of model failure modes by degradation type.

\---

#### Chapter 11 — Mask Writing

##### 11.1 The Image-Only Constraint (Restated)

As introduced in §1.7 and elaborated in §10.1, augmentation and degradation operations touch the image only. Chapter 10 never modifies the mask. This chapter describes the pipeline that *creates* the mask; after this pipeline runs, the mask is frozen. No subsequent chapter, operation, or augmentation pass may write to it.

The mask is written in pixel space at the same resolution as the output image (`image\_size`). It is a single-channel `uint8` NumPy array. Its values are a strict subset of `{0, 64, 128, 192, 255}` corresponding to the five classes defined in §1.2. Any pixel that does not receive an explicit write from the protocol below retains the background value of `0`.

##### 11.2 The Write-Order Priority Stack

The mask is written by a sequence of polygon-fill operations in a fixed priority order. Higher-priority operations overwrite lower-priority ones. The stack, reproduced and expanded from §8.6, is:

|Priority|Layer|Class written|Pixel value|
|-|-|-|-|
|1 (lowest)|Background fill|0|`0`|
|2|Footprint interior|4|`255`|
|3|Room fills|4|`255`|
|4|Wall polygons|1|`64`|
|5|Opening carve-outs|2 or 3|`128` or `192`|
|6|Icons and furniture|4|`255`|
|7 (highest)|Dimension annotations|0|`0`|

Each priority level is implemented as a single rasterisation pass over the mask array. A pass at priority N writes its pixels unconditionally, overwriting whatever value is already present from all passes 1 through N−1. The passes are not cumulative masks; they are direct pixel assignments.

**Why this ordering?** The stack encodes the semantic containment hierarchy of a floor plan. Walls bound rooms (4 is written before 1, so wall pixels that coincide with any room polygon are correctly class 1). Openings interrupt walls (2/3 is written after 1, carving through wall pixels). Furniture is placed inside rooms and may overlap wall edges (4 is written after 1, ensuring furniture does not inherit the wall class). Dimension annotations are drawn on top of everything and belong to background (0 is written last at priority 7, overwriting any structural class beneath a dimension line). The stack order is the ground truth for every pixel that is covered by elements from more than one layer.

##### 11.3 Polygon-Fill Protocol

All mask writes are performed by rasterising Shapely polygons to pixel arrays. The rasterisation function is:

```python
from rasterio.features import rasterize
from rasterio.transform import from\_bounds

def fill\_polygon(mask: np.ndarray, polygon: shapely.Polygon,
                 value: int) -> None:
    """
    Fills the interior of `polygon` (including boundary pixels) with
    `value` in `mask`. Operates in-place.
    mask    : (H, W) uint8 array
    polygon : Shapely Polygon in pixel coordinates (origin top-left, y-down)
    value   : integer in {0, 64, 128, 192, 255}
    """
    h, w = mask.shape
    transform = from\_bounds(0, 0, w, h, w, h)
    burned = rasterize(
        \[(polygon, value)],
        out\_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all\_touched=False,   # interior pixels only; boundary by scanline rule
    )
    mask\[burned != 0] = value
```

`all\_touched=False` uses the scanline fill convention: a pixel is interior if its centre point is strictly inside the polygon. This is consistent with how PIL and Pillow rasterise polygons and avoids the one-pixel boundary fringe that `all\_touched=True` would introduce.

The `rasterize` call is used in preference to `PIL.ImageDraw.polygon` because it is deterministic across platforms, handles sub-pixel vertex precision correctly, and supports polygons with holes (used in the opening carve-out protocol, §11.5).

##### 11.4 The Hollow-Wall Mask Rule

As specified in §1.2.2 and referenced again in §9.3.2: when a wall is drawn in `hollow` fill style — two parallel contour lines with an empty gap between them — the mask must label the *entire wall band* (both contour strokes and the interior gap) as class 1.

This rule is implemented at the polygon level, not the pixel level. The wall graph (Chapter 5) maintains each wall as a **filled band polygon**: a rectangle (or parallelogram, for diagonal walls) whose extents span the full wall thickness including the hollow interior. This polygon is what is passed to `fill\_polygon` at priority 4.

The image rendering and the mask rendering of a hollow wall therefore diverge intentionally:

* **Image pass**: draw two parallel lines (or a filled outline), leaving the centre white
* **Mask pass**: fill the entire band polygon with class 1, including the white centre

The image shows white; the mask shows 64. This is not an error. The model must learn that the white band between two wall lines is topologically wall, not room and not background. Training it to do so requires that the mask label the entire band as class 1.

The band polygon is computed from the wall's centreline and the era's `line\_weight\_range` upper bound, not from the rendered pixel width:

```python
def wall\_band\_polygon(centreline: LineString, half\_width: float) -> Polygon:
    """Returns the filled band polygon for a wall segment."""
    return centreline.buffer(half\_width, cap\_style=2, join\_style=2)
    # cap\_style=2 (flat), join\_style=2 (flat/miter) matches the butt-join
    # rendering convention used in scan and digital eras.
```

For the Soviet era's variable-width strokes, `half\_width` is set to the maximum of the stroke's `hand\_width\_profile()` output. This is a conservative overestimate; it ensures that the mask band is never narrower than the widest rendered ink, preventing any wall pixel from being labelled as background.

##### 11.5 The Opening Carve-Out Protocol

As specified in §1.2.3, openings are carved out of wall polygons before mask writing, ensuring zero overlap between class 1 and classes 2/3. The carve-out is a two-step operation that is atomic with respect to the write-order stack:

**Step 1 — Subtract the opening from the wall band.** Before the wall band polygon is written to the mask, the opening polygon (door or window) is subtracted from it:

```python
wall\_band\_minus\_opening = wall\_band\_polygon.difference(opening\_polygon)
fill\_polygon(mask, wall\_band\_minus\_opening, value=64)   # priority 4
```

**Step 2 — Write the opening.** The opening polygon is then written at priority 5:

```python
fill\_polygon(mask, opening\_polygon, value=128)  # window, or
fill\_polygon(mask, opening\_polygon, value=192)  # door
```

The subtraction in Step 1 ensures that no pixel covered by the opening polygon is written as class 1 in Step 1 and then overwritten to class 2/3 in Step 2. Either approach would produce the same final mask state, but the explicit subtraction removes any dependency on operation ordering within priority level 4 and 5 and makes the zero-overlap guarantee testable independently of write order.

**Door swing arcs.** The door opening polygon (class 3) includes the swing arc and the BTI cross-ridge symbol, as specified in §1.2.3. The swing arc is a pie-sector polygon. It may overlap with room furniture or floor area. Because it is written at priority 5 (after priority 6, icons), it does overwrite any furniture class pixels beneath it. This is correct: the swing arc is a structural annotation (it defines the clearance space of the door) and takes semantic precedence over furniture placed inside the swing zone.

Wait — priority 5 is written *before* priority 6. So icons written at priority 6 overwrite door swing arcs at priority 5. This is also correct: an icon placed inside a door swing zone takes visual precedence in the image rendering, and the mask should reflect that the model will see furniture pixels in that region. Both orderings are defensible; the ordering as specified (icons overwrite openings) matches the image rendering convention in which furniture composited on top of the floor plan visually covers the swing arc line.

**Window sill pixels.** The window polygon covers the full sill-and-reveal band as drawn — typically the gap between the inner and outer wall lines at the window location. Its width matches the wall thickness. Pixels of the wall band that coincide with the window polygon are written as class 2 (window) by the carve-out protocol, not class 1.

##### 11.6 Background as Footprint Complement

As documented in §2.9, the background class (0) is the bitwise complement of the rasterised footprint polygon. Concretely, the footprint polygon is rasterised at priority 2 as a filled region of class 4. All pixels that do not fall within the footprint receive class 0 from the priority 1 background fill and are never overwritten by any higher-priority operation (since all structural elements — walls, openings, furniture — lie within the footprint by construction). The only exception is dimension annotations at priority 7, which may extend outside the footprint into the margin; these are also class 0 and are therefore consistent with the background even in the margin.

The footprint rasterisation at priority 2 is redundant in the sense that it is immediately overwritten by rooms at priority 3 and walls at priority 4. Its purpose is defensive: it ensures that any gap in the BSP room decomposition (e.g., a thin corridor missed by the room subdivision) is labelled class 4 (furniture/contents) rather than class 0 (background). A gap inside the footprint that is labelled background would create a hole in the semantic map that the model would incorrectly interpret as exterior space.

##### 11.7 Era Independence

The mask is era-independent. No field of `EraTheme` is consulted during mask writing. The mask writing pass receives only:

* The `Footprint` dataclass (for the footprint polygon and metadata)
* A list of wall band polygons (from Chapter 5)
* A list of opening polygons tagged with class ID (from Chapter 6)
* A list of icon bounding polygons (from Chapter 7)
* A list of dimension annotation element polygons (from Chapter 8)

The era governs how these elements *look*. It does not govern what class they *are*. The wall rendering pass in Chapter 5 will draw a hollow wall differently for `scan` versus `soviet` era; the mask writing pass receives the same band polygon and writes the same class 1 fill regardless.

This separation is the mechanism that allows Chapter 10's augmentation to corrupt the image arbitrarily without invalidating the mask. The mask was derived from geometry, not from rendered pixels, and geometry is era-invariant.

##### 11.8 The `class\_mask` Encoding

After the `mask` array is written, the `class\_mask` is derived from it by a lookup:

```python
CLASS\_PIXEL\_MAP = {0: 0, 64: 1, 128: 2, 192: 3, 255: 4}

def encode\_class\_mask(mask: np.ndarray) -> np.ndarray:
    class\_mask = np.zeros\_like(mask)
    for pixel\_val, class\_id in CLASS\_PIXEL\_MAP.items():
        class\_mask\[mask == pixel\_val] = class\_id
    return class\_mask
```

This is a pure lookup and introduces no information. It exists to spare the training pipeline from implementing the lookup itself. The `class\_mask` is written as a separate output field and is never used internally by the generator after this point.

##### 11.9 Validation Pass

After the mask is written and the `class\_mask` is derived, a validation pass checks the following invariants:

```python
def validate\_mask(mask: np.ndarray, footprint: Footprint) -> list\[str]:
    errors = \[]

    # 1. All pixel values are legal
    unique = set(np.unique(mask))
    if not unique.issubset({0, 64, 128, 192, 255}):
        errors.append(f"Illegal pixel values: {unique - {0,64,128,192,255}}")

    # 2. No class 1/2/3 pixels outside the footprint
    interior = rasterize\_footprint(footprint)
    for val, name in \[(64, "wall"), (128, "window"), (192, "door")]:
        outside = np.sum((mask == val) \& \~interior)
        if outside > 0:
            errors.append(f"{outside} {name} pixels outside footprint")

    # 3. Zero overlap between class 1 and class 2/3
    wall\_pixels = mask == 64
    for val, name in \[(128, "window"), (192, "door")]:
        overlap = np.sum(wall\_pixels \& (mask == val))
        if overlap > 0:
            errors.append(f"{overlap} pixels simultaneously wall and {name}")

    # 4. Minimum class coverage (sanity check)
    total = mask.size
    if np.sum(mask == 64) / total < 0.005:
        errors.append("Wall class covers < 0.5% of image — suspicious")

    return errors
```

If `validate\_mask()` returns a non-empty error list, the sample is discarded and regenerated with a new seed. The discarded seed is logged in `metadata\["discarded\_seeds"]`. This prevents corrupt samples from entering the training corpus silently.

Validation is fast (< 1 ms at 512×512) and is always run. It is not a debug-only check.

##### 11.10 Implementation Notes

**Rasterio vs. Pillow.** The `rasterize` function from `rasterio.features` is used for all polygon fills. Pillow's `ImageDraw.polygon` is not used in the mask pipeline because its scanline fill behaviour at sub-integer vertices is not precisely specified and has been observed to vary between Pillow versions. The mask pipeline must be pixel-perfect and reproducible across platforms; `rasterio.features.rasterize` provides this guarantee.

**Integer vertex requirement.** All polygons passed to `fill\_polygon` must have integer-pixel vertices. This is enforced by the footprint normalisation pass (Chapter 2, Stage 5). Downstream chapters that modify or derive new polygons from the footprint must apply `polygon.simplify(0.5, preserve\_topology=True)` followed by coordinate rounding before passing to `fill\_polygon`.

**Coordinate axis orientation.** Shapely uses a mathematical (y-up) coordinate system. NumPy arrays use a row-major (y-down) system. All polygons in this generator are maintained in pixel-space (y-down) coordinates, passed directly to `rasterize` with a `from\_bounds` transform that maps pixel coordinates to the array indices without inversion. This convention is established in §2.6 and must be maintained throughout.

\---

#### Chapter 12 — Integration Tests and Performance Budget

##### 12.1 Purpose

This chapter specifies the minimum acceptance criteria for the complete generator pipeline. It defines the test matrix, documents the per-chapter performance budget that all implementations must meet, enumerates the known risks that accumulate through the build order, and specifies the continuous integration checks required for any code change.

The generator is a pipeline of eleven chapters of specification. Each chapter has local invariants. But the compound failure modes — the bugs that only manifest when Chapter 3's room decomposition interacts with Chapter 5's wall graph on a STAIR primitive at aggression 1.0, for a Soviet-era image with heavy augmentation — are not catchable by unit tests. Only integration tests that run the full pipeline end-to-end, systematically varying the key parameters, are sufficient. This chapter defines those tests.

##### 12.2 The Build Order and Accumulated Risks

From §1.7, the canonical build order is:

```
Chapter 2 → Chapter 3 → Chapter 5 → Chapter 4 → Chapter 6 → Chapter 7
         → Chapter 8 → Chapter 9/10 → Chapter 11
```

The risk profile accumulates along this chain. The table below documents, for each stage transition, the primary failure mode introduced at that transition, its likelihood, and its consequence if undetected.

|Transition|Primary failure mode|Likelihood|Consequence if undetected|
|-|-|-|-|
|**Ch. 2**|Validator rejects all cuts → degenerate near-rectangle|Low at aggression < 0.9; moderate at aggression = 1.0 with Z/STAIR|All downstream chapters see a simpler shape than intended; distribution bias|
|**Ch. 2 → 3**|STAIR step-vertices confuse BSP PCA axis selection → uneven room sizes|Low; Chapter 3 pre-simplifies for PCA|Rooms too small for icon placement; silently reduced icon count|
|**Ch. 3 → 5**|Room partition edges not aligned with wall graph exterior ring → orphaned wall segments|Medium; arises on U and T shapes at high aggression|Mask class 1 pixels with no corresponding image stroke|
|**Ch. 5 → 4**|Diagonal wall placement chosen on edge too short for valid attachment → wall polygon outside footprint|Low; Chapter 4 checks edge length|Wall pixels outside footprint detected by Chapter 11 validation; sample discarded|
|**Ch. 5/4 → 6**|Opening placed on diagonal wall despite `no\_annotate` flag → crash or silent skip|Very low; flag is set before placement pass|Crash (caught by try/except, sample discarded) or missing opening in image|
|**Ch. 6 → 7**|Icon bounding box overlaps wall polygon → furniture overwriting wall in mask|Medium; icon placement clips to room interior but not to wall band|Mask class 4 pixels where class 1 expected; training signal error|
|**Ch. 7 → 8**|Dimension label placed on very short segment (< MIN\_ANNOTATION\_LENGTH\_PX) after font scaling → label overflows segment → arrowheads outside canvas|Low; pre-filtered by length check|Arrowhead polygon partially outside canvas → rasterise crops → masking error|
|**Ch. 8 → 9/10**|Era theme not registered before augmentation preset lookup|Very low in production; possible in test environments|`KeyError` at `get\_theme()`|
|**Ch. 10 → 11**|Augmentation (JPEG decode) converts float32 back to uint8 with a range shift → mask values corrupted if mask accidentally processed|Low; mask is never passed to Ch. 10|Mask corruption; detected by validation pass in §11.9|
|**Ch. 11**|Hollow-wall band polygon narrower than rendered stroke → wall pixels outside band → mask class 0 where image class 1|Medium at high aggression on Soviet era (wide strokes)|Model trained on mismatched wall pixels; generalisation degraded|

These risks inform the test matrix in §12.3 and the monitoring strategy in §12.6.

##### 12.3 The Integration Test Matrix

The minimum acceptance test matrix runs the full pipeline for every combination of the following parameters:

|Axis|Values|
|-|-|
|**Primitive**|RECT, L, T, U, Z, STAIR, BEVEL — all 7|
|**Aggression**|0.0, 0.5, 1.0 — three levels|
|**Era**|`scan`, `digital`, `soviet` — all 3|

This yields **7 × 3 × 3 = 63 mandatory integration test cases**. Each case is run with a fixed seed of `42` for reproducibility. The test is considered passing if all of the following hold:

1. `generate\_sample()` completes without raising an exception.
2. `validate\_mask()` returns an empty error list.
3. The mask contains at least one pixel of each of class 0, class 1, and class 4. (Class 2 and 3 may be absent on degenerate footprints with no eligible opening edges, which is acceptable.)
4. `metadata\["era"]` matches the requested era.
5. The `image` shape matches `image\_size` exactly.
6. The `mask` shape matches `image\_size` exactly.
7. All values in `mask` are members of `{0, 64, 128, 192, 255}`.

In addition, the following **statistical distribution tests** are run over a batch of 500 images generated with random seeds, uniform era mix, and aggression sampled from `Uniform(0.0, 1.0)`:

|Metric|Required value|Tolerance|
|-|-|-|
|Fraction of monochrome images|≥ 0.70|± 0.03|
|Era distribution — scan|0.60|± 0.05|
|Era distribution — digital|0.25|± 0.05|
|Era distribution — soviet|0.15|± 0.05|
|Fraction of images with ≥ 1 door pixel|≥ 0.50|± 0.10|
|Fraction of images with ≥ 1 window pixel|≥ 0.55|± 0.10|
|Mean wall pixel fraction|0.08 – 0.22|hard bounds|
|Fraction of images passing `validate\_mask()`|1.00|zero tolerance|

These distribution tests are run in CI on every merge to the main branch (§12.6). Failures indicate a systematic bias introduced by a code change, which may not be caught by the per-sample matrix tests.

##### 12.4 Per-Chapter Acceptance Criteria

###### Chapter 2 — Footprint Generation

* All seven primitives must be constructible at all three aggression levels without raising an exception.
* The validator fallback path (Z → L) must be exercised and logged correctly: test by seeding the RNG to produce a Z primitive at aggression = 1.0 and verifying that `metadata\["primitive\_fallback"]` is populated when fallback occurs.
* `make\_valid()` + `GeometryCollection` handling must be tested explicitly: construct a thin-isthmus polygon that causes `difference()` to return a `GeometryCollection` and verify the largest-polygon extraction logic (§2.8).
* Pixel rounding drift test: generate a footprint, round vertices, call `is\_valid`, verify no failures on 1000 seeds.

###### Chapter 3 — Room Subdivision

* Every room polygon must be strictly interior to the footprint polygon (verified by `footprint.polygon.contains(room.polygon)` for all rooms).
* The union of all room polygons must cover at least 85% of the footprint interior area (gap tolerance for wall band widths).
* No two room polygons may overlap by more than 1 pixel of area (allows for rounding at shared edges).
* PCA axis selection must not crash on STAIR primitives after pre-simplification.

###### Chapter 5 — Wall Graph

* Every wall band polygon must be a subset of `footprint.polygon.buffer(max\_wall\_half\_width)` — no wall may protrude outside the footprint by more than the maximum wall thickness.
* The hollow-wall band polygon must have area ≥ 90% of the equivalent solid-fill polygon. (Less would indicate the band polygon is narrower than it should be.)
* Wall graph T-junction tests: generate at least one T, U, and STAIR primitive and verify that all interior partition junctions produce continuous, non-overlapping wall polygons.

###### Chapter 6 — Openings

* No opening polygon may overlap with any other opening polygon (zero-overlap guarantee).
* Every opening polygon must be strictly contained within a wall band polygon before carve-out.
* No opening may be placed on a diagonal wall: verify by checking that all opening polygons are axis-aligned (bounding box aspect ratio check).

###### Chapter 7 — Icons

* No icon bounding box may extend outside the footprint polygon by more than 2px (clipping tolerance).
* Icon compositing must not modify wall polygons or opening polygons.
* `icon\_pack\_dir` absence must raise `FileNotFoundError` with a descriptive message, not silently produce icon-free images.

###### Chapter 8 — Dimension Annotations

* All arrowhead polygons must be within the canvas boundary (no out-of-bounds rasterisation).
* `metadata\["annotation\_bboxes"]` must be populated correctly and match the number of rendered labels.
* Zero-annotation path (`p\_no\_annotations`): verify `metadata\["annotation\_count"] == 0` on at least one seeded test case.
* Soviet era: verify `TICK\_OBLIQUE` is the only terminal style used (no `ARROW\_FILLED` in soviet samples).

###### Chapter 9 — Era Themes

* Font availability: all `font\_family` paths referenced by registered themes must exist under `icon\_pack\_dir/fonts/`. This check is run at test startup, not at render time.
* `EraTheme` immutability: verify that downstream rendering functions cannot modify the theme instance (use a frozen dataclass or `\_\_setattr\_\_` override in testing).
* New theme registration: register a stub `"test"` theme, verify it is selectable via `get\_theme()`, and verify the 63-case matrix runs without error when era is overridden to `"test"`.

###### Chapter 10 — Augmentation

* Mask invariance: for every augmentation preset, run the full pipeline and verify that `output.mask` is bit-for-bit identical to the mask produced without augmentation (same seed, same parameters, augmentation disabled). This is the definitive test of the image-only constraint.
* Colour-to-grayscale invariance: verify that no `"heavy"` or `"medium"` augmentation operation produces a three-channel output when the input was single-channel.
* JPEG round-trip: verify that JPEG compression at `quality=40` (minimum in `"heavy"` preset) does not corrupt `uint8` dtype or shape.

###### Chapter 11 — Mask Writing

* `validate\_mask()` must pass on every sample in the 63-case integration matrix.
* Hollow-wall test: generate a hollow-wall sample, verify that all pixels in the wall band interior (the white gap in the image) are class 1 in the mask, not class 0.
* Opening carve-out test: verify zero overlap between class 1 and class 2/3 by assertion on every sample.
* Priority stack test: generate a sample in which a dimension line crosses a wall. Verify that the dimension-line pixels are class 0 in the mask, not class 1.

##### 12.5 Performance Budget

All timings are measured on a single CPU core (no GPU, no multiprocessing) at 512×512 unless otherwise specified. The total per-sample budget target is **200 ms** at 512×512, permitting throughput of 5 images per second per core — sufficient to saturate a 4-core machine with 20 images/second for a corpus of 100 000 images in approximately 90 minutes.

|Chapter / Pass|Budget (512×512)|Worst-case seed/config|Budget (1024×1024)|
|-|-|-|-|
|Ch. 2 — Footprint|18 ms|STAIR, aggression=1.0|40 ms|
|Ch. 3 — Subdivision|12 ms|U/T, aggression=1.0|28 ms|
|Ch. 5 — Wall graph|15 ms|Soviet hatch, high vertex count|35 ms|
|Ch. 4 — Diagonal walls|5 ms|BEVEL, 4 diagonals|10 ms|
|Ch. 6 — Openings|8 ms|Dense opening count|18 ms|
|Ch. 7 — Icons|20 ms|Full icon pack, high density|45 ms|
|Ch. 8 — Annotations|10 ms|Running dimensions, all edges|22 ms|
|Ch. 9/10 — Theme + Augmentation|55 ms|Soviet heavy, 1024×1024 stroke engine|120 ms|
|Ch. 11 — Mask writing|15 ms|High polygon count, full validation|32 ms|
|**Total**|**158 ms**|Mixed worst case|**350 ms**|

The 158 ms total at 512×512 is within the 200 ms budget. The 350 ms total at 1024×1024 exceeds the scaled budget (800 ms) comfortably.

The 60 ms allowance for the stroke rendering pass at 1024×1024, noted in §9.7, falls within the Ch. 9/10 budget of 120 ms (the stroke engine accounts for approximately half the augmentation pass time at 1024×1024).

If any single chapter exceeds its budget by more than 50% on a representative sample batch, a profiling run must be initiated before merging the offending change. The performance budget is a regression ceiling, not a target.

##### 12.6 Continuous Integration

The following checks are run on every pull request and every merge to the main branch:

|Check|Tool|Pass condition|
|-|-|-|
|Unit tests|pytest|All pass|
|63-case integration matrix|pytest-parametrize|All 63 cases pass all 7 criteria (§12.3)|
|Statistical distribution tests|pytest + batch run|All distribution metrics within tolerance (§12.3)|
|Font availability|pytest (startup fixture)|All registered theme fonts present|
|Mask invariance (augmentation)|pytest|Bit-identical masks across all presets|
|Performance regression|pytest-benchmark|No chapter exceeds its budget by >50%|
|Type checking|mypy (strict)|Zero errors|
|Linting|ruff|Zero errors|

The 63-case matrix and the statistical batch run are the most expensive CI steps. The matrix runs in approximately 45 seconds on a 4-core CI runner (63 samples × \~0.7 s per sample). The statistical batch (500 images) runs in approximately 100 seconds on the same runner. These are acceptable CI latencies for a dataset generation tool.

**Seed pinning.** The 63-case matrix uses seed 42 for all cases. If a code change causes the output of any case to change (even validly, e.g., because a new wall style is introduced), the pinned reference outputs must be regenerated and committed. The CI check treats any change in reference output as a failure until the references are explicitly updated, preventing accidental silent changes to the generation distribution.

**Coverage.** Line coverage of the generator core (chapters 2–11 implementation code) must remain above 85% as measured by the unit + integration test suite combined. Coverage below 85% blocks merge.

##### 12.7 Known Limitations and Out-of-Scope Items

The following items are out of scope for this generator and are documented here to prevent scope creep in future development:

* **Courtyard buildings.** Footprints with interior rings (holes in the Shapely polygon) are explicitly excluded by the validator (§2.5, Check 2). Adding courtyard support would require changes to Chapter 3's BSP algorithm, Chapter 5's wall graph exterior ring initialisation, and Chapter 11's background fill logic. It is a substantial, independent feature.
* **Multi-storey plans.** The generator produces a single floor plan per image. Generating multi-storey buildings (where one image contains superimposed or stacked plans) is not supported and would require a new composition layer above Chapter 11.
* **Imperial units.** All dimension annotations use metric labels with comma-decimal formatting. Imperial (`ft/in`) formatting requires a separate annotation renderer and a different scale system.
* **Colour walls.** The `digital` era supports per-room colour tints for floor areas, but wall fill colour is always drawn from the ink palette (near-black). Coloured wall fills (common in some contemporary BIM exports) are not modelled.
* **Text recognition quality.** The font rendering pipeline produces visually plausible annotations but is not calibrated to produce OCR-readable text. If the downstream task requires OCR on annotations, the font rendering in Chapter 8 would need to be replaced with a higher-fidelity text renderer.

These limitations are features of the current scope, not bugs. They are listed here so that future implementers can evaluate the cost of removing each constraint with full knowledge of its dependencies.

