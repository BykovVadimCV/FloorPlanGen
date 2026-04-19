# Russian BTI Floor Plan Specification for Synthetic Data Generation

A technical reference for programmatic rendering of BTI (Бюро технической инвентаризации) floor plans across three eras, intended for synthetic dataset generation for U-Net semantic segmentation (5 classes: background, wall, window, door, furniture). All specifications draw from GOST standards, МосгорБТИ/ПИБ/Ростехинвентаризация practice, and archival examples. Dimensions given in mm on paper unless stated; scale 1:100 is the default assumption (1 mm paper = 100 mm reality).

---

## 0. Global conventions and normative base

### 0.1 Governing standards (all eras)
- **ГОСТ 2.301-68** — paper formats (A0 1189×841, A1 841×594, A2 594×420, A3 420×297, A4 297×210 mm; tolerances ±1.5–3.0 mm).
- **ГОСТ 2.302-68** — scales. Architectural plan scales: 1:50, 1:100, 1:200, 1:400, 1:500. BTI apartment default **1:100**; МосгорБТИ **1:200 is the modern default**; 1:500/1:1000 for ситуационный план.
- **ГОСТ 2.303-68** — lines. Base thickness `s` = **0.5–1.4 mm**; thin = s/3 to s/2. Min ink thickness 0.2 mm on formats <841 mm.
- **ГОСТ 2.304-81** — drawing fonts. Type A (d=h/14) and Type B (d=h/10); both upright and slanted 75°. Standard heights: 1.8, **2.5**, **3.5**, **5.0**, 7.0, 10.0, 14.0, 20.0 mm.
- **ГОСТ 2.306-68** — material hatching; primary angle **45°**, spacing 1–10 mm.
- **ГОСТ 2.307-2011 / ГОСТ Р 21.101-2020** — dimensioning. Architectural convention uses **45° tick marks (засечки) 2–4 mm long, slanted to the right**, NOT arrowheads. Dimension line extends **0–3 mm** past outermost extension line. Extension line extends **1–5 mm** past dimension line.
- **ГОСТ 21.101-2020** — title block (основная надпись) form 3 = **185×55 mm**, bottom-right.
- **ГОСТ 21.201-2011 (СПДС)** — conventional signs for walls, partitions, doors (Table 7), windows (Table 8), stairs (Table 6), columns.
- **ГОСТ 21.205-93** — plumbing and heating symbols.
- **ГОСТ 21.501-2011 / -2018** — architectural working drawings. Wall contours in section: **сплошная толстая основная (s)**; visible features beyond section: **сплошная тонкая (s/3)**.

### 0.2 Language and numeric formatting (all eras)
- **Decimal separator: COMMA** (4,82 — never 4.82).
- Interior dimensions on working чертежи: **millimeters with no units suffix** (e.g., `3200`).
- Area values: **square meters with comma, one or two decimals, "м²" suffix** (e.g., `18,5 м²`, `42,85`).
- Elevation marks: **meters with three decimals and sign** (e.g., `+3,000`, `-1,500`).
- Cyrillic italic text slanted **75°** (15° forward lean).

### 0.3 CRITICAL caveat on load-bearing walls
**BTI plans do NOT reliably distinguish несущие (load-bearing) walls from перегородки (partitions) by line thickness alone.** Practitioner sites (pereplanirovkamos.ru, apb1.ru, gsps.ru) explicitly warn: *"Определить по плану БТИ, какие стены в квартире несущие, а какие ненесущие — невозможно."* The synthetic generator must randomize the inner/outer assignment of thickness independently of structural role to reproduce this ambiguity. Real wall thickness in masonry: 120/250/380/510/640 mm (brick multiples); panel 140–200 mm; internal partitions 50–120 mm. These compress on the drawing but do not map one-to-one to line weight.

### 0.4 Terminology disambiguation
- **Поэтажный план** — BTI-era term; BTI-style floor plan sheet.
- **План этажа** — federal cadastral term (post-2017) per Приказ №953 / П/0082.
- **ПИБ** — Проектно-инвентаризационное бюро (NOT "Петербургский…"); SPb equivalent of BTI.
- **Приказ №244** — dated **17.08.2006** (not 2011); governs техпаспорт ИЖС.
- **АО "Ростехинвентаризация — Федеральное БТИ"** — liquidated **31.10.2022**; successor is **ППК "Роскадастр"**. Pre-2022 documents keep the old стамп.

---

## 1. SOVIET-ERA BTI PLANS (pre-1990s)

### 1.1 Document production workflow
1. Field sketch (**абрис**) in pencil on graph-paper booklet (полевой журнал) on-site.
2. Drafting-room (**камеральный**) tracing in black India ink (тушь) onto **калька** (tracing paper) with рейсфедер (ruling pen, pre-1960) or **рапидограф** (tubular pen, post-1960).
3. Reproduction as **синька** (cyanotype or diazo blueprint copy).
4. Filing in inventory case (инвентарное дело) by address/quarter, stored bound with cotton thread through left-edge punch holes with sealing wax on verso.

### 1.2 Substrates and colors
| Substrate | Background RGB | Line RGB | Notes |
|---|---|---|---|
| Fresh калька | (245,232,210) | (5,5,5) black | waxy translucent |
| Aged калька (40–60 yr) | (230,215,185) → (225,205,160) | (55,40,25) sepia-brown | ink fades warm |
| Cyanotype (pre-1940s) | (25,50,110) → (15,40,95) Prussian blue | white (lines blow out +0.1–0.2 mm) | dark edges from exposure |
| Diazo (1940s–80s), dominant | (245,235,200) cream / pinkish | (40,55,140) violet-blue OR (110,75,40) sepia | ammonia-cured; strongly yellows |
| Light-faded region | local gradient to (225,205,160) | drops to (110,95,75) warm grey | exposure vignette |

**Foxing spots:** brown irregular blobs 2–8 mm, core #8B4513, halo #C68E5A, density 3–40 per sheet. **Tea/coffee stains:** irregular tan blotches, darker edge. **Fold creases:** one horizontal + 1–2 vertical; grime concentrated; cracked paper often mended with yellowed scotch tape strips 15–20 mm wide (RGB #F0E0A8 @ 50% opacity). **Archive punch holes:** 2 × Ø6 mm, 80 mm centers, left edge, rust halos after decades.

### 1.3 Line weights (rapidograph nominal series)
ISO rapidograph widths used: **0.1 / 0.18 / 0.25 / 0.35 / 0.5 / 0.7 / 1.0 / 1.4 mm**. Typical A4 apartment plan at 1:100:

| Element | Weight (mm) |
|---|---|
| Outer/capital wall contour | 0.5–0.7 |
| Partition contour | 0.3–0.4 |
| Door/window/plumbing outlines | 0.25–0.35 |
| Dimension and extension lines | 0.18–0.25 |
| Hatching / diagonal fill | 0.1–0.18 |
| Hand-lettering strokes | 0.35 (for h=3.5 mm text, d=h/10) |

**Hand-drawn line quality to reproduce:**
- Waviness amplitude ±0.1–0.3 mm over 100 mm segments.
- Line-weight jitter ±15% along a stroke; thicker at endpoints.
- Ink blots 0.5–1.5 mm at 1% of line-start/end positions.
- Corner overshoots 0.3–1.0 mm with ~50% probability.
- Small gaps or overlaps at junctions (rarely a perfect miter/butt).

### 1.4 Wall rendering (Soviet BTI)
- **Walls drawn as HOLLOW double-line contours** — two parallel thick lines with empty (substrate-colored) interior. Solid-fill poché is rare on Soviet BTI.
- Drawn wall thickness at 1:100: capital walls **4–6 mm**; partitions **1.5–2.5 mm**.
- **No hatching on wall plan view.** Hatching appears only on sections (разрезы) and inside chimneys/vent shafts/balcony floors.
- Wall junctions typically butt-joined (lines meet as T/L without miter).
- **Diagonal/angled walls:** uncommon in Soviet apartments (orthogonal series dominant). When present: 30°, 45°, or 60° relative to main axes; e.g., эркер facets.
- Color: pure black fresh → warm sepia-brown with age. Red ink (#C3201A faded to #B85040) used for **corrections and overwrites**; dashed red for illegal перепланировка (sparingly pre-1991 — мostly a post-Soviet practice).

### 1.5 Door rendering (Soviet BTI)
- **Swing door:** thin straight leaf line (0.3 mm, length = door width) + **quarter-arc (0.18 mm)**, radius = leaf width. Hand-drawn arcs often imperfectly circular — slight flattening, wobble. Arc sweeps **into the room**.
- Leaf line typically at **30° to wall** in classical drafting practice; MosgorBTI output drifts toward **90° (perpendicular, fully closed)** — mix 30°:90° ≈ 40:60 for Soviet.
- Opening gap in wall: **8–10 mm** at 1:100 for a standard 800–900 mm door; wall contour broken at opening with 1–2 mm jamb-return ticks.
- **Double door (двупольная):** two mirrored leaves + two arcs meeting at centerline.
- **Тамбур (vestibule):** small rectangle 900–1500 × 1200–2000 mm, both doors shown.
- **Entry door:** leaf line slightly heavier (0.35–0.5 mm); red apartment number at entrance side.
- **Balcony door (French):** narrow leaf 600–700 mm + arc; leaf rendered as double parallel lines (window-like) indicating glazing.
- **Sliding, pocket, accordion:** rare in Soviet stock; when shown, accordion = zigzag 4–6 segments of 1–2 mm.
- **Arch / opening without door:** wall gap + NO leaf + NO arc.

### 1.6 Window rendering (Soviet BTI)
- Standard: **two parallel thin lines** spanning wall thickness (at 1/3 and 2/3 of thickness). Often **three parallel lines** in Moscow/Leningrad practice (outer wall + glazing + inner wall).
- Four-line (double-glazing) variation on premium pre-war Сталинки.
- Wall contour thick line **broken** across window length.
- **Sill (подоконник):** optional 1–2 mm perpendicular ticks at interior ends.
- **Radiator under window:** rectangle ~3×10 mm (1000×200 mm real) with 3–4 internal vertical ticks representing секции. **Often omitted** on pure BTI inventory plans; more common on architectural plans.
- **Balcony interface:** combined oкoннo-дверной блок; window graphic continues into balcony door zone.
- **Эркер:** 3- or 5-faceted polygonal protrusion; each facet contains own window.

### 1.7 Stamps and seals (штампы)
**Round BTI stamp (круглая печать):** Ø **~40 mm** (USSR гербовая печать norm 38–42 mm). Double-circle border, outer ring 0.5–0.8 mm, inner ring at Ø~35 mm. Outer radial text: "БЮРО ТЕХНИЧЕСКОЙ ИНВЕНТАРИЗАЦИИ" + city (e.g., "ИСПОЛКОМ ЛЕНСОВЕТА · ЛЕНГОРБТИ"). Center: small star, "СССР", or district number. **Ink color:** violet #5F377D, blue #19377E, or red #B41E1E.

**Stamp artifacts to render:**
- Uneven pressure: one-side dark, opposite faded.
- "Donut" effect: darker perimeter, paler center.
- Tilt ±10° from page axis.
- Double-strike ghosting 1–3 mm offset (30% probability).
- Tangential smear 2–5 mm (10%).
- Partial impression: 20–40% of rim missing.
- Fill coverage 30–70% (internal voids/speckle).

**Rectangular title block (штамп-основная надпись):** 55×180 mm or 65×185 mm bottom-right. Pre-printed grid (0.2 mm lines) with hand-filled fields: город/район, адрес, № инвентарного дела, этаж, № квартиры, общая площадь, жилая площадь, масштаб, дата обследования, составил/проверил signatures.

**Inspector stamp (штамп инвентаризатора):** 25×55 mm, purple or blue ink, "Техник-инвентаризатор / ФИО / подпись / дата".

**Archive stamp:** 30×60 mm, often diagonal, "АРХИВ БТИ · КОПИЯ ВЕРНА", date, signature.

### 1.8 Title block content (period wording)
```
ИСПОЛКОМ <ГОРОД>СКОГО СОВЕТА ДЕПУТАТОВ ТРУДЯЩИХСЯ    (pre-1977)
ИСПОЛКОМ <ГОРОД>СКОГО СОВЕТА НАРОДНЫХ ДЕПУТАТОВ       (post-1977)
ОТДЕЛ КОММУНАЛЬНОГО ХОЗЯЙСТВА
БЮРО ТЕХНИЧЕСКОЙ ИНВЕНТАРИЗАЦИИ
ПОЭТАЖНЫЙ ПЛАН                         Масштаб 1:100
Адрес: ул. Ленина, д. 15, кв. 42
Инв. дело № 1234/5
Дата обследования: 12.05.1974
Составил: ____________  Проверил: ____________
```
The **"Совет депутатов трудящихся" → "Совет народных депутатов"** change in 1977 is a period-dating cue.

### 1.9 Lettering (ГОСТ 2.304-81)
- Dominant script: **Тип Б с наклоном 75°** (italic 15° lean).
- Text heights on A4 BTI plans: dimensions/areas **h=2.5 mm**; room numbers **h=3.5 mm**; title-block entries **h=3.5 mm**; main heading **h=5–7 mm**.
- Cyrillic italic peculiarities to reproduce: **т** = three-stroke (Latin "m"); **д** = Latin "g"-like; **и** = Latin "u"; **Ж** symmetric; **Я** mirrored R.
- Digits: European forms with closed-top 4, crossbar 7, footed 1.
- Marginalia (later additions, blue ballpoint): ordinary Cyrillic cursive, inconsistent 2–5 mm, slant 20–30°.

### 1.10 Room labels and annotations
- Room number sequential **1, 2, 3 …** inside **circle Ø 6–8 mm** (0.2 mm outline), or freestanding number.
- Numbering: start at entrance, clockwise.
- Area written as **"18,5"** (no units), **underlined** with 0.2 mm line, lower-right in room per ГОСТ 21.501. Alternative fraction form: **"2 / 18,5"** (room #2, 18.5 m²).
- Ceiling height: **"h=2,50"** in corner (30% of rooms); typical by era: Сталинка 3.00–3.20; Хрущёвка 2.50 (1-447 = 2.45); Брежневка 2.55–2.65; улучшенка 2.70.
- Apartment number: "кв. 42" large, near entry door.

### 1.11 Symbols at 1:100 (Soviet BTI)
| Element | Drawn size (mm) | Notes |
|---|---|---|
| Toilet | 5×12 oval/rounded rect + tank rectangle | with inner flush circle |
| Bathtub | 7×17 rectangle, rounded corners | drain circle one end |
| Sink / раковина | 5×7 rectangle | small drain circle |
| Kitchen sink (мойка) | 5×7, sometimes double basin | with mid-divider |
| Gas stove (плита) | 7×9 rectangle with 4 circles (burners) | "Э" label if electric |
| Masonry стove (печь) | 8–15 × 8–15 large rectangle | 45° hatching fill |
| Gas column (колонка) | Ø 4 mm circle on wall | kitchen wall |
| Vent shaft (вентканал) | 5×5 square | **CROSS-HATCHED (сетчатое)** @ 45° both ways — Soviet marker |
| Chimney (дымоход) | square with large "X" | or dense cross-hatch |
| Garbage chute (мусоропровод) | square + internal circle/diagonal | "МП" label |
| Stairs | parallel 1.5–2 mm tread lines + arrow + "вверх"/"вниз" | diagonal break |
| Balcony | rectangle outside wall, **45° diagonal hatching @ 2 mm spacing** inside | guardrail line |
| Loggia | recessed into footprint, same hatching | 3 enclosing walls |
| Built-in closet | rectangle with X-diagonal | label "Ш" or "вст.шк." |
| Column | solid filled black rectangle/circle | 3–6 mm |

### 1.12 Room-type abbreviations (Soviet/standard)
| Code | Meaning |
|---|---|
| Ж, Ж.к. | жилая комната |
| К, Кух. | кухня |
| С/У, С.У. | санузел совмещённый |
| В, Ван. | ванная |
| Т, Туал. | туалет |
| Кор | коридор |
| Пр | прихожая |
| Тм | тамбур |
| Х | холл |
| Б | балкон |
| Л, Лдж | лоджия |
| Ш, вст.шк. | встроенный шкаф |
| Кл | кладовая |
| Гар | гардеробная |
| ЛК | лестничная клетка |
| МП | мусоропровод |
| ВК | вентканал |
| Дым | дымоход |

### 1.13 Dimensions (Soviet BTI)
- **External chain structure**: 2–3 parallel dimension lines outside each wall:
  1. Openings (окна, двери, простенки) — 14–21 mm from wall.
  2. Between capital-wall axes — +7–10 mm outward.
  3. Overall length — +7–10 mm outward.
- Interior: L×W per room, placed along wall.
- **Ticks (засечки): 2–4 mm at 45° right-slant**, NOT arrowheads. Weight = main line (0.3–0.5 mm).
- Dimension line extends 0–3 mm past extension line.
- Extension line extends 1–5 mm past dimension line.
- Text height 2.5 mm, centered 0.5–1 mm above line.
- **Units: meters with two decimals, comma** ("3,20", "5,75") on BTI plans; occasionally mm on абрис ("3200", "1050").
- Hand-written number-size inconsistency ±20%; digit-crossouts and overwrites common.

### 1.14 Typical Soviet apartment layouts (for layout generator)
| Series | Ceiling h | 1-rm m² | 2-rm m² | 3-rm m² | Kitchen m² | Notes |
|---|---|---|---|---|---|---|
| Сталинка (1930–1955) | 3.00–3.20 | 32–50 | 44–70 | 57–85 | 9–12 | thick brick (510–640), high ceilings, isolated rooms, en-filade doors |
| Хрущёвка 1-447 | 2.45 | 29–33 | 41–46 | 55–58 | 5–6 | brick |
| Хрущёвка 1-464, 1-335, 1-515/5 | 2.50 | 29–33 | 41–46 | 55–58 | 5–6 | panel; смежные комнаты; совмещённый С/У |
| Брежневка П-44, П-3, II-49, II-57 | 2.55–2.65 | 33–38 | 48–55 | 62–72 | 6.5–9 | isolated rooms, separate С/У |
| Улучшенка | 2.70 | 38–42 | 55–65 | 72–85 | 8–10 | — |

Balconies: 1000×3000 mm typical; loggias 1500×3000–6000 mm.

---

## 2. POST-2000 / TRANSITIONAL BTI PLANS (1992–2015)

This era mixes: Soviet originals scanned/photocopied; typewriter labels pasted over hand-drawn bases; early CAD (AutoCAD R14/2000–2004 + **PlanTracer** at МосгорБТИ from ~2002) producing new plans; multi-generation copies.

### 2.1 Documents in circulation
1. **Технический паспорт** (multi-page) — standardized c. 2000 by **Приказ Минземстроя №37 (04.08.1998)** and **Постановление Правительства РФ №921 (04.12.2000)**.
2. **Поэтажный план + экспликация** abbreviated 2-sheet extract.
3. **Архивная копия** — photocopy of original inventory drawing.
4. **Кадастровый паспорт** (2008–2017) — Приказ МЭР №504.

**МосгорБТИ standardized ~2005–2017 A5 booklet**: laminated white cover with **red band/stripe** and coat of arms, holographic sticker, 5–6 stitched numbered pages: cover → сведения о доме → адресный/ситуационный план → поэтажный план → экспликация → (amendments).

### 2.2 Production tiers (all may coexist on one document)

**Tier 1 — hand-drawn originals**: Soviet conventions (§1) preserved; retraced on new калька for reissue; ±15% stroke jitter; hand wobble 0.3–1 px at 200 DPI.

**Tier 2 — typewritten overlays**: monospace Cyrillic (Courier-like, 10–12 pitch), glyph height 2.5–3 mm; baseline drift ±0.5 mm; uneven inking; missing character bottoms ("Д", "Ц"); typically ALLCAPS. Paste-strip edges leave rectangular shadow lines around numeric labels after re-copy.

**Tier 3 — early CAD (PlanTracer / AutoCAD)**: all text UPPERCASE in ISOCPEUR, Simplex, txt.shx, or Arial Narrow; heights 2.5/3.5/5 mm. Perfectly rigid straight lines (zero wobble). **Plotter line weights (ISO pen series): 0.13, 0.18, 0.25, 0.35, 0.50, 0.70, 1.00 mm**. Typical: walls 0.50–0.70, partitions 0.25, dims 0.18. Dimension text horizontal only (never rotated along walls). Fixture symbols from PlanTracer block library — subtly different from hand-drawn counterparts.

**Tier 4 — photocopier degradation** (cumulative over 2–4 generations):
- Edge darkening band 3–15 mm on 1–2 edges.
- Contrast loss: black → #3A3A3A, white → #E8E4D6–#D8D0BC.
- Thin lines break into dotted fragments.
- Dust/scratch transfer: 5–50 random 1–3 px black specks per page.
- Ghost/double-image offset 1–3 px.
- Toner streak lines 1–2 px, full page height, vertical.

**Tier 5 — fax** (rare, intermediates): 1728 px horizontal / ~100 DPI effective; horizontal synchronization streak bands 1–5 px; pure 1-bit thresholding.

**Tier 6 — scanner digitization**: 150 / 200 / 300 DPI (occasionally 600). Mostly 8-bit grayscale; some 24-bit color (for red ink/stamps); some 1-bit. Output: TIFF (gov archives), PDF, JPEG 70–85% quality.

### 2.3 Scanner/copier artifact parameter table

| Artifact | Parameters |
|---|---|
| Skew | Uniform random rotation −3° to +3°, mode ±0.5°–1.5° |
| Edge shadow (lid not closed) | 10–40 px dark gradient band #1A1A1A → #B0B0B0 |
| Edge crop | 2–8% of drawing cut on one edge |
| Fold line | 1–3 px dark line, full page; ±5° from axis; gray #6A6A6A–#3A3A3A; optional ±10 px shadow bands |
| Staple holes | 2 × Ø3–4 mm black dots, 3–5 mm from top-left |
| Staple rust halo | Core #A0522D, halo #D2B48C at 5–10 mm radius |
| Roller streaks (CIS) | 1 px vertical gray #C0C0C0 lines, full height, spacing 50–400 px |
| Moire (halftone copy) | Diagonal fringe, angle 15–45°, period 3–8 px |
| Bed dust | 5–50 spots, 1–5 px, black |
| Paper dust | Larger #707070 specks, 2–10 px |
| JPEG (Q70–85) | 8×8 block edges near text; ±2 px ringing at black lines; color banding in cream bg; red-stamp bleed from 4:2:0 subsampling |
| Bilinear 300→150 DPI | Thin 1 px lines → 0.7 px anti-aliased gray |
| Auto-level clip | Blacks → #0A0A0A, whites → #F8F4E8 |

### 2.4 Paper-condition background palette

| Condition | Hex range |
|---|---|
| Fresh cream (A5 booklet interior) | #FFF8DC → #FAF0D4 |
| Lightly yellowed (5–10 yr) | #F5EFDF → #EFE6CB |
| Moderately yellowed (15–20 yr) | #E8DCBF → #DCCFA8 |
| Heavily yellowed (archive tan) | #D2B48C → #C3A678 |
| Foxing core / halo | #8B4513 / #C68E5A |
| Coffee ring center / edge | #A67B5B / #6B4423 @ 40–70% opacity |
| Tape yellow | #F0E0A8 @ 50% opacity |
| Highlighter yellow / pink / blue | #FFEB99 / #FFB6B6 / #B0D4E8 @ 30% opacity |
| Pencil annotation | #4A4A4A @ 30–80% opacity |
| Whiteout (штрих-корректор) | #FFFFF5 chalky, raised-edge shadow |
| Red ink (перепланировка) | #C8202B → #A01820 |
| Blue ballpoint | #1E3A8A → #2B4A9E |

### 2.5 Colored-line visual language (МосгорБТИ standard)
This is the defining visual code of the transitional era — **critical for training data**:
- **Black lines** — current approved state of the apartment.
- **Red lines** — **unapproved redevelopment (самовольная перепланировка)** drawn by a BTI technician during re-inspection. Forms:
  - Red outlines of new/removed walls.
  - Red **crosshatch fill** over illegally closed openings (заложенный проём).
  - Red numbers replacing black for changed-area rooms.
  - Red dashed lines for demolished partitions.
- **Red entry arrow / room-entry number** — the apartment number at the entrance is **ALWAYS red**, regardless of redevelopment status. **Do not confuse** with перепланировка marks.
- **Dashed black lines** — "намеченные" partitions suggested by developer but not erected (common in новостройки со свободной планировкой).
- **Thin cyan/blue lines** — геометрия from cadastral import (later in period).

### 2.6 Stamps (transitional era)
- Color: **blue/violet** dominant (#1E3A8A → #3B5998); violet-black #2D2560; red #B8252E for inspector certifications.
- Ink coverage 30–70%, fuzzy edges with 1–2 px micro-dots.
- Rotation random ±15°.
- Overlap over lines at 40–70% opacity (multiply blend).
- **Round gerbovaya pechat' Ø 38–42 mm**: post-1992 Russian double-headed eagle in center (replacing hammer-and-sickle); text rings with БТИ name + organization.
- **Oval stamps 55×30 mm**: multi-line inspector info.
- **Rectangular stamps**: "КОПИЯ ВЕРНА", "ВЫДАНО ДЛЯ ПРЕДЪЯВЛЕНИЯ В…", "**РАЗРЕШЕНИЕ НА РЕКОНСТРУКЦИЮ (ПЕРЕПЛАНИРОВКУ) НЕ ПРЕДЪЯВЛЕНО**" (MosgorBTI marker for unauthorized redevelopment).
- **Cadastral engineer stamp** (post-2008): round, "КАДАСТРОВЫЙ ИНЖЕНЕР ФИО, № аттестата".
- At least **2 impressions** per plan sheet (one on plan, one on экспликация) — authenticity marker.

### 2.7 Wall/door/window rendering in transitional era
- Hand-drawn base: same as §1.4–1.6.
- Pure CAD tier: walls often **solid-filled polygonal** (poché becomes common), single uniform 0.25 mm linework ("monoblock") across the whole drawing; 4-line window (ГОСТ 21.201-compliant) rising over 3-line.
- Door leaf angle: **90° dominant** in BTI; 30° on project-origin drawings. Mix both 50/50 for training diversity.
- **Vent-shaft style is an era marker**: cross-hatched "сетка" = hand-drawn/pre-2000; single-direction 45° hatch or solid fill = CAD/post-2005.

### 2.8 Document metadata fields
- **Инвентарный номер**: 6–9 digits + letter suffix.
- **Кадастровый номер** (post-2008): `AA:BB:CCCCCCC:KK` (77 = Москва, 78 = СПб, 50 = МО, 54 = Новосибирск).
- **Адрес**: "г. Москва, ул. …, д. …, корп. …, кв. …".
- **Дата последней инвентаризации**, **Дата выдачи**: DD.MM.YYYY.
- **Материал стен**: кирпич/панель/блок/монолит.
- **Этажность здания, этаж квартиры**.
- **Общая / жилая / подсобная площадь**.
- **Подпись исполнителя**: handwritten blue ballpoint cursive, ~30×8 mm.

### 2.9 Composite layer order (for renderer pipeline)
1. Paper texture (per §2.4) + fiber noise.
2. Base plan (hand-drawn or CAD vectors) rasterized at 200–300 DPI with stroke jitter per tier.
3. Typewritten strips: darker-edge rectangles (paste shadow).
4. **Red-line overlay** (перепланировка), if applicable.
5. Stamps (multiply blend, ±15° rotation, 50–80% opacity).
6. Handwritten marginalia (blue/pencil).
7. Whiteout corrections + overtyped text.
8. Page aging (foxing, coffee, highlighter).
9. Fold creases (darkened line + soft shadow).
10. Scan pipeline: Gaussian σ=0.4–0.8 px blur → contrast clip → grayscale or 1-bit posterize → speckle/dust/roller streaks → rotate ±3° → edge crop → edge shadow → JPEG Q75–85.

---

## 3. MODERN DIGITAL BTI/ЕГРН PLANS (2015+)

### 3.1 Regulatory framework (current as of April 2026)
- **ФЗ-218** (13.07.2015, in force 01.01.2017) — established ЕГРН.
- **Приказ Росреестра П/0082** (15.03.2022, as amended through П/0250/25 of 24.07.2025) — **current технический план form**; superseded earlier Приказ МЭР №953.
- **Приказ Росреестра П/0329** (04.09.2020) — ЕГРН выписка forms; **Раздел 5 = "План расположения помещения на этаже"**.
- **Приказ Росреестра П/0347** (06.09.2023) — current XML schema.
- **Постановление Правительства Москвы 106-ПП** (17.03.2017) — MosgorBTI monopoly in Moscow.
- Technical plan is mandatory **electronic XML** with УКЭП (CAdES-BES) of кадастровый инженер; PDF optional.

### 3.2 Технический план structure (per П/0082)
**Текстовая часть (9 sections)**: общие сведения → исходные данные → геодезическая сеть и средства измерений → описание местоположения → характеристики объекта → характеристики помещений/машино-мест → части объекта → заключение кадастрового инженера → приложение.

**Графическая часть (4 sections)**: схема геодезических построений → схема расположения здания → чертёж контура → **план здания / план этажа / фрагмент плана**.

Cover page fields: "Технический план", дата ДД.ММ.ГГГГ, cadastral engineer ФИО + № аттестата + СРО + СНИЛС + контакты, "Лист № __ Всего листов __". Formats: A4 portrait (text), A4/A3 landscape (graphic).

**Key rendering rules from П/0082 §59–76:**
- §67: **precision 0.5 mm on paper** ("в том числе средствами компьютерной графики").
- §68: plan placed symmetrically; **facade parallel to bottom edge**; if indeterminate, south at bottom.
- §69: show in scale — stены/перегородки, окна/двери, лестницы, балконы, internal wall protrusions, conventional room/parking markers.
- §71: floor type/number centered above plan ("1-й этаж", "Цокольный этаж").
- §72: linear dimensions **parallel to the walls they measure**.
- §76: room boundary = **internal** face of enclosing walls (not external).

**Allowed conventional signs (Приложение 2)** — only 6 prescribed: стена с окном и дверью; лестница; дверь остеклённая; веранда; терраса; перегородка. Additional symbols permitted with explicit legend.

### 3.3 Cadastral number format
`AA:BB:CCCCCCC:KK`
- AA = region code (77 Москва, 78 СПб, 50 МО, 54 Новосибирская обл.)
- BB = кадастровый район
- CCCCCCC = квартал (6–7 digits with leading zeros)
- KK = object № (1–7 digits, no leading zeros)

Example: `54:35:0101001:247` (hypothetical Novosibirsk).

### 3.4 Line weights (modern CAD, 1:100 on A3/A4)
| Element | Weight (mm) | Line type |
|---|---|---|
| Capital wall outline | 0.50–0.70 | сплошная толстая основная (s) |
| Capital wall fill | solid black poché **or** double-line hollow (≈50/50 split) | |
| Partition | 0.25–0.35 | сплошная тонкая |
| Door leaf, window frame, fixture outlines | 0.18–0.25 | сплошная тонкая |
| Dimension + extension lines | 0.13–0.18 | сплошная тонкая |
| 45° tick marks (засечки) | 0.25, length 2–3 mm | — |
| Hatching 45° | 0.13, spacing 1–2 mm | сплошная тонкая |
| Axes | 0.18–0.25 | штрихпунктирная |
| Hidden/demolished | ≈s/2 | штриховая |
| Red perepланировка lines | 0.25–0.35 | pure red #FF0000 |

### 3.5 Fonts and text (modern)
- Primary technical font: **ISOCPEUR** (closest TTF to GOST Type B italic with Cyrillic) or **"GOST 2.304 type A/B"**, **"MIP GOST"**.
- Body text in tables: **Arial** or **Times New Roman** (explicitly permitted by ГОСТ Р 21.1101-2013 §5.4.7 for explanatory text).
- Heights: dimensions **2.5 mm** (sometimes 3.5); room labels **3.5 mm** (sometimes 5); titles 5–7 mm; sheet titles 10 mm.
- Italic slant 15° (75° from horizontal) for ISOCPEUR / GOST.
- Decimal comma; dimensions in mm without suffix; areas in m² with "м²".

### 3.6 Modern wall/door/window rendering
- Walls: **solid black poché** ~50% of documents; double-line hollow ~50%. Partitions almost always double-line hollow or single-thick.
- Doors: **90°** door-leaf angle dominates BTI/ЕГРН output; 30° more typical in проектная документация. Arc = **full 90° quarter-circle**, R = leaf width, weight 0.18–0.25 mm.
- Windows: **4 parallel lines** per ГОСТ 21.201-2011 most common (2 wall + 2 frame); 3-line still acceptable.
- Panoramic window: single mullion tick at center.
- Balcony glazing: **Z-shaped line (зигзаг)** along perimeter.

### 3.7 Symbol dimension table at 1:100 (modern)

| Element | Plan size (mm) | Line weight | Fill |
|---|---|---|---|
| Capital wall | 2.5–5 (250–500 real) | 0.6 outline | solid black OR hollow |
| Partition | 0.6–1.2 (60–120 real) | 0.3 outline | hollow |
| Door leaf | 6–9 long | 0.3 | — |
| Door arc | R=leaf width | 0.18 | — |
| Window (4-line) | across opening 12–18 wide | 0.25 | — |
| Radiator (when shown) | 8–10 × 1.5–2 | 0.2 | section ticks |
| Toilet D-shape | 3.7×5.8 + tank 1.8×3.7 | 0.25 | — |
| Bathtub | 17 × 7 | 0.25 | inner offset 0.4 |
| Sink (bath) | 5 × 4 | 0.25 | drain Ø 0.8 |
| Kitchen sink | 5×6 (single) / 8×5 (double) | 0.25 | drain per basin |
| Shower | 9×9 | 0.25 | corner quarter-arc + drain |
| Gas stove | 5–6 × 6 | 0.25 | 4 circles Ø 1 |
| Vent shaft | 1.5–1.5 to 4×6 | 0.25 | 45° hatch OR solid |
| Elevator shaft | 15–25 × 15–25 | 0.6 | "X" cross + hatch |
| Column (ж/б) | 3–6 square | — | solid black fill |
| Built-in wardrobe | 6 × 10–20 | 0.25 | single diagonal |
| Balcony | 10 × 30 | guardrail 0.2 | empty (no Soviet hatch) |
| Loggia | 15 × 30–60 | 0.2 | empty |
| Stairs tread | spacing 2.5–3 per step | 0.18 | + arrow + "ВВЕРХ" |
| Засечка | length 2–4 at 45° | 0.25 | — |
| Room number circle | Ø 5–6 | 0.25 | — |
| North arrow (site plans only) | 20–30 long | 0.4 | filled head |

Note: **Radiators, refrigerators, dishwashers, movable furniture are NOT shown on BTI/ЕГРН plans** (explicit МосгорБТИ rule). Only sanitary fixtures and built-in wardrobes.

### 3.8 Dimension conventions (modern)
- **45° засечки 2–3 mm** still mandatory per ГОСТ Р 21.101-2020 §5.4.2 (NOT arrowheads).
- **External chains (typically 3)** from closest to wall outward:
  1. Openings + piers (10 mm from wall).
  2. Between capital-wall axes (+7–10 mm).
  3. Overall length (+7–10 mm).
- Internal: L×W per room, 8–10 mm from interior face.
- Text above line, centered, 2.5 mm height, gap 0.5–1 mm.
- **Dimensions in mm (integer, no units)** for interior чертёж; meters with comma for ситуационный план / общие характеристики.

### 3.9 Электронная подпись visualization (ЭЦП stamp)
- Blue rectangle ~**60×30 mm**, typically bottom-right of each page or last page only.
- Color: #0046C8 → #1F4E8C.
- Text: "ДОКУМЕНТ ПОДПИСАН ЭЛЕКТРОННОЙ ПОДПИСЬЮ" + serial number + владелец (ФИО) + validity dates.
- Thin blue border; sometimes with security pattern.
- May be rendered as light opacity watermark (20–30%).
- True signature is in accompanying `.sig` file; the visual stamp is only a визуализация.

**Note on QR codes**: **QR codes are NOT standard on ЕГРН выписки** as of 2026 — only the blue ЭЦП visualization. Some regional МФЦ variants add verification QR but it's not normative. Exclude QR by default from synthetic data unless simulating МФЦ print-outs.

### 3.10 Электронный формат
- **ZIP archive** containing:
  - `.xml` (machine-readable, the actual document)
  - `.xml.sig` (detached CAdES signature)
  - `.pdf` (human-readable визуализация)
  - `.pdf.sig`
- Page size predominantly **A4 (210×297 mm)**; A3 (297×420 mm) for full-floor МКД plans.
- Raster resolution when displayed: 150–300 DPI.
- Color mode: pure B&W; occasional red (#FF0000) for updates; blue (#0046C8) for signatures.
- **No paper aging** (unlike eras 1–2).

### 3.11 Modern apartment proportions

| Apartment type | Area (m²) | Typical footprint | Ceiling h |
|---|---|---|---|
| 1-комнатная / студия | 30–45 | 6–9 × 5–6 m | 2.5 / 2.7–3.0 / 3.0–3.5 |
| 2-комнатная | 45–70 | 8–12 × 6–8 m | — |
| 3-комнатная | 70–100 | 10–14 × 7–10 m | — |
| 4+ / элит | 100–200+ | irregular, with diagonals, curved partitions | — |

Wall thicknesses: external panel 300–400, brick 380–510, monolith 250–300; internal bearing 150–200; partitions 80–120 mm. Doors: interior 700–900; entry 900–1000. Windows: 1200–1800 standard; 2100–2400 living rooms; 600–800 bathrooms.

Scale mapping for A4 render:
- 1:100 → 45 m² flat ≈ 60×75 mm plan
- 1:200 → full МКД floor fits
- 1:50 → single detailed room

### 3.12 ЕГРН выписка Раздел 5 (simplified floor plan)
- A4 vertical.
- Header: "Раздел 5. План расположения помещения, машино-места на этаже (плане этажа)".
- Cadastral number prominently.
- **Simplified plan** — whole floor outline with target помещение **highlighted** (thick bold border or tinted fill, often red).
- Footer: "Лист № 5 Раздел 5 Всего листов раздела: __".
- If plan unavailable: "Сведения, необходимые для заполнения раздела отсутствуют".

### 3.13 Экспликация modern format
Tabular columns:
| № | Обозначение | Наименование | Площадь (м²) |

Extended (BTI-style):
| № | Наименование | Площадь | Высота потолка | Материал стен |

**Room name vocabulary (modern)**: Жилая комната / Спальня / Гостиная; Кухня / Кухня-ниша / Кухня-столовая; Коридор / Прихожая / Холл; Санузел совмещённый / Ванная / Туалет; Лоджия (коэфф. 0,5) / Балкон (коэфф. 0,3); Кладовая / Гардеробная / Постирочная; Тамбур / Встроенный шкаф.

**Area classifications**: общая площадь (with balcony coeff since 2021 per ст. 15 ЖК РФ amendments); жилая площадь; вспомогательная (auxiliary); площадь квартиры (without balconies); площадь с учётом летних помещений.

---

## 4. COMMON ELEMENTS ACROSS ERAS

### 4.1 Regional variations

**МосгорБТИ (Moscow)** — since 19.06.2015 ГБУ, prior ГУП.
- Monopoly on Moscow technical inventarization (ПП Москвы 106-ПП).
- Post-2024 issues documents **exclusively electronic** with УКЭП; paper copies supplementary.
- Supports **.dwg export** (10,03 ₽/m² vs 8,14 ₽/m² standard).
- Stamp: double-headed eagle (pre-2015 ГУП) OR Moscow coat of arms — **St. George slaying dragon** (ГБУ post-2015).
- Preferred scale: **1:200** for apartments (1:100 for small/single rooms).
- **Tradition of красные линии** for unauthorized redevelopment — defining Moscow visual marker.
- Legend still uses Forms №22 (поэтажный план) and №25 (экспликация) from Приказ №37.
- Archive covers inherited inventories from ГУП Мосгоргеотрест, ОАО "Госземкадастрсъемка — ВИСХАГИ", ФГУП Ростехинвентаризация, ГУП МО МОБТИ (as of 01.01.2013).

**ПИБ / ГБУ "ГУИОН" (St. Petersburg)**
- Full name: **Городское управление инвентаризации и оценки недвижимого и движимого имущества**.
- **ПИБ = Проектно-инвентаризационное бюро** (district branches, now 5 consolidated client centers).
- Pre-1995 archival cutoff: **03.04.1995** — ПИБ issues "справки о наличии/отсутствии строений зарегистрированных до 03.04.1995".
- Stamp: **St. Petersburg coat of arms** (two crossed anchors — sea + river — with scepter, red shield).
- Title often uses "ПЛАН" not "поэтажный план"; archival sheets labeled "Ленгорисполком, БТИ, [район]" through 1991.
- Unique document: **выписка из реестровой книги ПИБ** (not found in Moscow).
- Uses **MapInfo GIS** (MosgorBTI uses AutoCAD + proprietary).

**АО "Ростехинвентаризация — Федеральное БТИ" (ФБТИ)**
- Federal AO liquidated **31.10.2022**; successor **ППК "Роскадастр"**.
- Pre-liquidation covered ~80 regional filials outside Moscow/SPb.
- Stamp: Russian Federation double-headed eagle + "АКЦИОНЕРНОЕ ОБЩЕСТВО 'РОСТЕХИНВЕНТАРИЗАЦИЯ — ФЕДЕРАЛЬНОЕ БТИ'" + ОГРН + филиал.
- Follows Приказ МЭР №37 formulary strictly; typically includes **Схема расположения объекта** (situational plan) even for apartments (unlike МосгорБТИ).
- Title block bottom-right.

**Novosibirsk (user's region)**
- **ГБУ НСО "Новосибирский центр кадастровой оценки и инвентаризации" (ГБУ НСО «ЦКО и БТИ»)**, noti.ru.
- Address: ул. Сибирская, 15 (also ул. Ленина, 1 for "БТИ Про").
- Reorganized **12.10.2018** from ОГУП «Техцентр НСО».
- Stamp: **Novosibirsk Oblast coat of arms** (two sables flanking shield with lighthouse).
- Follows federal format strictly; **no Moscow-style красные линии** on standard forms (applied only on inspection notes).
- Apartments 1:200; site plans 1:500/1:1000.
- Siberian regional notes: earthquake-resistance, permafrost indicator (сваи на вечной мерзлоте) in older archival plans.

**Other regional patterns**
- **Ekaterinburg**: СОГУП "Областной Центр недвижимости" / ЕМУП "БТИ г. Екатеринбурга" — Свердловская область emblem.
- **Kazan**: РГУП "БТИ Министерства строительства РТ" — bilingual Russian/Tatar, Tatarstan winged leopard emblem.
- **Rostov-on-Don**: ГУПТИ РО — unique "паспорт домовладения" with earlier detailed участок info.
- **Vladivostok**: КГУП "Приморский краевой центр ТИ" — federal format.
- **Krasnodar**: ГУП КК "Крайтехинвентаризация — Краевое БТИ" (kubbti.ru).

Deviations are moderate: stamp design (regional emblem), exact cartouche layout, default scale, occasional regional technical notes. **Moscow is the outlier** with the most unique conventions.

### 4.2 Object-type differences
- **Квартира**: single извлечённый floor plan of the apartment; sanitary fixtures; экспликация with жилая / подсобная / лоджии×0,5 / балконы×0,3. No ситуационный план.
- **Индивидуальный жилой дом (ИЖС)**: form per Приказ МЭР №244 (17.08.2006). Contains титульный лист, ситуационный план с участком, поэтажные планы всех этажей (подвал/чердак), экспликация, расчётно-инвентаризационная ведомость (Forms 1–3), outbuildings (гараж, баня).
- **Нежилое помещение**: renamed columns "основная / вспомогательная", категория пожарной опасности (А/Б/В1-В4/Г/Д) per ГОСТ 21.501-2018 for industrial.
- **Многоквартирный дом**: floor plans of every level (including подвал, технический этаж, чердак) + экспликации + конструктивное описание; tens to hundreds of pages.

### 4.3 Document types (coverage matrix)

| Document | Era | Issuer | Status |
|---|---|---|---|
| Поэтажный план | all | BTI | active |
| Экспликация | all | BTI | active |
| Технический паспорт | pre-2017 primary, still issued | BTI | active (required for перепланировка, газификация) |
| Кадастровый паспорт | 2008–2017 | Росреестр | repealed 01.01.2017 |
| Технический план | 2017+ | кадастровый инженер | active (Приказ П/0082) |
| Выписка из техпаспорта (Форма 1а/1б/4/5) | all | BTI | active |
| Справка БТИ | all | BTI | active (short validity 30 d – 1 yr) |
| Выписка из ЕГРН с планом помещения | 2017+ | Росреестр | active (replaces КП) |
| Plan without экспликация | all | informal / extracts | not legally binding |

### 4.4 Annotation conventions (all eras)
- **Ceiling height**: usually in экспликация, occasionally on plan as "h=2,50".
- **Level changes**: thin solid line + arrow ↑/↓ or "ступень −150"; elevation triangle with +X,XXX / −X,XXX.
- **Mezzanine (антресоль)**: dashed rectangle (above cut plane) + label "антр. h=…".
- **Multi-level apartments (разноуровневая квартира)**: one floor plan per level, with elevation marks.
- **Material codes** (in passport data, not as plan hatching): К (кирпич), Б (блок), Бет (бетон), Ж/Б (железобетон), Д (дерево), Пан (панель), Моно (монолит), Кар (каркас), Г/Б (газобетон), П/Б (пенобетон), Сил (силикатный), См (смешанные).
- **Notes section (Примечания)**: below plan or in separate block of passport; blue-ballpoint handwritten or typed.
- **Exterior perimeter markings**: наружные dimensions in 3 chains; на ИЖС additionally указывают ориентацию (north arrow 20–30 mm with "С" letter, top-right or bottom-right).

---

## 5. IMPLEMENTATION CHEATSHEET

### 5.1 Master line-weight table (1:100 scale, A3/A4 paper)

| Element | Soviet | Transitional | Modern |
|---|---|---|---|
| Capital wall contour | 0.5–0.7 mm | 0.25 uniform OR 0.50–0.70 CAD | 0.50–0.70 |
| Partition contour | 0.3–0.4 | 0.25 uniform | 0.25–0.35 |
| Wall fill | hollow (double-line) | hollow or solid polyfill | ~50% solid, ~50% hollow |
| Door leaf | 0.3 | 0.25–0.3 | 0.18–0.25 |
| Door swing arc | 0.18 | 0.18 | 0.18 |
| Window frame | 0.25 | 0.25 | 0.18–0.25 |
| Dimension line | 0.18–0.25 | 0.18 | 0.13–0.18 |
| Extension line | 0.18–0.25 | 0.18 | 0.13–0.18 |
| Tick (засечка) | 0.3–0.5, L=2–4 mm, 45°R | 0.25, L=2–4, 45°R | 0.25, L=2–3, 45°R |
| Hatching | 0.1–0.18, spacing 1–3 mm | 0.18, 2 mm | 0.13, 1–2 mm |
| Red перепланировка | 0.3 (rare) | **0.3–0.4 #C8202B** | 0.25–0.35 #FF0000 |
| Stamps | Ø40 violet/blue/red | Ø38–42 blue #1E3A8A | digital ЭЦП #0046C8 rect 60×30 |
| Jitter (stroke) | ±15%, waviness 0.1–0.3 mm | ±5% or 0 for CAD | 0 |
| Endpoint overshoot | 0.3–1.0 mm, 50% prob | 0–0.3 mm, 20% prob | 0 |

### 5.2 Symbol dimension master table (1:100)

| Symbol | Real (mm) | Paper (mm) | Notes |
|---|---|---|---|
| Door interior Д-1 (bath/WC) | 600 | 6.0 | arc R=6 |
| Door Д-5 (kitchen) | 700 | 7.0 | arc R=7 |
| Door Д-2 (room) | 800 | 8.0 | arc R=8 |
| Door accessible | 900 | 9.0 | arc R=9 |
| Double door Д-9 | 1390 (600+800) | 13.9 | two arcs |
| Double door Д-10 | 1790 (800+900) | 17.9 | two arcs |
| Entry door | 900–1000 | 9–10 | leaf 0.35–0.5 |
| Window standard | 870/1170/1470/1770 | 8.7/11.7/14.7/17.7 | 3 or 4 lines |
| Sill height (irrelevant for plan) | 900 | — | — |
| Balcony | 1000×3000 | 10×30 | Soviet hatched; modern empty |
| Loggia | 1500×3000–6000 | 15×30–60 | recessed |
| Toilet | 370×700 + tank 180×370 | 3.7×7 + 1.8×3.7 | — |
| Bathtub | 700×1700 | 7×17 | drain one end |
| Sink (bath) | 500–600×400 | 5–6×4 | drain circle |
| Kitchen sink single | 500×600 | 5×6 | — |
| Kitchen sink double | 800×500 | 8×5 | divider mid |
| Shower | 900×900 | 9×9 | corner arc + drain |
| Gas stove 4-burner | 500–600×600 | 5–6×6 | 4 circles Ø 1 |
| Gas column | Ø 300–400 | Ø 3–4 | circle on wall |
| Masonry stove (Soviet) | 800–1500 square | 8–15 | 45° hatched |
| Radiator | 1000×100–200 | 10×1.5–2 | 50 mm from wall (0.5) |
| Vent shaft | 150–400 × 150–600 | 1.5–4 × 1.5–6 | cross-hatch (Soviet) / 45° (modern) |
| Elevator shaft | 1500–2500 square | 15–25 | X cross + hatch |
| Garbage chute | 400–500 sq + pipe | 4–5 sq + Ø inner | label М |
| Column ж/б | 300–600 sq | 3–6 sq | solid black |
| Built-in wardrobe | 600×1000–2000 | 6×10–20 | diagonal X |
| Staircase tread | 250–300 | 2.5–3 | spacing |
| North arrow | 2000–3000 | 20–30 | "С" letter |

### 5.3 Font size table

| Text | Height (mm) | Font |
|---|---|---|
| Dimensions | **2.5** (sometimes 3.5) | GOST type B italic / ISOCPEUR |
| Areas (in room) | 3.5 | Same, underlined |
| Room numbers | 3.5–5 | Same, in Ø 5–8 circle |
| Ceiling height "h=2,50" | 2.5 | Same |
| Room type labels | 3.5 | Same |
| Abbreviations (Ж/К/С.У.) | 3.5 | Same |
| Title block entries | 3.5–5 | Same |
| Section/sheet titles | 5–7 | Same |
| Main heading (ПОЭТАЖНЫЙ ПЛАН) | 7–10 | Same |
| Marginalia (ballpoint) | 2–4 | cursive, non-standard |

### 5.4 Dimension chain geometry (all eras)

```
offset from wall to chain 1 :   14–21 mm (14 typical)
chain 1 to chain 2          :   7–10 mm
chain 2 to chain 3          :   7–10 mm
extension line past dim     :   1–5 mm
dim line past extension     :   0–3 mm
tick (засечка) length       :   2–4 mm
tick angle                  :   exactly 45° right-slant
text gap above dim line     :   0.5–1 mm
text height                 :   2.5 mm
decimal separator           :   COMMA (,)
units (interior mm)         :   no suffix
units (meters, area)        :   "м²" / "м" with 1–3 decimals
```

### 5.5 Era preset summary

**Soviet (pre-1990):**
- Substrate: aged калька (#E6D7A0) or diazo cream (#F5EBC8) with blue-violet #28378C lines OR sepia #6E4B28 OR fresh black #050505→aged #37281A.
- Stroke jitter ±15%, waviness 0.1–0.3 mm, overshoots 0.3–1.0 mm (50% prob), 1% ink-blot rate.
- Walls hollow double-line; no fill.
- Doors at 30° (60%) or 90° (40%), arcs slightly flattened (eccentricity 0.05–0.15).
- Windows: 3 lines.
- Balconies: 45° diagonal hatch inside, 2 mm spacing.
- Vent shafts: cross-hatched сетка.
- No radiators, no free furniture.
- Lettering hand ГОСТ type A italic 75°, per-letter rotation jitter ±3°, baseline drift.
- Stamps: round Ø40 violet/blue/red, double-strike 30%, partial fade 30%, ±10° rotation.
- Decimal comma, meters 2 decimals.
- Aging: foxing, tea stains, folds, punch holes, light-fade vignette, 40% blue-ballpoint marginalia.

**Transitional (1992–2015):**
- Mix tiers: 30% hand-drawn retraced, 20% typewritten overlay, 50% CAD (AutoCAD/PlanTracer).
- Uniform 0.25 mm "monoblock" linework on pure raster scans; multi-weight on CAD-origin.
- Walls: hollow, or solid poché in CAD tier.
- Doors 30°:90° = 30:70 (BTI skews 90°).
- Windows: 3 or 4 lines.
- Radiators sometimes present.
- Red-line перепланировка overlay (#C8202B) on ~25% of samples; red entry-door number on 100%.
- Stamps: blue #1E3A8A round Ø38–42 with RF double-eagle + at least 2 impressions per page.
- Post-2008: cadastral engineer round stamp.
- Scanner/copier artifacts heavily applied: skew ±3°, edge shadow, JPEG Q70–85, dust, roller streaks, fold lines, staple holes.
- Paper palette: cream → tan.

**Modern (2015+):**
- Pure white #FFFFFF, pure black #000000; no aging.
- Two-weight CAD linework: 0.5/0.2 (walls/detail).
- Walls 50% poché solid, 50% hollow double-line.
- Doors 30° or 90° equally (GOST compliance increasing).
- Windows: 4-line dominant per ГОСТ 21.201-2011.
- Z-line balcony glazing.
- Panoramic windows and монолит columns in apartment plans.
- Fonts: ISOCPEUR / GOST type A italic, or Arial/Times for body tables.
- Blue ЭЦП visualization rectangle #0046C8 60×30 mm bottom-right.
- Cadastral number prominent, format `AA:BB:CCCCCCC:KK`.
- Electronic artifact: vector-crisp, no scan noise; occasional rasterize-from-PDF at 150 DPI with slight anti-aliasing.
- NO QR codes by default.
- Floor label top-center ("1-й этаж").
- Facade parallel to bottom edge; precision 0.5 mm on paper.

### 5.6 Render pipeline (recommended layer order)

1. **Substrate layer** — sample era-appropriate background and texture.
2. **Base linework layer** — walls, partitions, openings, fixtures, stairs (per §1–3 era).
3. **Symbol layer** — fixtures and built-in elements with ±2° rotation, ±10% scale, ±1 mm position jitter.
4. **Text layer** — dimensions, labels, areas (with font per era).
5. **Red overlay** — перепланировка + entry number (for transitional/modern).
6. **Annotation layer** — room numbers in circles, ceiling-height notes.
7. **Title block + stamps** — bottom-right cartouche, round/oval/rectangular stamps.
8. **Handwritten marginalia** — blue ballpoint / pencil (Soviet + transitional).
9. **Whiteout corrections** — opaque chalky rectangles with overtyped text.
10. **Aging layer** — foxing, tea stains, folds, tape, punch holes (Soviet + transitional).
11. **Photocopy/scan pipeline** — blur → contrast clip → grayscale/posterize → speckle/dust/streaks → rotate → crop → edge shadow → JPEG (transitional only).
12. **Digital signature stamp** — blue #0046C8 rectangle (modern only).

### 5.7 Class-label guidance for U-Net (5 classes)

| Class | Includes | Excludes |
|---|---|---|
| **background** | paper substrate, empty rooms, dimension lines, text labels, stamps, title block, all annotations, exterior of apartment, stairs, balconies/loggias | — |
| **wall** | all thick linework of capital walls and partitions, including solid poché fill if present; elevator/vent shaft outlines | door/window openings (class: door/window) |
| **window** | window-in-wall region (double/triple/quadruple-line span), including balcony glazing zones | radiators below |
| **door** | door opening region including leaf line + swing arc; arch openings; sliding/pocket door cavities | the wall segments flanking the opening |
| **furniture** | all sanitary fixtures (toilet, bathtub, sink, shower), kitchen stove, built-in wardrobe with diagonal mark, radiators (if rendered), garbage chute symbol, columns if inside living space | walls of kitchen/bathroom rooms (class: wall) |

Rationale: BTI plans do not show free-standing furniture, so "furniture" class in this dataset = fixed fixtures + built-in storage + structural-but-enclosed elements (columns, shafts). This matches the realistic distribution of what appears on real BTI plans.

### 5.8 Legal/stamp text templates (for stamp generator)

**Soviet title header (pre-1977):**
```
ИСПОЛКОМ ЛЕНИНГРАДСКОГО ГОРОДСКОГО СОВЕТА ДЕПУТАТОВ ТРУДЯЩИХСЯ
ОТДЕЛ КОММУНАЛЬНОГО ХОЗЯЙСТВА
БЮРО ТЕХНИЧЕСКОЙ ИНВЕНТАРИЗАЦИИ ФРУНЗЕНСКОГО РАЙОНА
```

**Soviet title header (1977–1991):**
```
ИСПОЛКОМ ... СОВЕТА НАРОДНЫХ ДЕПУТАТОВ
...
```

**Transitional stamp text samples:**
- "ГУП МосгорБТИ · Для документов · №"
- "КОПИЯ ВЕРНА · [дата] · подпись"
- "РАЗРЕШЕНИЕ НА РЕКОНСТРУКЦИЮ (ПЕРЕПЛАНИРОВКУ) НЕ ПРЕДЪЯВЛЕНО"
- "ПЕРЕПЛАНИРОВКА"
- "ВЫДАНО ДЛЯ ПРЕДЪЯВЛЕНИЯ В ______"

**Modern ЭЦП stamp text:**
```
ДОКУМЕНТ ПОДПИСАН
ЭЛЕКТРОННОЙ ПОДПИСЬЮ
Сертификат: 01A2B3C4D5E6...
Владелец: Иванов И.И.
Действителен: с 01.02.2024 по 01.02.2025
```

### 5.9 Procedural generation priorities

For maximum dataset realism, vary these dimensions independently per sample:

1. **Era** — 30% Soviet / 40% Transitional / 30% Modern (adjust to training needs).
2. **Region** — 30% Moscow / 15% SPb / 10% Novosibirsk / 45% generic federal.
3. **Apartment series/layout** — sample from Сталинка/Хрущёвка/Брежневка/Улучшенка/новостройка/монолит families.
4. **Object type** — 80% квартира / 10% жилой дом / 10% нежилое.
5. **Wall fill style** — hollow vs solid, independently randomized (not tied to structure).
6. **Door leaf angle** — 30° vs 90° by era distribution.
7. **Window line count** — 2/3/4 by era.
8. **Vent shaft style** — cross-hatch vs 45° hatch vs solid (era marker).
9. **Balcony fill** — hatched (Soviet) vs empty (modern).
10. **Presence of перепланировка red lines** — 20–30% of transitional/modern.
11. **Unauthorized marker stamp** — 15% of transitional.
12. **Paper aging level** — continuous 0–1 for Soviet/transitional.
13. **Scanner artifact intensity** — continuous 0–1 for transitional.
14. **Rotation** — ±3° for scanned samples.
15. **Stamp count** — 1–4 per page; at least 1 on plan, 1 on экспликация.

---

## Conclusion: synthesis notes for implementation

Three visual eras define a single visual language that drifted under pressure from technology (рапидограф → AutoCAD → XML), institutional change (Гор­исполком БТИ → ГУП → ГБУ → кадастровый инженер), and regulatory reform (Приказ ЦСУ 380/1985 → Приказ Минземстроя 37/1998 → Приказ МЭР 953/2015 → Приказ Росреестра П/0082/2022). Era-authenticity for synthetic BTI data reduces to three concentric constraints: correct **linework mechanics** (weights, jitter, colors), correct **symbol vocabulary** (doors/windows/fixtures per §1.11, §3.7, Appendix §5.2), and correct **document chrome** (title block text, stamps, red-line overlays, ЭЦП visualizations).

The single most important rendering rule — and the most commonly misstated belief about BTI plans — is that **line thickness does not indicate load-bearing status**. A synthetic generator that randomizes inner/outer wall thickness independently of structural role will produce plans that are both realistic and honest to the source material's actual information content. This matters for downstream tasks: a U-Net trained on plans where "thicker wall = несущая" will learn a signal that does not exist in the real data distribution and will overfit.

Second-most important: the **красные линии** tradition of МосгорБТИ. This is the signature visual feature distinguishing Moscow BTI from federal/regional output and from proektная документация; it is also semantically dense (redevelopment detection). Including это reliably in the transitional-era preset gives the model a discriminative feature for real-world document triage.

Third: the **three-era distribution of window line counts (2 / 3 / 4)** and **vent-shaft hatching style (cross / diagonal / solid)** are strong era markers — include both as controlled generator variables to let the segmentation network generalize across eras without memorizing single conventions.

With these parameters, the generator should produce a dataset in which era, region, and object-type variation covers the realistic space of plans a production CV pipeline in Novosibirsk or anywhere in Russia will encounter, from a 1974 хрущёвка архивная копия on diazo paper through a 2005 МосгорБТИ A5 booklet with red-line самовольная перепланировка overlays to a 2025 XML технический план with CAdES blue-stamp visualization.