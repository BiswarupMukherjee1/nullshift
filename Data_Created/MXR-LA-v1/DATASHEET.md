# MXR-LA-v1 — real bitemporal window pairs (Maxar Open Data, Los Angeles wildfires, Dec 2024 – Jan 2025)

**Purpose.** A *real* arena for NullShift, addressing two limitations of synthetic null pairs: LEVIR-CD pairs are 5–14 years apart
(natural differences hide instrument nuisance), and synthetic null pairs build the operator from the very image that is
perturbed. Here t1 and t2 are **different satellite acquisitions** of the same place, and the operator only sees t1.
**No manual labelling**: labels come from acquisition dates and public burn perimeters.

**Source.** Maxar Open Data Program, event `WildFires-LosAngeles-Jan-2025` (STAC:
`https://maxar-opendata.s3.amazonaws.com/events/WildFires-LosAngeles-Jan-2025/collection.json`), licence CC BY-NC 4.0.
40 acquisitions (WorldView-2, WorldView-3, GeoEye-1), 14 Dec 2024 – 20 Jan 2025, off-nadir 7–39°. Maxar ARD visual
COGs: all acquisitions of a tile share one 0.305 m UTM-11N grid (17408² px, 5.3 km tiles) → pairs are aligned by the
provider's orthorectification; what remains is *real residual* misregistration + building lean (no DSM in ARD).
Burn perimeters: NIFC WFIGS Interagency Perimeters, Palisades (OBJECTID 36398, 23,448 ac) and Eaton (53438, 14,021 ac).
Metadata scan: `data/raw/maxar/WildFires-LosAngeles-Jan-2025_items.csv`; perimeters `data/raw/maxar/la_fire_perimeters_2025.geojson`.

**Pair types** (fires started 7 Jan 2025):
| label | t1 / t2 | window location | meaning |
|---|---|---|---|
| `null_pre` | two pre-fire acquisitions, ≤ 14 days apart | anywhere | near-null: real nuisance only |
| `null_post` | two post-fire acquisitions, ≤ 7 days apart | outside perimeters + 300 m | near-null |
| `null_long` | pre (≤ 1 Jan) vs post (≥ 9 Jan) | outside perimeters + 300 m | near-null over 1–5 weeks |
| `change` | pre (≤ 1 Jan) vs post (≥ 9 Jan) | centre ≥ 60 m inside a perimeter | real destruction (burned houses and/or vegetation) |

**Windows.** 192 m × 192 m read with antialiased `average` resampling to **0.5 m** (LEVIR's GSD) → 384² px =
256² evaluated + 64 px margin (same pipeline as NSB-v1). Validity: < 0.5 % black pixels and mean brightness < 0.8 in
both images. ≤ 4 windows per acquisition pair; caps 150/150/100/200. Seed 20260929. Files: `images/<pair_id>_t{1,2}.png`,
`manifest.csv` (dates, platforms, off-nadir t1/t2 and difference, view azimuths, sun-elevation difference, GSDs,
cross-sensor flag).

**Known limitations.** (1) "Near-null" is not exactly null: cars, construction, cleanup, seasonal vegetation and shadow
changes occur within days. (2) `change` windows inside a perimeter include unburned patches and much burned vegetation
(wildland), not only destroyed buildings → window-level labels are weak; patch-level positives are noisy. Both effects
dilute all methods equally, so comparisons remain valid, absolute AUROCs are conservative. (3) Single event/region;
dry-season Californian landscape. (4) CC BY-NC: research use only; do not redistribute images.
