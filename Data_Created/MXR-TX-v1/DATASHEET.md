# MXR-TX-v1 — real bitemporal window pairs, SmokeHouse Creek wildfires (Texas panhandle, Feb 2024)
Built with the identical protocol as MXR-LA-v1 (`scripts/create_maxar_event.py`, seed 20260930; see the LA datasheet for
window size/resampling/validity rules). **Second real event, used only as a pristine held-out test (E09).**
**Source.** Maxar Open Data event `SmokeHouseCreek-Wildfires-Texas-Mar24` (CC BY-NC 4.0), 16 acquisitions 2021-07 →
2024-03 (WorldView-2, GeoEye-1), UTM 14N; NIFC perimeters Smokehouse Creek (OBJECTID 24509, 1.05 M ac), Windy Deuce
(37174), Grape Vine Creek (24516), fires discovered 26–27 Feb 2024 (`data/raw/maxar/tx_fire_perimeters_2024.geojson`).
**Parameters.** pre ≤ 2024-02-25, post ≥ 2024-02-29; change/null_long t1 in 2023-06-01…2023-12-31; caps 100/100/60/120.

| label | n | quadkeys | median Δt | median Δoff-nadir | cross-sensor |
|---|---|---|---|---|---|
| change (inside perimeters, 2023 vs Mar 2024) | 120 | 33 | 216 d | 6.2° | 100 % |
| null_long (outside, 2023 vs Mar 2024) | 60 | 21 | 140 d | 4.4° | 77 % |
| null_pre (≤ 14 d apart, pre-fire) | 100 | 25 | 0 d (mostly **same-day**, seconds apart) | 5.9° | 24 % |
| null_post (1 vs 4 Mar 2024, outside) | 18 | 6 | 0 d | 5.6° | 28 % |

**Differences from LA:** grassland/rangeland + small towns (Fritch, Canadian) instead of urban-wildland; winter vs autumn
season; mostly WV-2/GE-1; 89 same-day pairs isolate pure view-geometry nuisance. **Confound:** change pairs are months
apart and cross-sensor, most nulls are same-day → the fair comparison is change vs `null_long` (same gap, sensors mostly
different); change vs all nulls is reported but is easier. Burned grassland is a strong radiometric change.
