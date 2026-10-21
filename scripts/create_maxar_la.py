"""Create MXR-LA-v1: REAL bitemporal window pairs from Maxar Open Data (LA wildfires, Dec 2024 - Jan 2025).

Why: LEVIR pairs are 5-14 years apart, so natural differences dwarf acquisition nuisance, and synthetic null pairs build
the operator from the very image that is perturbed. Here both images are genuinely different acquisitions (different
satellite / off-nadir / sun / date) of the same ARD tile, already on Maxar's common 0.305 m grid.
Pair types (window label, no manual annotation):
  null_pre  : two pre-fire acquisitions (<= 2025-01-07), |dt| <= 14 d            -> near-null (real nuisance only)
  null_post : two post-fire acquisitions (>= 2025-01-09), |dt| <= 7 d, window outside perimeters + 300 m
  null_long : pre vs post, window outside perimeters + 300 m                     -> near-null, longer gap
  change    : pre (<= 2025-01-01) vs post (>= 2025-01-09), window fully inside a burn perimeter (Palisades / Eaton)
Windows: 192 m (384 px at 0.5 m = 256 evaluated + 64 margin), antialiased 'average' resampling from 0.305 m, identical
for both dates. Validity: < 0.5 % black (outside footprint) and mean brightness < 0.8 (cloud/haze screen) on both.
Usage: PYTHONPATH=src python scripts/create_maxar_la.py   (network-bound, ~1 h)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import shapely.geometry as sg  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.warp import transform_geom  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "maxar"
OUT = ROOT / "Data_Created" / "MXR-LA-v1"
SEED = 20260929
WIN_M, OUT_PX = 192.0, 384
PRE_END, POST_START, CHANGE_PRE_END = pd.Timestamp("2025-01-07T23:59Z"), pd.Timestamp("2025-01-09T00:00Z"), pd.Timestamp("2025-01-01T23:59Z")
CAPS = {"null_pre": 150, "null_post": 150, "null_long": 100, "change": 200}
PER_PAIR = 4          # windows per acquisition pair
MAX_TRIES = 30


def read_window(url, bounds):
    with rasterio.open(url) as r:
        w = from_bounds(*bounds, transform=r.transform)
        a = r.read(window=w, out_shape=(3, OUT_PX, OUT_PX), resampling=Resampling.average, boundless=True, fill_value=0)
    return np.transpose(a, (1, 2, 0))


def valid(a):
    black = (a.max(-1) == 0).mean()
    return black < 0.005 and a.mean() / 255 < 0.8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(RAW / "WildFires-LosAngeles-Jan-2025_items.csv", parse_dates=["datetime"])
    df = df[df.visual.notna() & (df.cloud.fillna(0) <= 10)]
    per = json.load(open(RAW / "la_fire_perimeters_2025.geojson"))
    polys = [sg.shape(transform_geom("EPSG:4326", "EPSG:32611", f["geometry"])).buffer(0) for f in per["features"]]
    fire = unary_union(polys)
    fire_buf = fire.buffer(300)
    fire_in = fire.buffer(-60)          # window centre >= 60 m inside -> most of the 192 m window inside
    rows, counts = [], {k: 0 for k in CAPS}
    pairs = []
    for qk, g in df.groupby("quadkey"):
        g = g.sort_values("datetime").drop_duplicates("acq")
        recs = g.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                dt = abs((b["datetime"] - a["datetime"]).total_seconds()) / 86400
                if b["datetime"] <= PRE_END and dt <= 14:
                    typ = "null_pre"
                elif a["datetime"] >= POST_START and dt <= 7:
                    typ = "null_post"
                elif a["datetime"] <= CHANGE_PRE_END and b["datetime"] >= POST_START:
                    typ = "prepost"      # -> change (inside) or null_long (outside), decided per window
                else:
                    continue
                pairs.append((typ, a, b, dt))
    rng.shuffle(pairs)
    print(f"{len(pairs)} candidate acquisition pairs", flush=True)
    for typ, a, b, dt in pairs:
        if all(counts[k] >= CAPS[k] for k in CAPS):
            break
        with rasterio.open(a["visual"]) as r:
            L, B, R, T = r.bounds
        tile = sg.box(L, B, R, T)
        want = ["change", "null_long"] if typ == "prepost" else [typ]
        if all(counts[w] >= CAPS[w] for w in want):
            continue
        if typ == "prepost" and not tile.intersects(fire) and counts["null_long"] >= CAPS["null_long"]:
            continue
        got = 0
        for _ in range(MAX_TRIES):
            if got >= PER_PAIR:
                break
            cx, cy = rng.uniform(L + WIN_M / 2, R - WIN_M / 2), rng.uniform(B + WIN_M / 2, T - WIN_M / 2)
            pt = sg.Point(cx, cy)
            if typ == "prepost":
                lab = "change" if fire_in.contains(pt) else ("null_long" if not fire_buf.contains(pt) else None)
            elif typ == "null_post":
                lab = "null_post" if not fire_buf.contains(pt) else None
            else:
                lab = "null_pre"
            if lab is None or counts[lab] >= CAPS[lab]:
                continue
            bounds = (cx - WIN_M / 2, cy - WIN_M / 2, cx + WIN_M / 2, cy + WIN_M / 2)
            try:
                ia, ib = read_window(a["visual"], bounds), read_window(b["visual"], bounds)
            except Exception as e:  # network hiccup: skip window
                print("read error", e, flush=True)
                continue
            if not (valid(ia) and valid(ib)):
                continue
            pid = f"{lab}_{counts[lab]:04d}"
            Image.fromarray(ia).save(OUT / "images" / f"{pid}_t1.png")
            Image.fromarray(ib).save(OUT / "images" / f"{pid}_t2.png")
            rows.append({"pair_id": pid, "label": lab, "quadkey": a["quadkey"], "cx": round(cx, 1), "cy": round(cy, 1),
                         "epsg": 32611, "acq_t1": a["acq"], "acq_t2": b["acq"], "platform_t1": a["platform"],
                         "platform_t2": b["platform"], "date_t1": str(a["datetime"]), "date_t2": str(b["datetime"]),
                         "dt_days": round(dt, 2), "offnadir_t1": a["off_nadir"], "offnadir_t2": b["off_nadir"],
                         "d_offnadir": round(abs(a["off_nadir"] - b["off_nadir"]), 2),
                         "view_az_t1": a["azimuth"], "view_az_t2": b["azimuth"],
                         "d_sun_el": round(abs(a["sun_el"] - b["sun_el"]), 2), "gsd_t1": a["gsd"], "gsd_t2": b["gsd"],
                         "cross_sensor": a["platform"] != b["platform"]})
            counts[lab] += 1
            got += 1
            if len(rows) % 25 == 0:
                print(counts, flush=True)
                pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    print("final", counts, flush=True)
    print(man.groupby("label")[["dt_days", "d_offnadir", "d_sun_el"]].describe().round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
