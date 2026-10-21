"""Scan a Maxar Open Data event STAC (metadata only) -> CSV of ARD tile items: acquisition id, platform, datetime,
quadkey, off-nadir, sun elevation/azimuth, GSD, visual COG URL. Usage: python scripts/scan_maxar_event.py <Event-Name>"""
import concurrent.futures as cf
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://maxar-opendata.s3.amazonaws.com/events/"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def join(base, href):
    return urllib.parse.urljoin(base, href)


def scan_acq(url):
    c = get(url)
    rows = []
    for l in c["links"]:
        if l["rel"] != "item":
            continue
        iu = join(url, l["href"])
        it = get(iu)
        p = it["properties"]
        vis = it["assets"].get("visual", {}).get("href")
        rows.append({"acq": p.get("catalog_id"), "platform": p.get("platform"), "datetime": p.get("datetime"),
                     "quadkey": p.get("quadkey"), "grid_code": p.get("grid:code"), "off_nadir": p.get("view:off_nadir"),
                     "azimuth": p.get("view:azimuth"), "sun_el": p.get("view:sun_elevation"),
                     "sun_az": p.get("view:sun_azimuth"), "gsd": p.get("gsd"), "epsg": p.get("proj:epsg"),
                     "cloud": p.get("tile:clouds_percent"), "visual": join(iu, vis) if vis else None, "item": iu})
    return rows


if __name__ == "__main__":
    ev = sys.argv[1]
    col = get(BASE + ev + "/collection.json")
    acqs = [join(BASE + ev + "/collection.json", l["href"]) for l in col["links"] if l["rel"] == "child"]
    rows = []
    with cf.ThreadPoolExecutor(8) as ex:
        for r in ex.map(scan_acq, acqs):
            rows += r
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "raw" / "maxar" / f"{ev}_items.csv"
    df.to_csv(out, index=False)
    print(len(df), "items;", df.acq.nunique(), "acquisitions;", df.quadkey.nunique(), "tiles ->", out)
