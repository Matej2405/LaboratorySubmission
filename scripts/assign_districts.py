"""Assign each facility to a Census 2011 district via point-in-polygon.

Approach
--------
1. Load `data/reference/india_districts.geojson` (HindustanTimesLabs / DataMeet
   2011 boundaries). Each feature exposes `st_nm` (state) and `district`.
2. Build a `cKDTree` of polygon *centroids* so we can pre-filter to the
   nearest K candidates per facility instead of testing all 641 polygons.
3. For each facility (lat, lng), check the nearest 8 polygons with
   `shapely.geometry.Polygon.contains`. Fall back to the closest polygon if
   no exact hit (covers offshore points or Census boundary noise).

The function `assign_districts(df, lat_col, lng_col)` returns a copy with two
new columns:
    `district`    - district name (or None if no polygon available)
    `district_state` - state name from the GeoJSON (helpful for fuzzy-join)

Designed to run on a laptop in < 30 s for 10 k points.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon, shape

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOJSON = ROOT / "data" / "reference" / "india_districts.geojson"


class DistrictIndex:
    """Spatial index over Indian district polygons."""

    def __init__(self, geojson_path: Path = DEFAULT_GEOJSON):
        if not geojson_path.exists():
            raise FileNotFoundError(
                f"{geojson_path} missing - run `python scripts/fetch_reference_data.py` first."
            )
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        self.geoms: list[Polygon | MultiPolygon] = []
        self.props: list[dict] = []
        centroids: list[tuple[float, float]] = []
        for feat in gj["features"]:
            try:
                geom = shape(feat["geometry"])
            except Exception:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            self.geoms.append(geom)
            self.props.append(feat.get("properties", {}) or {})
            c = geom.centroid
            centroids.append((c.x, c.y))
        self._tree = cKDTree(np.array(centroids))
        log.info("DistrictIndex loaded %d polygons", len(self.geoms))

    def assign(self, lat: float, lng: float, k: int = 8) -> Optional[dict]:
        """Return the properties dict of the polygon containing (lat, lng).

        Falls back to the nearest centroid's polygon if no polygon contains
        the point (handles boundary noise or coastal facilities).
        """
        if lat is None or lng is None:
            return None
        try:
            lat = float(lat)
            lng = float(lng)
        except (TypeError, ValueError):
            return None
        if not (6.0 <= lat <= 38.0 and 68.0 <= lng <= 98.0):
            return None
        pt = Point(lng, lat)
        dists, idxs = self._tree.query([(lng, lat)], k=min(k, len(self.geoms)))
        idxs = np.atleast_1d(idxs).ravel()
        for i in idxs:
            geom = self.geoms[int(i)]
            try:
                if geom.contains(pt):
                    return self.props[int(i)]
            except Exception:
                continue
        nearest = int(idxs[0])
        return self.props[nearest]


_INDEX_SINGLETON: Optional[DistrictIndex] = None


def _index() -> DistrictIndex:
    global _INDEX_SINGLETON
    if _INDEX_SINGLETON is None:
        _INDEX_SINGLETON = DistrictIndex()
    return _INDEX_SINGLETON


def assign_districts(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lng_col: str = "longitude",
    *,
    geojson_path: Path = DEFAULT_GEOJSON,
) -> pd.DataFrame:
    """Return a copy of `df` with new `district` and `district_state` columns."""
    if not geojson_path.exists():
        out = df.copy()
        out["district"] = None
        out["district_state"] = None
        log.warning(
            "%s missing - skipping district assignment. Run scripts/fetch_reference_data.py.",
            geojson_path,
        )
        return out

    idx = DistrictIndex(geojson_path) if geojson_path != DEFAULT_GEOJSON else _index()
    districts: list[Optional[str]] = []
    states: list[Optional[str]] = []
    for lat, lng in zip(df[lat_col].tolist(), df[lng_col].tolist()):
        props = idx.assign(lat, lng)
        if props is None:
            districts.append(None)
            states.append(None)
        else:
            districts.append(props.get("district") or props.get("DISTRICT"))
            states.append(props.get("st_nm") or props.get("STATE") or props.get("state"))
    out = df.copy()
    out["district"] = districts
    out["district_state"] = states
    return out


def assign_iter(
    coords: Iterable[tuple[float, float]],
    *,
    geojson_path: Path = DEFAULT_GEOJSON,
) -> list[Optional[dict]]:
    """Convenience generator-style API for one-off lookups."""
    idx = DistrictIndex(geojson_path) if geojson_path != DEFAULT_GEOJSON else _index()
    return [idx.assign(lat, lng) for lat, lng in coords]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries",
        default=str(ROOT / "data" / "cache" / "facility_summaries.parquet"),
        help="Path to facility_summaries parquet (input).",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "cache" / "facility_summaries.parquet"),
        help="Where to write the augmented parquet (default: overwrite input).",
    )
    args = parser.parse_args()

    summaries = pd.read_parquet(args.summaries)
    print(f"loaded {len(summaries)} rows from {args.summaries}")
    augmented = assign_districts(summaries)
    coverage = augmented["district"].notna().mean() * 100
    print(f"assigned districts to {coverage:.1f}% of facilities")
    augmented.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")
