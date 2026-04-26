
"""Fetch reference data needed for the district choropleth + population join.

Downloads:
* `india_districts.geojson` - DataMeet's Census 2011 India districts polygons
* `district_population.csv` - District-level Census 2011 population

Both files land under `data/reference/` and are git-ignored. Re-runs are
idempotent (skips if file exists and `--force` is not passed).

Usage:
    python scripts/fetch_reference_data.py
    python scripts/fetch_reference_data.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data" / "reference"
REF_DIR.mkdir(parents=True, exist_ok=True)


# Public Census 2011 / 2019 India district polygons.
# HindustanTimesLabs hosts a clean ~14 MB file built from 2011 boundaries.
# geohacker/india is a heavier (~34 MB) but well-maintained fallback.
DISTRICT_GEOJSON_PRIMARY = (
    "https://raw.githubusercontent.com/HindustanTimesLabs/shapefiles/master/india/district/india_2011_district.json"
)
DISTRICT_GEOJSON_FALLBACK = (
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
)

# State-level fallback if the district file is unreachable.
STATE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
)

# Public Census 2011 district population (Indian Govt open data, mirrored on GitHub).
DISTRICT_POPULATION_URL = (
    "https://raw.githubusercontent.com/nishusharma1608/India-Census-2011-Analysis/master/india-districts-census-2011.csv"
)


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "vf-health-hackathon/1.0"})
    print(f"GET {url}")
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"  -> {dest}  ({dest.stat().st_size:,} bytes)")


def _download_with_fallback(urls: list[str], dest: Path) -> bool:
    last_err: Exception | None = None
    for url in urls:
        try:
            _download(url, dest)
            return True
        except Exception as e:
            last_err = e
            print(f"  FAILED ({url}): {e}", file=sys.stderr)
    if last_err:
        print(f"  giving up on {dest.name}", file=sys.stderr)
    return False


def fetch(force: bool = False) -> None:
    district_dest = REF_DIR / "india_districts.geojson"
    state_dest = REF_DIR / "india_states.geojson"
    pop_dest = REF_DIR / "district_population.csv"

    if district_dest.exists() and not force:
        print(f"skip {district_dest.name} (exists; pass --force to redownload)")
    else:
        _download_with_fallback(
            [DISTRICT_GEOJSON_PRIMARY, DISTRICT_GEOJSON_FALLBACK], district_dest
        )

    if state_dest.exists() and not force:
        print(f"skip {state_dest.name}")
    else:
        try:
            _download(STATE_GEOJSON_URL, state_dest)
        except Exception as e:
            print(f"  state file failed (non-fatal): {e}")

    if pop_dest.exists() and not force:
        print(f"skip {pop_dest.name}")
    else:
        try:
            _download(DISTRICT_POPULATION_URL, pop_dest)
        except Exception as e:
            print(f"  population file failed (non-fatal): {e}")

    if not district_dest.exists():
        if state_dest.exists():
            print(
                "Districts file unavailable; the pipeline will fall back to state-level polygons."
            )
        else:
            raise SystemExit("No GeoJSON could be downloaded - check network access.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Redownload even if file exists.")
    args = parser.parse_args()
    fetch(force=args.force)
