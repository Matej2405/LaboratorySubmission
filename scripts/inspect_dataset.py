"""Quick local profiler for the Virtue Foundation India dataset.

Run: python scripts/inspect_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "VF_Hackathon_Dataset_India_Large.xlsx"
)


def main() -> None:
    print(f"Reading {DATA_PATH}")
    xls = pd.ExcelFile(DATA_PATH, engine="openpyxl")
    print(f"Sheets: {xls.sheet_names}\n")

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        print(f"=== Sheet: {sheet}  shape={df.shape} ===")
        print("Columns:")
        for c in df.columns:
            dtype = str(df[c].dtype)
            non_null = df[c].notna().sum()
            sample = df[c].dropna().astype(str).head(1).tolist()
            sample_str = (sample[0][:120] + "...") if sample and len(sample[0]) > 120 else (sample[0] if sample else "")
            print(f"  - {c!r:40s}  dtype={dtype:10s}  non_null={non_null:>6}  sample={sample_str!r}")

        str_cols = df.select_dtypes(include="object").columns
        if len(str_cols):
            print("\nString-column length stats:")
            stats = (
                df[str_cols]
                .astype("string")
                .map(lambda x: len(x) if isinstance(x, str) else 0)
                .agg(["min", "median", "mean", "max"])
            )
            print(stats.T.round(1).to_string())

        print("\nFirst 2 rows:")
        for i, row in df.head(2).iterrows():
            print(f"  --- row {i} ---")
            for c in df.columns:
                v = row[c]
                if isinstance(v, str) and len(v) > 200:
                    v = v[:200] + "..."
                print(f"    {c}: {v!r}")
        print("\n")

    summary = {
        "sheets": xls.sheet_names,
        "primary_sheet": xls.sheet_names[0],
        "columns": list(xls.parse(xls.sheet_names[0]).columns),
    }
    out = Path(__file__).parent / "dataset_profile.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
