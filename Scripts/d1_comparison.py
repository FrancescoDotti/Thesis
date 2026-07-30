"""
(D1) justification: compare the aggregated firm-specific variance share from
the 2-variable VAR (Thesis_2) against the sum of the private and public
firm-specific shares from the 3-variable VAR (Thesis_3), year by year.

Outputs a LaTeX-ready CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from data_interface import (read_bloomberg_two_row_sheet, split_price_volume,
                            build_daily_canonical_panel)
from Thesis_2 import run_thesis2_from_daily_panel
from Thesis_3 import run_thesis3_from_daily_panel

PROJECT_DIR = SCRIPTS_DIR.parent
DATA_FILE = PROJECT_DIR / "Data" / "data.xlsx"
OUT_DIR = PROJECT_DIR / "Outputs" / "quarterly_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MARKET_TICKER = "SXXP Index"

print("Loading daily panel ...")
raw = read_bloomberg_two_row_sheet(DATA_FILE, sheet_name="daily_")
parts = split_price_volume(raw)
daily = build_daily_canonical_panel(parts["prices"], parts["volume"],
                                    market_ticker=MARKET_TICKER)
print(f"  {daily['stock'].nunique()} stocks, {len(daily):,} rows")

print("Running Thesis_2 (2-variable VAR, aggregated firm-specific) ...")
res2 = run_thesis2_from_daily_panel(daily)["results"].copy()
print("Running Thesis_3 (3-variable VAR, private + public) ...")
res3 = run_thesis3_from_daily_panel(daily)["results"].copy()

# Standardise keys (Thesis_2 carries 'period'=year-string; Thesis_3 carries 'year')
res2["year"] = res2["period"].astype(str).str[:4].astype(int)
if "stock" not in res3.columns and "ticker" in res3.columns:
    res3 = res3.rename(columns={"ticker": "stock"})
res3["year"] = res3["year"].astype(int)

res3["FirmSumShare"] = res3["PrivateInfoShare"] + res3["PublicInfoShare"]

a = res2[["stock", "year", "FirmInfoShare"]].dropna()
b = res3[["stock", "year", "FirmSumShare"]].dropna()
merged = a.merge(b, on=["stock", "year"], how="inner")
print(f"  matched stock-year cells: {len(merged):,}")

merged["abs_diff"] = (merged["FirmInfoShare"] - merged["FirmSumShare"]).abs()

rows = []
for yr, g in merged.groupby("year"):
    rows.append({
        "year": int(yr),
        "n": len(g),
        "agg_firm": g["FirmInfoShare"].mean(),        # Thesis_2 aggregated
        "priv_pub": g["FirmSumShare"].mean(),          # Thesis_3 private+public
        "corr": g["FirmInfoShare"].corr(g["FirmSumShare"]),
        "mad": g["abs_diff"].mean(),                    # mean abs difference (pp)
    })
tab = pd.DataFrame(rows).sort_values("year")

# Pooled row
pooled = {
    "year": "All",
    "n": len(merged),
    "agg_firm": merged["FirmInfoShare"].mean(),
    "priv_pub": merged["FirmSumShare"].mean(),
    "corr": merged["FirmInfoShare"].corr(merged["FirmSumShare"]),
    "mad": merged["abs_diff"].mean(),
}
tab = pd.concat([tab, pd.DataFrame([pooled])], ignore_index=True)

pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(tab.to_string(index=False))

out_csv = OUT_DIR / "d1_firm_specific_comparison.csv"
tab.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
