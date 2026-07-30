"""Quarterly robustness of the (D1) near-identity:
aggregated firm-specific (2-var VAR, quarterly) vs private+public (3-var VAR, quarterly)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from data_interface import (read_bloomberg_two_row_sheet, split_price_volume,
                            build_daily_canonical_panel)

# Force the 3-variable VAR to one lag, matching the quarterly 2-variable VAR
# (deviation D3), so the comparison is apples-to-apples at quarterly frequency.
from statsmodels.tsa.api import VAR as _VAR
_orig_fit = _VAR.fit
def _fit_p1(self, *a, **k):
    k["maxlags"] = 1
    k.pop("ic", None)
    return _orig_fit(self, *a, **k)
_VAR.fit = _fit_p1

from Thesis_3 import process_stock_year_data

PROJECT_DIR = SCRIPTS_DIR.parent
DATA_FILE = PROJECT_DIR / "Data" / "data.xlsx"
CACHE = PROJECT_DIR / "Outputs" / "quarterly_regression" / "quarterly_decomp.pkl"

print("Loading daily panel ...")
raw = read_bloomberg_two_row_sheet(DATA_FILE, sheet_name="daily_")
parts = split_price_volume(raw)
daily = build_daily_canonical_panel(parts["prices"], parts["volume"],
                                    market_ticker="SXXP Index")
daily["date"] = pd.to_datetime(daily["date"])
daily["period"] = (daily["date"].dt.year.astype(str) + "-Q"
                   + daily["date"].dt.quarter.astype(str))

df = daily.rename(columns={"stock": "ticker", "stock_ret": "r", "market_ret": "rm"})
df = df.dropna(subset=["ticker", "period", "date", "rm", "r", "volume", "price"])

print("Running Thesis_3 at QUARTERLY frequency (private + public) ...")
res3 = process_stock_year_data(df, stock_col="ticker", year_col="period",
                               market_ret_col="rm", stock_ret_col="r",
                               volume_col="volume", price_col="price")
res3 = res3.rename(columns={"year": "period"})
res3["FirmSumShare"] = res3["PrivateInfoShare"] + res3["PublicInfoShare"]

# Cached Thesis_2 quarterly aggregated firm-specific share
res2 = pd.read_pickle(CACHE)[["stock", "period", "FirmInfoShare"]].dropna()

m = res2.merge(res3[["stock", "period", "FirmSumShare"]], on=["stock", "period"], how="inner")
m["abs_diff"] = (m["FirmInfoShare"] - m["FirmSumShare"]).abs()

print(f"\nMatched stock-quarter cells : {len(m):,}")
print(f"Mean aggregated firm share  : {m['FirmInfoShare'].mean():.3f}")
print(f"Mean private+public share   : {m['FirmSumShare'].mean():.3f}")
print(f"Correlation                 : {m['FirmInfoShare'].corr(m['FirmSumShare']):.3f}")
print(f"Mean absolute difference pp : {m['abs_diff'].mean():.3f}")
print(f"Median absolute difference  : {m['abs_diff'].median():.3f}")
