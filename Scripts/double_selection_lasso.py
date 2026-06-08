#!/usr/bin/env python3
"""
double_selection_lasso.py

Double-selection LASSO (Belloni, Chernozhukov & Hansen 2014) to identify
candidate control variables for the quarterly ESG panel regression.

Procedure (per outcome Y):
  1. Two-way demean all variables (firm + quarter FE).
  2. S1 = support of Lasso(D_ESG_SCORE ~ candidate controls, alpha by CV)
  3. S2 = support of Lasso(Y           ~ candidate controls, alpha by CV)
  4. Candidates to consider = S1 ∪ S2

Requires (run the notebook Step 1 first):
  Outputs/quarterly_regression/quarterly_decomp.pkl
  Outputs/quarterly_regression/quarterly_realised_vol.pkl
  Data/data.xlsx

Output: table printed to stdout + CSV saved to Outputs/quarterly_regression/
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_DIR  = SCRIPTS_DIR.parent
DATA_FILE    = PROJECT_DIR / "Data" / "data.xlsx"
OUTPUTS_DIR  = PROJECT_DIR / "Outputs" / "quarterly_regression"
DECOMP_CACHE = OUTPUTS_DIR / "quarterly_decomp.pkl"
VOL_CACHE    = OUTPUTS_DIR / "quarterly_realised_vol.pkl"

# ── Configuration ──────────────────────────────────────────────────────────────
OUTCOMES    = ["FirmInfoShare", "MktInfoShare", "NoiseShare"]
TREATMENT   = "D_ESG_SCORE"
START_YEAR  = 2015
END_YEAR    = 2025
LASSO_FOLDS = 5
RANDOM_SEED = 42

# Base variables from which L1_ (lag) and D_ (diff) candidates are generated.
# ESG_SCORE generates only L1_ because D_ESG_SCORE is the treatment.
BLOOMBERG_BASE = [
    "PCT_INSIDER_SHARES_OUT",
    "ESG_SCORE",
    "BS_TOT_ASSET",
    "HEADLINE_BVPS",
    "PROF_MARGIN",
    "ASSET_TURNOVER",
    "FNCL_LVRG",
    "RETURN_COM_EQY",
    "CASH_FLOW_PER_SH",
    "HEADLINE_CAPEX",
]

WINSORISE_VARS = {"FNCL_LVRG", "PROF_MARGIN", "RETURN_COM_EQY"}


# ── Data helpers ───────────────────────────────────────────────────────────────
def read_quarterly_panel(data_file):
    """Parse the quarterly_ sheet into a long-format firm-quarter DataFrame."""
    ticker_meta = pd.read_excel(data_file, sheet_name="ticker", index_col=0)
    tickers = ticker_meta.index.tolist()
    raw = pd.read_excel(data_file, sheet_name="quarterly_", header=None)

    VARS_PER_STOCK = 20
    raw_var_names = raw.iloc[0, 1:VARS_PER_STOCK + 1].tolist()
    seen, col_names, keep_vars = set(), [], []
    for name in raw_var_names:
        if name not in seen:
            seen.add(name); col_names.append(name); keep_vars.append(name)
        else:
            col_names.append(f"__dup_{name}")

    date_col = pd.to_numeric(raw.iloc[1:, 0], errors="coerce")
    dates    = pd.to_datetime(date_col, origin="1899-12-30", unit="D", errors="coerce")
    valid    = dates.notna()

    chunks = []
    for s, ticker in enumerate(tickers):
        if ticker == "SXXP Index":
            continue
        c0  = 1 + s * VARS_PER_STOCK
        blk = raw.iloc[1:, c0:c0 + VARS_PER_STOCK].copy()[valid.values]
        blk.columns = col_names
        blk["stock"] = ticker
        blk["date"]  = dates[valid].values
        chunks.append(blk)

    df = pd.concat(chunks, ignore_index=True)
    df["date"]    = pd.to_datetime(df["date"])
    df["year"]    = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["period"]  = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    for col in keep_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (df[["stock", "date", "year", "quarter", "period"] + keep_vars]
            .sort_values(["stock", "date"])
            .reset_index(drop=True))


def winsorise(s, q=0.01):
    return s.clip(s.quantile(q), s.quantile(1 - q))


# ── LASSO helpers ──────────────────────────────────────────────────────────────
def two_way_demean(df, cols, entity="stock", time="period", tol=1e-10, max_iter=50):
    """Iterative two-way within-transformation (firm + time FE)."""
    X = df[cols].astype(float).copy()
    for _ in range(max_iter):
        prev = X.copy()
        X = X.sub(X.groupby(df[entity]).transform("mean"))
        X = X.sub(X.groupby(df[time]).transform("mean"))
        if (X - prev).abs().max().max() < tol:
            break
    return X


def lasso_support(X_df, y, cv=5, seed=42):
    """Return the set of column names with non-zero LassoCV coefficients."""
    Xv = X_df.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    ok = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)
    if ok.sum() < max(20, 2 * cv):
        return set()
    Xs  = StandardScaler().fit_transform(Xv[ok])
    mdl = LassoCV(cv=cv, max_iter=20_000, random_state=seed, n_jobs=-1)
    mdl.fit(Xs, yv[ok])
    return set(X_df.columns[mdl.coef_ != 0])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    for path in (DECOMP_CACHE, VOL_CACHE):
        if not path.exists():
            sys.exit(f"Cache not found: {path}\nRun the notebook (Step 1) first.")

    print("Loading decomposition cache…")
    decomp = pd.read_pickle(DECOMP_CACHE)[["stock", "period"] + OUTCOMES]
    decomp = decomp[decomp["period"].str[:4].astype(int).between(START_YEAR, END_YEAR)]

    rvol = pd.read_pickle(VOL_CACHE).rename(columns={"REALISED_VOL_Q_q": "REALISED_VOL_Q"})

    print("Reading quarterly_ sheet…")
    ctrl = read_quarterly_panel(DATA_FILE)
    ctrl = ctrl[ctrl["year"].between(START_YEAR, END_YEAR)].copy()

    # Derived and winsorised variables
    ctrl["LOG_MKT_CAP"] = np.log(ctrl["CUR_MKT_CAP"].clip(lower=1))
    for v in WINSORISE_VARS:
        if v in ctrl.columns:
            ctrl[v] = winsorise(ctrl[v])
    ctrl = ctrl.merge(rvol, on=["stock", "period"], how="left")

    # Generate L1_ and D_ transforms for all base variables
    all_base = ["LOG_MKT_CAP", "REALISED_VOL_Q"] + BLOOMBERG_BASE
    ctrl = ctrl.sort_values(["stock", "period"])
    for v in all_base:
        ctrl[f"L1_{v}"] = ctrl.groupby("stock")[v].shift(1)
        if v != "ESG_SCORE":          # D_ESG_SCORE is the treatment, not a candidate
            ctrl[f"D_{v}"] = ctrl.groupby("stock")[v].diff(1)
    ctrl[TREATMENT] = ctrl.groupby("stock")["ESG_SCORE"].diff(1)

    candidate_cols = (
        [f"L1_{v}" for v in all_base] +
        [f"D_{v}"  for v in all_base if v != "ESG_SCORE"]
    )

    # Merge decomp + controls
    panel = decomp.merge(
        ctrl[["stock", "period", TREATMENT] + candidate_cols],
        on=["stock", "period"], how="inner"
    )

    # Coverage in merged panel (before listwise deletion)
    coverage = {c: 100 * panel[c].notna().mean() for c in [TREATMENT] + candidate_cols}

    # Double-selection per outcome
    print()
    sel = {}
    for outcome in OUTCOMES:
        # Complete cases for treatment, outcome, and all candidates
        sub = (panel[["stock", "period", TREATMENT, outcome] + candidate_cols]
               .dropna())
        print(f"  {outcome}: {len(sub):,} obs, {sub['stock'].nunique()} stocks "
              f"({len(candidate_cols)} candidates, listwise deletion)")

        dm = two_way_demean(sub, [TREATMENT, outcome] + candidate_cols)
        s1 = lasso_support(dm[candidate_cols], dm[TREATMENT], cv=LASSO_FOLDS, seed=RANDOM_SEED)
        s2 = lasso_support(dm[candidate_cols], dm[outcome],   cv=LASSO_FOLDS, seed=RANDOM_SEED)
        sel[outcome] = {"s1": s1, "s2": s2}

    # ── Build output table ─────────────────────────────────────────────────────
    rows = []
    for col in candidate_cols:
        cov    = coverage.get(col, np.nan)
        row    = {"Variable": col, "Coverage (%)": f"{cov:.1f}"}
        in_any = False
        for outcome in OUTCOMES:
            s1, s2 = sel[outcome]["s1"], sel[outcome]["s2"]
            if   col in s1 and col in s2: tag = "S1+S2"; in_any = True
            elif col in s1:               tag = "S1   "; in_any = True
            elif col in s2:               tag = "   S2"; in_any = True
            else:                         tag = ""
            row[outcome] = tag
        rows.append((in_any, cov, row))

    # Sort: selected first, then by coverage descending
    rows.sort(key=lambda x: (not x[0], -x[1]))
    table = pd.DataFrame([r for _, _, r in rows])

    # Print
    sep = "=" * 82
    print(f"\n{sep}")
    print("DOUBLE-SELECTION LASSO — CANDIDATE CONTROL VARIABLES")
    print("S1 = survives Lasso(D_ESG_SCORE ~ controls)   (confound of treatment)")
    print("S2 = survives Lasso(outcome      ~ controls)   (predictor of outcome)")
    print(sep)
    print(table.to_string(index=False))
    print(sep)
    print("\nNote: listwise deletion on all 29 candidates reduces N vs. the baseline "
          "regression (~21k). Variables selected here should be validated for "
          "sample-size impact before adding to the final spec.")

    # Save
    out_path = OUTPUTS_DIR / "double_selection_candidates.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
