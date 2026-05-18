"""
Quarterly VAR lag-order selection via AIC.

Companion to Thesis_2_quarterly.py: for each stock-quarter, fit a VAR with
maxlags=10 and select the optimal lag order using AIC (the criterion used in
Brogaard et al. 2022). Reports the distribution of selected lags across
stock-quarters and by quarter, so MAX_LAGS in Thesis_2_quarterly.py can be set
on an empirical basis instead of hard-coded.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from data_interface import read_bloomberg_two_row_sheet, split_price_volume

warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
OUTPUTS_DIR = PROJECT_DIR / "Outputs" / "thesis_2_quarterly_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

WINSOR_BOUNDS = (0.05, 0.95)
RETURN_SCALE = 10000
MIN_VALID_OBS = 20
LAG_MIN = 1
LAG_MAX = 10
CRITERIA = ("aic", "bic", "hqic", "fpe")


def load_data():
    """Load stock data from data.xlsx daily_ sheet."""

    file_path = DATA_DIR / "data.xlsx"
    index_name = "SXXP Index"

    raw = read_bloomberg_two_row_sheet(file_path, sheet_name="daily_")
    parts = split_price_volume(raw)

    prices = parts["prices"].ffill(limit=5)
    market_ret = prices[index_name].pct_change(fill_method=None)

    return {"prices": prices, "market_ret": market_ret, "index": index_name}


def select_lag_single_period(market_ret, stock_ret, stock_name, period_label):
    """
    For one stock-quarter, fit VAR up to LAG_MAX lags and return selected
    orders under each information criterion.
    """

    var_data = pd.DataFrame(
        {
            "market_ret": market_ret * RETURN_SCALE,
            "stock_ret": stock_ret * RETURN_SCALE,
        }
    ).dropna()

    for col in ["market_ret", "stock_ret"]:
        q_low = var_data[col].quantile(WINSOR_BOUNDS[0])
        q_high = var_data[col].quantile(WINSOR_BOUNDS[1])
        var_data[col] = var_data[col].clip(lower=q_low, upper=q_high)

    n_obs = len(var_data)
    if n_obs < MIN_VALID_OBS:
        return None

    # Cap the search range to what the sample size will support.
    # VAR with 2 vars and a constant uses (2*p + 1) parameters per equation,
    # so require n_obs - p > 2*p + 1 → p < (n_obs - 1) / 3.
    max_supported = max(1, min(LAG_MAX, (n_obs - 1) // 3))

    try:
        model = VAR(var_data)
        order_result = model.select_order(maxlags=max_supported, trend="c")
    except Exception:
        return None

    selected = order_result.selected_orders  # dict: {'aic': k, 'bic': k, ...}

    record = {
        "stock": stock_name,
        "period": period_label,
        "n_obs": n_obs,
        "maxlags_searched": max_supported,
    }
    for crit in CRITERIA:
        record[f"lag_{crit}"] = int(selected.get(crit, np.nan)) if selected.get(crit) is not None else np.nan

    return record


def run_lag_selection(prices, market_ret):
    """Loop over (stock, quarter) cells and collect selected lag orders."""

    stock_ret = prices.pct_change()
    common_idx = stock_ret.index.intersection(market_ret.index)
    stock_ret = stock_ret.loc[common_idx]
    market_ret = market_ret.loc[common_idx]

    quarters = sorted(
        {(d.year, d.quarter) for d in stock_ret.index},
        key=lambda yq: (yq[0], yq[1]),
    )

    records = []
    for year, quarter in quarters:
        period_label = f"{year}-Q{quarter}"
        mask = (stock_ret.index.year == year) & (stock_ret.index.quarter == quarter)
        q_stock_ret = stock_ret[mask]
        q_market_ret = market_ret[mask]

        n_processed = 0
        for ticker in q_stock_ret.columns:
            rec = select_lag_single_period(
                market_ret=q_market_ret,
                stock_ret=q_stock_ret[ticker],
                stock_name=ticker,
                period_label=period_label,
            )
            if rec is not None:
                records.append(rec)
                n_processed += 1

        print(f"   {period_label}: {n_processed} stocks")

    return pd.DataFrame(records)


def summarize(df):
    """Print and return distribution summaries of selected lags."""

    print("\n" + "=" * 80)
    print("OVERALL DISTRIBUTION OF SELECTED LAGS (across all stock-quarters)")
    print("=" * 80)

    summary_rows = []
    for crit in CRITERIA:
        col = f"lag_{crit}"
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        summary_rows.append(
            {
                "criterion": crit.upper(),
                "mean": vals.mean(),
                "median": vals.median(),
                "mode": int(vals.mode().iloc[0]),
                "p25": vals.quantile(0.25),
                "p75": vals.quantile(0.75),
                "min": int(vals.min()),
                "max": int(vals.max()),
                "n": int(len(vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("FREQUENCY OF EACH LAG (count, % of valid stock-quarters)")
    print("=" * 80)

    freq_rows = []
    for crit in CRITERIA:
        col = f"lag_{crit}"
        vals = df[col].dropna().astype(int)
        if len(vals) == 0:
            continue
        counts = vals.value_counts().sort_index()
        for lag, n in counts.items():
            freq_rows.append(
                {
                    "criterion": crit.upper(),
                    "lag": int(lag),
                    "count": int(n),
                    "pct": 100 * n / len(vals),
                }
            )
    freq_df = pd.DataFrame(freq_rows)
    if not freq_df.empty:
        pivot = freq_df.pivot(index="lag", columns="criterion", values="pct").fillna(0).round(2)
        print("\nPercent of stock-quarters selecting each lag:")
        print(pivot)

    print("\n" + "=" * 80)
    print("MEAN SELECTED LAG BY QUARTER (AIC)")
    print("=" * 80)
    by_quarter = df.groupby("period")[[f"lag_{c}" for c in CRITERIA]].mean().round(2)
    print(by_quarter.head(20))

    return summary_df, freq_df, by_quarter


def main():
    print("=" * 80)
    print("QUARTERLY VAR LAG-ORDER SELECTION (AIC / BIC / HQIC / FPE)")
    print("=" * 80)
    print(f"Lag search range: [{LAG_MIN}, {LAG_MAX}]")
    print(f"Min observations per stock-quarter: {MIN_VALID_OBS}")
    print(f"Winsorization: returns at {int(WINSOR_BOUNDS[0]*100)}%-{int(WINSOR_BOUNDS[1]*100)}%")

    print("\n1. Loading data...")
    data = load_data()
    prices = data["prices"]
    market_ret = data["market_ret"]
    print(f"   - {len(prices)} dates, {len(prices.columns)} stocks")
    print(f"   - {prices.index[0].date()} → {prices.index[-1].date()}")

    print("\n2. Selecting lag order for each stock-quarter...")
    df = run_lag_selection(prices, market_ret)
    print(f"\n   Total stock-quarter cells with a valid selection: {len(df)}")

    if df.empty:
        print("No results produced — aborting.")
        return

    summary_df, freq_df, by_quarter = summarize(df)

    print("\n3. Saving outputs...")
    df.to_csv(OUTPUTS_DIR / "lag_selection_results.csv", index=False)
    summary_df.to_csv(OUTPUTS_DIR / "lag_selection_summary.csv", index=False)
    freq_df.to_csv(OUTPUTS_DIR / "lag_selection_frequency.csv", index=False)
    by_quarter.to_csv(OUTPUTS_DIR / "lag_selection_by_quarter.csv")
    print(f"   - Saved to {OUTPUTS_DIR}")

    aic_mean = df["lag_aic"].mean()
    aic_median = df["lag_aic"].median()
    aic_mode = int(df["lag_aic"].dropna().mode().iloc[0])
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"AIC mean lag:   {aic_mean:.2f}")
    print(f"AIC median lag: {aic_median:.1f}")
    print(f"AIC modal lag:  {aic_mode}")
    print(
        "\nSuggestion: set MAX_LAGS in Thesis_2_quarterly.py to the AIC median or mode,\n"
        "or pass ic='aic' to VAR.fit(maxlags=10, ic='aic', trend='c') to let statsmodels\n"
        "pick per-cell."
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
