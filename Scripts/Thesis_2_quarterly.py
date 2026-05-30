"""
Quarterly Variance Decomposition Script
Based on the methodology in "What Moves Stock Prices"

Identical to Thesis_2.py but groups daily data by calendar quarter instead of
calendar year, producing stock-quarter variance decomposition shares.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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
# maxlags=5 (used for annual VAR) causes explosive VAR estimates. 
# maxlags=2 keeps the observations-to-parameters ratio comparable to the annual specification.
# AIC advises <1 lag, so max_lags=1 is a conservative choice.
MAX_LAGS = 1


def run_thesis2_quarterly_from_daily_panel(daily_df, winsor_bounds=WINSOR_BOUNDS):
    """
    Run quarterly Thesis_2 decomposition from a canonical daily panel.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Must include: stock, date, stock_ret, market_ret

    Returns
    -------
    dict with keys: results, ew, vw, diagnostics
    """

    required_cols = {"stock", "date", "stock_ret", "market_ret"}
    missing_cols = required_cols - set(daily_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["period"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)
    df = df.dropna(subset=["stock", "date", "stock_ret", "market_ret"])

    all_results = []

    for period, period_group in df.groupby("period"):
        market_ret = period_group.groupby("date")["market_ret"].first().sort_index()

        for stock, stock_group in period_group.groupby("stock"):
            stock_ret = stock_group.set_index("date")["stock_ret"].sort_index()

            result = decompose_variance_single_period(
                market_ret=market_ret,
                stock_ret=stock_ret,
                stock_name=stock,
            )

            if result is not None:
                result["period"] = period
                all_results.append(result)

    if len(all_results) == 0:
        return {
            "results": pd.DataFrame(),
            "ew": pd.DataFrame(),
            "vw": pd.DataFrame(),
            "diagnostics": pd.DataFrame(),
        }

    results_df = pd.DataFrame(all_results)

    component_cols = ["MktInfo", "FirmInfo", "Noise"]
    results_df = winsorize_by_period(results_df, component_cols, bounds=winsor_bounds)

    share_cols = ["MktInfoShare", "FirmInfoShare", "NoiseShare"]
    ew_df, vw_df = calculate_aggregate_shares(results_df, share_cols)
    diagnostics_df = build_period_diagnostics(results_df, share_cols)

    return {
        "results": results_df,
        "ew": ew_df,
        "vw": vw_df,
        "diagnostics": diagnostics_df,
    }


def load_data():
    """Load stock data from data.xlsx daily_ sheet."""

    file_path = DATA_DIR / "data.xlsx"
    index_name = "SXXP Index"

    raw = read_bloomberg_two_row_sheet(file_path, sheet_name="daily_")
    parts = split_price_volume(raw)

    prices = parts["prices"].ffill(limit=5)
    volume = parts["volume"].ffill(limit=5)
    market_ret = prices[index_name].pct_change(fill_method=None)

    return {
        "prices": prices,
        "volume": volume,
        "market_ret": market_ret,
        "index": index_name,
    }


def decompose_variance_single_period(market_ret, stock_ret, stock_name):
    """
    Decompose variance of a single stock in a single period using VAR.

    Follows Brogaard et al. (2022) methodology. Identical to Thesis_2.py —
    the period length (quarter vs year) is controlled by the caller.

    Returns dict with variance decomposition results, or None if estimation fails.
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

    if len(var_data) < MIN_VALID_OBS:
        return None

    try:
        var_result = VAR(var_data).fit(maxlags=MAX_LAGS, trend="c")
    except Exception:
        return None

    resids = var_result.resid
    e_market = resids.iloc[:, 0].values
    e_stock = resids.iloc[:, 1].values

    sigma2_e_market = np.var(e_market, ddof=1)
    sigma2_e_stock = np.var(e_stock, ddof=1)

    if sigma2_e_market <= 0:
        return None

    b10 = np.cov(e_stock, e_market)[0, 1] / sigma2_e_market

    sigma2_eps_market = sigma2_e_market
    sigma2_eps_stock = sigma2_e_stock - b10**2 * sigma2_e_market

    if sigma2_eps_stock < 0:
        return None

    params = var_result.params.values
    n_vars = 2
    used_lags = var_result.k_ar
    A_sum = np.zeros((n_vars, n_vars))
    for lag in range(used_lags):
        start_row = 1 + lag * n_vars
        end_row = 1 + (lag + 1) * n_vars
        A_lag = params[start_row:end_row, :].T
        A_sum += A_lag

    max_eigenvalue = np.max(np.abs(np.linalg.eigvals(A_sum)))
    if max_eigenvalue >= 1.0:
        return None

    try:
        LR_matrix = np.linalg.inv(np.eye(n_vars) - A_sum)
    except np.linalg.LinAlgError:
        return None

    B0_inv = np.array([[1.0, 0.0], [b10, 1.0]])
    LR_structural = LR_matrix @ B0_inv

    theta_market = LR_structural[1, 0]
    theta_stock = LR_structural[1, 1]

    MktInfo = theta_market**2 * sigma2_eps_market
    FirmInfo = theta_stock**2 * sigma2_eps_stock

    actual_returns = var_data["stock_ret"].iloc[used_lags:].values
    Noise = max(
        np.var(actual_returns, ddof=1)
        - (theta_market**2 * sigma2_eps_market + theta_stock**2 * sigma2_eps_stock),
        0,
    )

    TotalVar = MktInfo + FirmInfo + Noise
    if TotalVar <= 0:
        return None

    return {
        "stock": stock_name,
        "n_obs": len(var_data),
        "k_ar": used_lags,
        "max_eigenvalue": max_eigenvalue,
        "b10": b10,
        "theta_market": theta_market,
        "theta_stock": theta_stock,
        "sigma2_eps_market": sigma2_eps_market,
        "sigma2_eps_stock": sigma2_eps_stock,
        "MktInfo": MktInfo,
        "FirmInfo": FirmInfo,
        "Noise": Noise,
        "TotalVar": TotalVar,
        "MktInfoShare": 100 * MktInfo / TotalVar,
        "FirmInfoShare": 100 * FirmInfo / TotalVar,
        "NoiseShare": 100 * Noise / TotalVar,
    }


def decompose_all_stocks_period(market_ret, stock_prices, period_label):
    """Decompose variance for all stocks in a given period."""

    stock_ret = stock_prices.pct_change()
    common_idx = stock_ret.index.intersection(market_ret.index)
    stock_ret = stock_ret.loc[common_idx]
    market_ret = market_ret.loc[common_idx]

    results = []
    for ticker in stock_prices.columns:
        if ticker in stock_ret.columns and market_ret.notna().sum() > MIN_VALID_OBS:
            result = decompose_variance_single_period(market_ret, stock_ret[ticker], ticker)
            if result is not None:
                result["period"] = period_label
                results.append(result)

    return pd.DataFrame(results) if results else None


def winsorize_by_period(df, component_cols, bounds=(0.05, 0.95)):
    """Winsorize variance components by period, then recalculate shares."""

    df_w = df.copy()

    for period in df_w["period"].unique():
        mask = df_w["period"] == period
        for col in component_cols:
            if col in df_w.columns:
                q_low = df_w.loc[mask, col].quantile(bounds[0])
                q_high = df_w.loc[mask, col].quantile(bounds[1])
                df_w.loc[mask, col] = df_w.loc[mask, col].clip(q_low, q_high)

    df_w["TotalVar"] = df_w["MktInfo"] + df_w["FirmInfo"] + df_w["Noise"]
    df_w["MktInfoShare"] = 100 * df_w["MktInfo"] / df_w["TotalVar"]
    df_w["FirmInfoShare"] = 100 * df_w["FirmInfo"] / df_w["TotalVar"]
    df_w["NoiseShare"] = 100 * df_w["Noise"] / df_w["TotalVar"]

    return df_w


def calculate_aggregate_shares(results_df, share_cols=None):
    """Equal-weighted and variance-weighted average shares by period."""

    if share_cols is None:
        share_cols = ["MktInfoShare", "FirmInfoShare", "NoiseShare"]

    EW = results_df.groupby("period")[share_cols].mean()

    def vw_aggregate(group):
        clean = group.dropna(subset=share_cols + ["TotalVar"])
        clean = clean[clean["TotalVar"] > 0]
        if len(clean) == 0:
            return pd.Series({col: np.nan for col in share_cols})
        weights = clean["TotalVar"].values
        return pd.Series({col: np.average(clean[col].values, weights=weights) for col in share_cols})

    VW = results_df.groupby("period").apply(vw_aggregate)

    return EW, VW


def build_period_diagnostics(results_df, share_cols, period_col="period", total_col="TotalVar"):
    """Build per-quarter diagnostics table."""

    diagnostics = []
    for period_value, group in results_df.groupby(period_col):
        valid_share_mask = group[share_cols].notna().all(axis=1)
        positive_total_mask = group[total_col] > 0
        valid_mask = valid_share_mask & positive_total_mask
        share_sum = group.loc[valid_mask, share_cols].sum(axis=1)

        diagnostics.append(
            {
                "period": str(period_value),
                "n_stock_quarters": len(group),
                "n_unique_stocks": group["stock"].nunique() if "stock" in group.columns else np.nan,
                "n_valid_shares": int(valid_mask.sum()),
                "pct_valid_shares": 100 * valid_mask.mean() if len(group) > 0 else np.nan,
                "total_var_mean": group[total_col].mean(),
                "total_var_median": group[total_col].median(),
                "share_sum_mean": share_sum.mean() if len(share_sum) > 0 else np.nan,
                "share_sum_std": share_sum.std(ddof=1) if len(share_sum) > 1 else np.nan,
                "max_eigenvalue_mean": group["max_eigenvalue"].mean() if "max_eigenvalue" in group.columns else np.nan,
                "max_eigenvalue_max": group["max_eigenvalue"].max() if "max_eigenvalue" in group.columns else np.nan,
                "pct_stationary_lt_0_99": (
                    100 * (group["max_eigenvalue"] < 0.99).mean()
                    if "max_eigenvalue" in group.columns and len(group) > 0
                    else np.nan
                ),
                "k_ar_mean": group["k_ar"].mean() if "k_ar" in group.columns else np.nan,
                "n_obs_mean": group["n_obs"].mean() if "n_obs" in group.columns else np.nan,
            }
        )

    return pd.DataFrame(diagnostics).sort_values("period")


def main():
    """Main execution function."""

    print("=" * 80)
    print("QUARTERLY VARIANCE DECOMPOSITION: What Moves Stock Prices")
    print("=" * 80)
    print("\nMethodology: Brogaard, Nguyen, Putnins & Wu (2022)")
    print(f"Winsorization: Components at {int(WINSOR_BOUNDS[0]*100)}%-{int(WINSOR_BOUNDS[1]*100)}% (before calculating shares)")
    print("Frequency: Quarterly")
    print("=" * 80)

    print("\n1. Loading data...")
    data = load_data()
    prices = data["prices"]
    market_ret = data["market_ret"]

    print(f"   - Loaded {len(prices)} dates")
    print(f"   - {len(prices.columns)} stocks")
    print(f"   - Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    print("\n2. Computing variance decomposition by quarter...")

    all_results = []
    # Build (year, quarter) pairs in chronological order.
    quarters = sorted(
        {(d.year, d.quarter) for d in prices.index},
        key=lambda yq: (yq[0], yq[1]),
    )

    for year, quarter in quarters:
        period_label = f"{year}-Q{quarter}"
        print(f"   Processing {period_label}...", end=" ")

        mask = (prices.index.year == year) & (prices.index.quarter == quarter)
        quarter_prices = prices[mask]
        quarter_market = market_ret[mask]

        quarter_results = decompose_all_stocks_period(
            quarter_market,
            quarter_prices,
            period_label,
        )

        if quarter_results is not None:
            all_results.append(quarter_results)
            print(f"✓ {len(quarter_results)} stocks")
        else:
            print("✗ No valid results")

    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\n   Total stock-quarter observations: {len(results_df)}")

    print(f"\n3. Winsorizing variance components at {int(WINSOR_BOUNDS[0]*100)}%-{int(WINSOR_BOUNDS[1]*100)}% bounds...")

    component_cols = ["MktInfo", "FirmInfo", "Noise"]
    results_df = winsorize_by_period(results_df, component_cols, bounds=WINSOR_BOUNDS)

    print("\n4. Aggregating results...")

    share_cols = ["MktInfoShare", "FirmInfoShare", "NoiseShare"]
    EW, VW = calculate_aggregate_shares(results_df, share_cols)
    diagnostics_df = build_period_diagnostics(results_df, share_cols)

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: Quarterly Variance Share Decomposition")
    print("=" * 80)

    print("\nVariance-Weighted Average (%):")
    print("-" * 80)
    print(VW)

    print("\n\nEqual-Weighted Average (%):")
    print("-" * 80)
    print(EW)

    vw_mean = VW[share_cols].mean()
    ew_mean = EW[share_cols].mean()

    print("\n" + "=" * 80)
    print("OVERALL SAMPLE STATISTICS")
    print("=" * 80)
    print("\nVariance-Weighted Mean Across Full Period:")
    print(f"  Market Information:    {vw_mean['MktInfoShare']:6.2f}%")
    print(f"  Firm-Specific Info:    {vw_mean['FirmInfoShare']:6.2f}%")
    print(f"  Noise:                 {vw_mean['NoiseShare']:6.2f}%")
    print("\nEqual-Weighted Mean Across Full Period:")
    print(f"  Market Information:    {ew_mean['MktInfoShare']:6.2f}%")
    print(f"  Firm-Specific Info:    {ew_mean['FirmInfoShare']:6.2f}%")
    print(f"  Noise:                 {ew_mean['NoiseShare']:6.2f}%")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CHECKS")
    print("=" * 80)

    print(f"\nSample size by quarter (first 12):")
    print(results_df.groupby("period").size().head(12))
    print(f"\nTotal stock-quarter observations: {len(results_df)}")
    print(f"Obs with valid variance shares: {results_df[share_cols].notna().all(axis=1).sum()}")
    print(f"Mean daily obs per stock-quarter: {results_df['n_obs'].mean():.1f}")

    print(f"\nVAR stationarity (max eigenvalue):")
    print(f"  Mean: {results_df['max_eigenvalue'].mean():8.4f}")
    print(f"  Max:  {results_df['max_eigenvalue'].max():8.4f}")
    print(f"  % with max_eig < 0.99: {(results_df['max_eigenvalue'] < 0.99).sum() / len(results_df) * 100:.1f}%")

    print("\nQuarterly diagnostics (first 12 rows):")
    print(diagnostics_df.head(12))

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n5. Creating visualizations...")

    # For plots with many quarters, only label Q1 of each year on the x-axis.
    period_labels = VW.index.tolist()
    tick_positions = [i for i, p in enumerate(period_labels) if p.endswith("Q1")]
    tick_labels = [p[:4] for p in period_labels if p.endswith("Q1")]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: VW time series
    ax = axes[0, 0]
    x = np.arange(len(VW))
    for col, color in zip(share_cols, colors):
        ax.plot(x, VW[col], label=col.replace("Share", ""), linewidth=1.5, color=color)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("Variance-Weighted Shares Over Time (Quarterly)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Plot 2: EW time series
    ax = axes[0, 1]
    for col, color in zip(share_cols, colors):
        ax.plot(x, EW[col], label=col.replace("Share", ""), linewidth=1.5, color=color)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("Equal-Weighted Shares Over Time (Quarterly)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Plot 3: Stacked bar (VW)
    ax = axes[1, 0]
    bottom = np.zeros(len(VW))
    for col, color in zip(share_cols, colors):
        ax.bar(x, VW[col], 0.8, bottom=bottom, label=col.replace("Share", ""), color=color)
        bottom += VW[col].values
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("Variance-Weighted Shares (Stacked, Quarterly)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 100])

    # Plot 4: Cross-sectional distribution in the latest quarter
    ax = axes[1, 1]
    latest_period = results_df["period"].max()
    latest_data = results_df[results_df["period"] == latest_period][share_cols]
    bp = ax.boxplot(
        [latest_data[col].dropna() for col in share_cols],
        labels=["Market Info", "Firm Info", "Noise"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title(f"Cross-Sectional Distribution ({latest_period})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "variance_decomposition_quarterly.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   - Saved: variance_decomposition_quarterly.png")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n6. Saving detailed results...")

    results_df.to_csv(OUTPUTS_DIR / "variance_decomposition_quarterly_results.csv", index=False)
    VW.to_csv(OUTPUTS_DIR / "variance_decomposition_quarterly_VW.csv")
    EW.to_csv(OUTPUTS_DIR / "variance_decomposition_quarterly_EW.csv")
    diagnostics_df.to_csv(OUTPUTS_DIR / "variance_decomposition_quarterly_diagnostics.csv", index=False)
    print("   - Saved: _results.csv, _VW.csv, _EW.csv, _diagnostics.csv")

    with open(OUTPUTS_DIR / "variance_decomposition_quarterly_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("QUARTERLY VARIANCE DECOMPOSITION: What Moves Stock Prices\n")
        f.write("Brogaard, Nguyen, Putnins & Wu (2022) Methodology\n")
        f.write("Frequency: Quarterly\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"WINSORIZATION: Variance components at "
            f"{int(WINSOR_BOUNDS[0]*100)}%-{int(WINSOR_BOUNDS[1]*100)}% "
            f"(before calculating shares)\n\n"
        )
        f.write("VARIANCE-WEIGHTED RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(VW.to_string())
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("EQUAL-WEIGHTED RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(EW.to_string())
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        f.write("Variance-Weighted Mean (%):\n")
        f.write(f"  Market Information:    {vw_mean['MktInfoShare']:6.2f}%\n")
        f.write(f"  Firm-Specific Info:    {vw_mean['FirmInfoShare']:6.2f}%\n")
        f.write(f"  Noise:                 {vw_mean['NoiseShare']:6.2f}%\n\n")
        f.write("Equal-Weighted Mean (%):\n")
        f.write(f"  Market Information:    {ew_mean['MktInfoShare']:6.2f}%\n")
        f.write(f"  Firm-Specific Info:    {ew_mean['FirmInfoShare']:6.2f}%\n")
        f.write(f"  Noise:                 {ew_mean['NoiseShare']:6.2f}%\n")
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("QUARTERLY DIAGNOSTICS\n")
        f.write("-" * 80 + "\n")
        f.write(diagnostics_df.to_string(index=False))

    print("   - Saved: variance_decomposition_quarterly_summary.txt")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    return results_df, EW, VW


if __name__ == "__main__":
    results_df, EW, VW = main()
