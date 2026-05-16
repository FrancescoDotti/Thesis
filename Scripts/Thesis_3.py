#!/usr/bin/env python
# coding: utf-8
"""
4-way variance decomposition following Brogaard, Nguyen, Putnins, and Wu (2022).

All public functions are import-safe. Run as a script (python Thesis_3.py) to
execute the full Bloomberg-data pipeline and write outputs to thesis_3_outputs/.

Five-step procedure from Appendix A of "What Moves Stock Prices?":
1. Estimate the reduced-form VAR in Equation (A1), saving residuals and their
   variance/covariance matrix.
2. Estimate b1,0, c1,0, c2,0 from regressions of the reduced-form residuals
   (second and third equations in A2).
3. Estimate structural innovation variances using Equation (A3).
4. Estimate long-run cumulative return responses theta_rm, theta_x, theta_r
   using reduced-form IRFs with shocks from Equations (A4).
5. Combine structural variances and long-run responses to get variance
   components and shares per Equations (9) and (10).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

WINSOR_BOUNDS = (0.05, 0.95)
MIN_VALID_OBS = 20
RETURN_SCALE = 10000


def decompose_variance_single_stock(
    df_stock,
    market_ret_col="rm",
    stock_ret_col="r",
    volume_col="volume",
    price_col="price",
):
    """Decompose one stock-year into 4 variance components."""

    df = df_stock.copy()

    # Build signed dollar volume (Brogaard Appendix A).
    df["x"] = np.sign(df[stock_ret_col]) * df[price_col] * df[volume_col]

    var_data = df[[market_ret_col, "x", stock_ret_col]].dropna().copy()

    # Scale raw decimal returns to basis points for VAR stability.
    var_data[market_ret_col] = var_data[market_ret_col] * RETURN_SCALE
    var_data[stock_ret_col] = var_data[stock_ret_col] * RETURN_SCALE

    # Winsorize VAR inputs for robust estimation.
    for col in [market_ret_col, "x", stock_ret_col]:
        q_low = var_data[col].quantile(WINSOR_BOUNDS[0])
        q_high = var_data[col].quantile(WINSOR_BOUNDS[1])
        var_data[col] = var_data[col].clip(lower=q_low, upper=q_high)

    if len(var_data) < MIN_VALID_OBS:
        return None

    # Step 1: reduced-form VAR.
    try:
        var_result = VAR(var_data).fit(maxlags=5, trend="c")
    except Exception:
        return None

    used_lags = var_result.k_ar
    resids = var_result.resid
    e_rm = resids.iloc[:, 0]
    e_x = resids.iloc[:, 1]
    e_r = resids.iloc[:, 2]

    sigma2_erm = np.var(e_rm, ddof=1)
    sigma2_ex = np.var(e_x, ddof=1)
    sigma2_er = np.var(e_r, ddof=1)

    if sigma2_erm <= 0:
        return None

    # Step 2: structural parameters from Equation (A2).
    b10 = np.cov(e_x, e_rm)[0, 1] / np.var(e_rm, ddof=1)
    X_mat = np.column_stack([e_rm, e_x])
    c10, c20 = np.linalg.lstsq(X_mat, e_r, rcond=None)[0]

    # Step 3: structural innovation variances (Equation A3).
    sigma2_eps_rm = sigma2_erm
    sigma2_eps_x = sigma2_ex - b10**2 * sigma2_erm
    sigma2_eps_r = (
        sigma2_er
        - (c10**2 + 2 * c10 * c20 * b10) * sigma2_erm
        - c20**2 * sigma2_ex
    )

    if sigma2_eps_x < 0 or sigma2_eps_r < 0:
        return None

    # Lag-polynomial sum for stationarity check.
    params = var_result.params.values
    n_vars = 3
    A_sum = np.zeros((n_vars, n_vars))
    for lag in range(used_lags):
        start_row = 1 + lag * n_vars
        end_row = 1 + (lag + 1) * n_vars
        A_lag = params[start_row:end_row, :].T
        A_sum += A_lag

    max_eigenvalue = np.max(np.abs(np.linalg.eigvals(A_sum)))
    if max_eigenvalue >= 1.0:
        return None

    # Step 4: long-run structural multipliers.
    try:
        LR_matrix = np.linalg.inv(np.eye(n_vars) - A_sum)
    except np.linalg.LinAlgError:
        return None

    # B0_inv maps structural shocks to reduced-form residuals.
    B0_inv = np.array(
        [
            [1.0, 0.0, 0.0],
            [b10, 1.0, 0.0],
            [c10 + c20 * b10, c20, 1.0],
        ]
    )
    LR_structural = LR_matrix @ B0_inv

    theta_rm = LR_structural[2, 0]
    theta_x = LR_structural[2, 1]
    theta_r = LR_structural[2, 2]

    # Step 5: variance components (Equation 9).
    mkt_info = theta_rm**2 * sigma2_eps_rm
    private_info = theta_x**2 * sigma2_eps_x
    public_info = theta_r**2 * sigma2_eps_r
    sigma2_w = mkt_info + private_info + public_info

    # Noise = Var(r) - permanent information variance, floored at zero.
    actual_returns = var_data[stock_ret_col].iloc[used_lags:].values
    noise = max(np.var(actual_returns, ddof=1) - sigma2_w, 0)

    return {
        "MktInfo": mkt_info,
        "PrivateInfo": private_info,
        "PublicInfo": public_info,
        "Noise": noise,
        "k_ar": used_lags,
        "max_eigenvalue": max_eigenvalue,
        "sigma2_w": sigma2_w,
        "theta_rm": theta_rm,
        "theta_x": theta_x,
        "theta_r": theta_r,
        "b10": b10,
        "c10": c10,
        "c20": c20,
    }


def winsorize_by_period(df, component_cols, period_col="year", bounds=(0.05, 0.95)):
    """Winsorize component levels within each period."""
    df_winsorized = df.copy()
    for period_value in df_winsorized[period_col].unique():
        period_mask = df_winsorized[period_col] == period_value
        for col in component_cols:
            q_low = df_winsorized.loc[period_mask, col].quantile(bounds[0])
            q_high = df_winsorized.loc[period_mask, col].quantile(bounds[1])
            df_winsorized.loc[period_mask, col] = df_winsorized.loc[period_mask, col].clip(
                lower=q_low, upper=q_high
            )
    return df_winsorized


def build_yearly_diagnostics(results_df, share_cols, period_col="year", total_col="VarTotal"):
    """Build yearly diagnostics table for decomposition quality checks."""
    diagnostics = []
    for period_value, group in results_df.groupby(period_col):
        valid_share_mask = group[share_cols].notna().all(axis=1)
        positive_total_mask = group[total_col] > 0
        valid_mask = valid_share_mask & positive_total_mask
        share_sum = group.loc[valid_mask, share_cols].sum(axis=1)

        diagnostics.append(
            {
                "period": str(period_value),
                "n_stock_years": len(group),
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
            }
        )

    return pd.DataFrame(diagnostics).sort_values("period")


def process_stock_year_data(
    df,
    stock_col="ticker",
    year_col="year",
    market_ret_col="rm",
    stock_ret_col="r",
    volume_col="volume",
    price_col="price",
):
    """Run 4-way decomposition across all stock-years in a daily panel."""
    results = []
    for (stock, year), group in df.groupby([stock_col, year_col]):
        res = decompose_variance_single_stock(
            group,
            market_ret_col=market_ret_col,
            stock_ret_col=stock_ret_col,
            volume_col=volume_col,
            price_col=price_col,
        )
        if res is not None:
            res["stock"] = stock
            res["year"] = year
            res["n_obs"] = len(group)
            results.append(res)

    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        return results_df

    component_cols = ["MktInfo", "PrivateInfo", "PublicInfo", "Noise"]
    results_df = winsorize_by_period(
        results_df, component_cols=component_cols, period_col="year", bounds=WINSOR_BOUNDS
    )

    results_df["VarTotal"] = results_df[component_cols].sum(axis=1)
    results_df["MktInfoShare"] = 100 * results_df["MktInfo"] / results_df["VarTotal"]
    results_df["PrivateInfoShare"] = 100 * results_df["PrivateInfo"] / results_df["VarTotal"]
    results_df["PublicInfoShare"] = 100 * results_df["PublicInfo"] / results_df["VarTotal"]
    results_df["NoiseShare"] = 100 * results_df["Noise"] / results_df["VarTotal"]

    return results_df


def aggregate_variance_shares_fixed(results_df, share_cols=None):
    """Compute yearly equal-weighted and variance-weighted share averages."""
    if share_cols is None:
        share_cols = ["MktInfoShare", "PrivateInfoShare", "PublicInfoShare", "NoiseShare"]

    ew = results_df.groupby("year")[share_cols].mean()

    def vw_avg(group):
        clean = group.dropna(subset=share_cols + ["VarTotal"])
        clean = clean[clean["VarTotal"] > 0]
        if len(clean) == 0:
            return pd.Series({col: np.nan for col in share_cols})
        weights = clean["VarTotal"].values
        return pd.Series({col: np.average(clean[col].values, weights=weights) for col in share_cols})

    vw = results_df.groupby("year").apply(vw_avg)
    return ew, vw


def run_thesis3_from_daily_panel(daily_df):
    """
    Entry point for the panel pipeline.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Must include: stock, date, price, volume, stock_ret, market_ret
        (stock_ret and market_ret as raw decimal returns).

    Returns
    -------
    dict with keys: results, ew, vw, diagnostics
    """
    required_cols = {"stock", "date", "price", "volume", "stock_ret", "market_ret"}
    missing_cols = required_cols - set(daily_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for Thesis_3 adapter: {sorted(missing_cols)}")

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.rename(columns={"stock": "ticker", "stock_ret": "r", "market_ret": "rm"})
    df = df.dropna(subset=["ticker", "year", "date", "rm", "r", "volume", "price"])

    results_df = process_stock_year_data(
        df,
        stock_col="ticker",
        year_col="year",
        market_ret_col="rm",
        stock_ret_col="r",
        volume_col="volume",
        price_col="price",
    )

    if len(results_df) == 0:
        return {
            "results": results_df,
            "ew": pd.DataFrame(),
            "vw": pd.DataFrame(),
            "diagnostics": pd.DataFrame(),
        }

    share_cols = ["MktInfoShare", "PrivateInfoShare", "PublicInfoShare", "NoiseShare"]
    ew_df, vw_df = aggregate_variance_shares_fixed(results_df, share_cols=share_cols)
    diagnostics_df = build_yearly_diagnostics(results_df, share_cols, period_col="year", total_col="VarTotal")

    return {"results": results_df, "ew": ew_df, "vw": vw_df, "diagnostics": diagnostics_df}


if __name__ == "__main__":
    from data_interface import read_bloomberg_two_row_sheet, split_price_volume

    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "Data"
    OUTPUTS_DIR = PROJECT_DIR / "Outputs" / "thesis_3_outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    file_path = DATA_DIR / "data.xlsx"
    index = "SXXP Index"

    raw = read_bloomberg_two_row_sheet(file_path, sheet_name="daily_")
    parts = split_price_volume(raw)
    prices = parts["prices"].ffill(limit=5)
    volume = parts["volume"].ffill(limit=5)
    market_ret = prices[index].pct_change(fill_method=None)

    print(f"Data loaded from {file_path}")

    stock_tickers = [col for col in prices.columns if col != index]
    returns = prices[stock_tickers].pct_change(fill_method=None)

    # Build long-format panel. Pass raw decimal returns — scaling happens inside
    # decompose_variance_single_stock, matching the run_thesis3_from_daily_panel path.
    df_list = []
    for ticker in stock_tickers:
        df_list.append(
            pd.DataFrame(
                {
                    "date": prices.index,
                    "ticker": ticker,
                    "price": prices[ticker].values,
                    "volume": volume[ticker].values,
                    "r": returns[ticker].values,
                    "rm": market_ret.values,
                }
            )
        )

    df = pd.concat(df_list, ignore_index=True)
    df["year"] = df["date"].dt.year
    df = df.dropna(subset=["r", "rm", "price", "volume"])
    df = df.groupby(["ticker", "year"]).filter(lambda x: len(x) >= MIN_VALID_OBS)
    df = df[["ticker", "year", "date", "rm", "r", "volume", "price"]]

    print(f"Final dataset shape: {df.shape}")
    print(f"Stocks: {df['ticker'].nunique()}")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")
    print(f"\nFirst few rows:\n{df.head()}")

    results_df = process_stock_year_data(df)

    share_cols = ["MktInfoShare", "PrivateInfoShare", "PublicInfoShare", "NoiseShare"]
    EW, VW = aggregate_variance_shares_fixed(results_df, share_cols=share_cols)

    print("VW head:\n", VW.head(10))
    print("\nVW tail:\n", VW.tail(10))
    print("\nVW info:")
    print(VW.info())

    print("\n=== Data Quality Checks ===")
    print(f"Total stock-years: {len(results_df)}")
    print(f"Stock-years with NaN in shares: {results_df[share_cols].isna().any(axis=1).sum()}")
    print(f"Stock-years with NaN in VarTotal: {results_df['VarTotal'].isna().sum()}")
    print(f"Stock-years with VarTotal <= 0: {(results_df['VarTotal'] <= 0).sum()}")

    diagnostics_df = build_yearly_diagnostics(results_df, share_cols, period_col="year", total_col="VarTotal")
    print("\nHarmonized yearly diagnostics (first 10 rows):")
    print(diagnostics_df.head(10))

    print("\nStocks per year:")
    print(results_df.groupby("year").size())

    print("\nSample of VarTotal by year:")
    print(results_df.groupby("year")["VarTotal"].agg(["count", "mean", "min", "max"]))

    print("\nVAR stationarity (max eigenvalue):")
    print(f"  Mean: {results_df['max_eigenvalue'].mean():8.4f}")
    print(f"  Max:  {results_df['max_eigenvalue'].max():8.4f}")
    print(f"  % with max_eig < 0.99: {(results_df['max_eigenvalue'] < 0.99).sum() / len(results_df) * 100:.1f}%")

    # --- Plots ---
    colors_4way = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    def _label(col):
        return (
            col.replace("Share", "")
            .replace("MktInfo", "Market")
            .replace("PrivateInfo", "Private")
            .replace("PublicInfo", "Public")
        )

    ax = axes[0, 0]
    for col, color in zip(share_cols, colors_4way):
        ax.plot(VW.index, VW[col], marker="o", label=_label(col), linewidth=2, color=color)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("Variance-Weighted Shares Over Time (4-way)", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    ax = axes[0, 1]
    for col, color in zip(share_cols, colors_4way):
        ax.plot(EW.index, EW[col], marker="s", label=_label(col), linewidth=2, color=color)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("Equal-Weighted Shares Over Time (4-way)", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    ax = axes[1, 0]
    x = np.arange(len(VW))
    bottom = np.zeros(len(VW))
    for col, color in zip(share_cols, colors_4way):
        ax.bar(x, VW[col], 0.8, bottom=bottom, label=_label(col), color=color)
        bottom += VW[col].values
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title("Variance-Weighted Shares (Stacked, 4-way)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(VW.index, rotation=45, fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 100])

    ax = axes[1, 1]
    latest_year = int(results_df["year"].max())
    latest_data = results_df[results_df["year"] == latest_year][share_cols]
    bp = ax.boxplot(
        [latest_data[col].dropna() for col in share_cols],
        labels=["Market", "Private", "Public", "Noise"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], colors_4way):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Share (%)", fontsize=11)
    ax.set_title(f"Cross-Sectional Distribution ({latest_year})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "variance_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   - Saved: variance_decomposition.png")

    # --- Save outputs ---
    results_df.to_csv(OUTPUTS_DIR / "variance_decomposition_results.csv", index=False)
    VW.to_csv(OUTPUTS_DIR / "variance_decomposition_VW.csv")
    EW.to_csv(OUTPUTS_DIR / "variance_decomposition_EW.csv")
    diagnostics_df.to_csv(OUTPUTS_DIR / "variance_decomposition_diagnostics.csv", index=False)
    print("   - Saved: variance_decomposition_results.csv, _VW.csv, _EW.csv, _diagnostics.csv")

    vw_mean = VW.mean()
    ew_mean = EW.mean()

    with open(OUTPUTS_DIR / "variance_decomposition_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("VARIANCE DECOMPOSITION (4-way): What Moves Stock Prices\n")
        f.write("Brogaard, Nguyen, Putnins & Wu (2022) Methodology\n")
        f.write("Extended 3-Variable VAR: Market Return, Signed Dollar Volume, Stock Return\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"WINSORIZATION: Variance components at "
            f"{int(WINSOR_BOUNDS[0]*100)}%-{int(WINSOR_BOUNDS[1]*100)}% "
            f"(before calculating shares)\n\n"
        )
        f.write("COMPONENTS:\n")
        f.write("  1. Market Information:  driven by market-wide shocks\n")
        f.write("  2. Private Information: driven by private information (via signed volume)\n")
        f.write("  3. Public Information:  driven by public information (own return residual)\n")
        f.write("  4. Noise:               residual unexplained variance\n\n")
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
        f.write(f"  Market Information:     {vw_mean['MktInfoShare']:6.2f}%\n")
        f.write(f"  Private Information:    {vw_mean['PrivateInfoShare']:6.2f}%\n")
        f.write(f"  Public Information:     {vw_mean['PublicInfoShare']:6.2f}%\n")
        f.write(f"  Noise:                  {vw_mean['NoiseShare']:6.2f}%\n\n")
        f.write("Equal-Weighted Mean (%):\n")
        f.write(f"  Market Information:     {ew_mean['MktInfoShare']:6.2f}%\n")
        f.write(f"  Private Information:    {ew_mean['PrivateInfoShare']:6.2f}%\n")
        f.write(f"  Public Information:     {ew_mean['PublicInfoShare']:6.2f}%\n")
        f.write(f"  Noise:                  {ew_mean['NoiseShare']:6.2f}%\n")
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("HARMONIZED YEARLY DIAGNOSTICS\n")
        f.write("-" * 80 + "\n")
        f.write(diagnostics_df.to_string(index=False))

    print("   - Saved: variance_decomposition_summary.txt")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n=== VARIANCE-WEIGHTED MEANS (Full Period) ===")
    print(VW.mean().round(2))
    print("\n=== EQUAL-WEIGHTED MEANS (Full Period) ===")
    print(EW.mean().round(2))
