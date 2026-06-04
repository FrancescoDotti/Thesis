"""
Quarterly Variance Decomposition — Noise Methodology Comparison
================================================================

Mirror of `Thesis_2_quarterly.py` that computes BOTH noise variants and
saves a line-graph comparison.

Noise methodologies compared
----------------------------
(A) Residual method (the one used in Thesis_2_quarterly.py):
        Noise_A = max( Var(r_t) - theta_m^2 * sigma_eps_m^2
                                - theta_s^2 * sigma_eps_s^2,  0 )

(B) Beveridge–Nelson method (the one used in Brogaard et al. 2022):
        eps_m,t       = e_market,t                 # market exogenous
        eps_s,t       = e_stock,t - b10 * e_market,t   # structural shock
        w_t           = theta_m * eps_m,t + theta_s * eps_s,t
        Delta s_t     = r_t - w_t                    # drift omitted (doesn't affect variance)
        Noise_B       = Var(Delta s_t)

Outputs
-------
Outputs/noise_comparison/
  - noise_comparison_results.csv         (one row per stock-quarter, both shares)
  - noise_comparison_EW.csv              (equal-weighted by quarter)
  - noise_comparison_VW.csv              (variance-weighted by quarter)
  - noise_comparison_timeseries.png      (the line graph)
  - noise_comparison_scatter.png         (per-cell comparison)
  - noise_comparison_summary.txt         (overall statistics)
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
OUTPUTS_DIR = PROJECT_DIR / "Outputs" / "noise_comparison"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

WINSOR_BOUNDS = (0.05, 0.95)
RETURN_SCALE = 10000
MIN_VALID_OBS = 20
MAX_LAGS = 1


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
    Decompose variance of a single stock-quarter, returning BOTH noise estimates.

    The decomposition logic (VAR, structural innovations, theta) is identical to
    Thesis_2_quarterly.py — only the noise side is extended.
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

    # Contemporaneous loading (phi in the thesis text)
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
    var_actual = np.var(actual_returns, ddof=1)

    # ── (A) Residual-method noise ──────────────────────────────────────────────
    Noise_A = max(var_actual - (MktInfo + FirmInfo), 0.0)

    # ── (B) Beveridge–Nelson noise via pricing-error series ────────────────────
    # Structural innovations
    eps_market_series = e_market                          # market exogenous
    eps_stock_series = e_stock - b10 * e_market           # structural stock shock

    # Permanent (information) component of each period's return
    w_t = theta_market * eps_market_series + theta_stock * eps_stock_series

    # Pricing-error change — drift omitted because it's a constant (BNPW fn. 13)
    delta_s = actual_returns - w_t

    Noise_B = np.var(delta_s, ddof=1)

    # Totals and shares for both methodologies
    TotalVar_A = MktInfo + FirmInfo + Noise_A
    TotalVar_B = MktInfo + FirmInfo + Noise_B
    if TotalVar_A <= 0 or TotalVar_B <= 0:
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
        "var_actual": var_actual,
        "MktInfo": MktInfo,
        "FirmInfo": FirmInfo,
        # ── Method A (residual) ───────────────────────────────────────────────
        "Noise_A": Noise_A,
        "TotalVar_A": TotalVar_A,
        "MktInfoShare_A": 100 * MktInfo / TotalVar_A,
        "FirmInfoShare_A": 100 * FirmInfo / TotalVar_A,
        "NoiseShare_A": 100 * Noise_A / TotalVar_A,
        # ── Method B (BN Delta-s) ─────────────────────────────────────────────
        "Noise_B": Noise_B,
        "TotalVar_B": TotalVar_B,
        "MktInfoShare_B": 100 * MktInfo / TotalVar_B,
        "FirmInfoShare_B": 100 * FirmInfo / TotalVar_B,
        "NoiseShare_B": 100 * Noise_B / TotalVar_B,
    }


def run_decomposition(daily_df):
    """Loop over all (stock, quarter) cells and collect both decompositions."""

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

    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def decompose_all_stocks_period(market_ret, stock_prices, period_label):
    """Decompose variance for all stocks in a given period (wide-format input)."""

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


def winsorize_components(df, bounds=(0.05, 0.95)):
    """Winsorize variance components by period and recalculate shares for BOTH methods."""

    df_w = df.copy()
    component_cols = ["MktInfo", "FirmInfo", "Noise_A", "Noise_B"]

    for period in df_w["period"].unique():
        mask = df_w["period"] == period
        for col in component_cols:
            if col in df_w.columns:
                q_low = df_w.loc[mask, col].quantile(bounds[0])
                q_high = df_w.loc[mask, col].quantile(bounds[1])
                df_w.loc[mask, col] = df_w.loc[mask, col].clip(q_low, q_high)

    # Recompute shares after winsorisation
    df_w["TotalVar_A"] = df_w["MktInfo"] + df_w["FirmInfo"] + df_w["Noise_A"]
    df_w["MktInfoShare_A"] = 100 * df_w["MktInfo"] / df_w["TotalVar_A"]
    df_w["FirmInfoShare_A"] = 100 * df_w["FirmInfo"] / df_w["TotalVar_A"]
    df_w["NoiseShare_A"] = 100 * df_w["Noise_A"] / df_w["TotalVar_A"]

    df_w["TotalVar_B"] = df_w["MktInfo"] + df_w["FirmInfo"] + df_w["Noise_B"]
    df_w["MktInfoShare_B"] = 100 * df_w["MktInfo"] / df_w["TotalVar_B"]
    df_w["FirmInfoShare_B"] = 100 * df_w["FirmInfo"] / df_w["TotalVar_B"]
    df_w["NoiseShare_B"] = 100 * df_w["Noise_B"] / df_w["TotalVar_B"]

    return df_w


def aggregate_by_period(results_df):
    """Compute EW and VW aggregates per quarter for both methodologies."""

    share_cols_A = ["MktInfoShare_A", "FirmInfoShare_A", "NoiseShare_A"]
    share_cols_B = ["MktInfoShare_B", "FirmInfoShare_B", "NoiseShare_B"]
    share_cols = share_cols_A + share_cols_B

    EW = results_df.groupby("period")[share_cols].mean()

    def vw_aggregate(group):
        out = {}
        # Method A weights
        clean_A = group.dropna(subset=share_cols_A + ["TotalVar_A"])
        clean_A = clean_A[clean_A["TotalVar_A"] > 0]
        if len(clean_A) > 0:
            w_A = clean_A["TotalVar_A"].values
            for c in share_cols_A:
                out[c] = np.average(clean_A[c].values, weights=w_A)
        else:
            for c in share_cols_A:
                out[c] = np.nan
        # Method B weights
        clean_B = group.dropna(subset=share_cols_B + ["TotalVar_B"])
        clean_B = clean_B[clean_B["TotalVar_B"] > 0]
        if len(clean_B) > 0:
            w_B = clean_B["TotalVar_B"].values
            for c in share_cols_B:
                out[c] = np.average(clean_B[c].values, weights=w_B)
        else:
            for c in share_cols_B:
                out[c] = np.nan
        return pd.Series(out)

    VW = results_df.groupby("period").apply(vw_aggregate)
    return EW, VW


def plot_timeseries(EW, VW, save_path):
    """Line graph: NoiseShare_A vs NoiseShare_B over time, EW and VW panels."""

    period_labels = EW.index.tolist()
    x = np.arange(len(period_labels))
    tick_positions = [i for i, p in enumerate(period_labels) if p.endswith("Q1")]
    tick_labels = [p[:4] for p in period_labels if p.endswith("Q1")]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1 — Equal-weighted
    ax = axes[0]
    ax.plot(x, EW["NoiseShare_A"], label="Method A — residual: $\\mathrm{Var}(r) - \\mathrm{MktInfo} - \\mathrm{FirmInfo}$",
            linewidth=1.8, color="#1f77b4")
    ax.plot(x, EW["NoiseShare_B"], label="Method B — Beveridge–Nelson: $\\mathrm{Var}(\\Delta s_t)$",
            linewidth=1.8, color="#d62728", linestyle="--")
    ax.set_ylabel("Noise share (%)", fontsize=11)
    ax.set_title("Equal-weighted noise share across STOXX 600 (quarterly)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2 — Variance-weighted
    ax = axes[1]
    ax.plot(x, VW["NoiseShare_A"], label="Method A — residual",
            linewidth=1.8, color="#1f77b4")
    ax.plot(x, VW["NoiseShare_B"], label="Method B — Beveridge–Nelson",
            linewidth=1.8, color="#d62728", linestyle="--")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Noise share (%)", fontsize=11)
    ax.set_title("Variance-weighted noise share across STOXX 600 (quarterly)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter(results_df, save_path):
    """Per-cell scatter of NoiseShare_A vs NoiseShare_B with 45-degree reference."""

    df = results_df.dropna(subset=["NoiseShare_A", "NoiseShare_B"])
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["NoiseShare_A"], df["NoiseShare_B"],
               s=6, alpha=0.25, color="#1a5276", edgecolors="none")

    lo = min(df["NoiseShare_A"].min(), df["NoiseShare_B"].min())
    hi = max(df["NoiseShare_A"].max(), df["NoiseShare_B"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=1, label="45° line")

    corr = df[["NoiseShare_A", "NoiseShare_B"]].corr().iloc[0, 1]
    ax.set_xlabel("Noise share — Method A (residual, %)", fontsize=11)
    ax.set_ylabel("Noise share — Method B (Beveridge–Nelson, %)", fontsize=11)
    ax.set_title(f"Stock-quarter noise share: A vs B  |  $\\rho$ = {corr:.3f}  |  N = {len(df):,}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 80)
    print("QUARTERLY VARIANCE DECOMPOSITION — NOISE METHODOLOGY COMPARISON")
    print("=" * 80)
    print("\nMethod A: Noise = max(Var(r) - MktInfo - FirmInfo, 0)   (residual)")
    print("Method B: Noise = Var(Delta s_t)                          (Beveridge–Nelson)")
    print(f"\nLags = {MAX_LAGS}   |   Winsor = "
          f"{int(WINSOR_BOUNDS[0]*100)}%–{int(WINSOR_BOUNDS[1]*100)}%   |   "
          f"Min obs/cell = {MIN_VALID_OBS}")
    print("=" * 80)

    # ── Step 1: load daily data ───────────────────────────────────────────────
    print("\n1. Loading data...")
    data = load_data()
    prices = data["prices"]
    market_ret = data["market_ret"]
    print(f"   - {len(prices)} dates, {len(prices.columns)} stocks (incl. index)")
    print(f"   - Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    # ── Step 2: per-stock-quarter decomposition ───────────────────────────────
    print("\n2. Computing variance decomposition by quarter (both methods)...")
    all_results = []
    quarters = sorted(
        {(d.year, d.quarter) for d in prices.index},
        key=lambda yq: (yq[0], yq[1]),
    )
    for year, quarter in quarters:
        period_label = f"{year}-Q{quarter}"
        print(f"   Processing {period_label}...", end=" ")
        mask = (prices.index.year == year) & (prices.index.quarter == quarter)
        quarter_results = decompose_all_stocks_period(
            market_ret[mask], prices[mask], period_label,
        )
        if quarter_results is not None:
            all_results.append(quarter_results)
            print(f"✓ {len(quarter_results)} stocks")
        else:
            print("✗ none")

    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\n   Total stock-quarter observations: {len(results_df):,}")

    # ── Step 3: winsorise components and recompute shares ────────────────────
    print(f"\n3. Winsorising components at "
          f"{int(WINSOR_BOUNDS[0]*100)}%–{int(WINSOR_BOUNDS[1]*100)}% by quarter...")
    results_df = winsorize_components(results_df, bounds=WINSOR_BOUNDS)

    # ── Step 4: aggregate ────────────────────────────────────────────────────
    print("\n4. Aggregating across stocks (EW & VW)...")
    EW, VW = aggregate_by_period(results_df)

    # ── Summary statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL SAMPLE STATISTICS (averaged across quarters)")
    print("=" * 80)

    overall = {
        "EW Noise (A — residual)":  EW["NoiseShare_A"].mean(),
        "EW Noise (B — BN Delta-s)": EW["NoiseShare_B"].mean(),
        "VW Noise (A — residual)":  VW["NoiseShare_A"].mean(),
        "VW Noise (B — BN Delta-s)": VW["NoiseShare_B"].mean(),
    }
    for k, v in overall.items():
        print(f"  {k:35s} {v:6.2f}%")

    corr_cell = results_df[["NoiseShare_A", "NoiseShare_B"]].corr().iloc[0, 1]
    diff = (results_df["NoiseShare_A"] - results_df["NoiseShare_B"]).dropna()
    print(f"\n  Cross-cell correlation A vs B :  {corr_cell:6.4f}")
    print(f"  Mean (A - B)                  : {diff.mean():+6.3f} pp")
    print(f"  Median (A - B)                : {diff.median():+6.3f} pp")
    print(f"  Std  (A - B)                  :  {diff.std():6.3f} pp")

    # ── Step 5: visualisations ───────────────────────────────────────────────
    print("\n5. Producing plots...")
    ts_path = OUTPUTS_DIR / "noise_comparison_timeseries.png"
    sc_path = OUTPUTS_DIR / "noise_comparison_scatter.png"
    plot_timeseries(EW, VW, ts_path)
    plot_scatter(results_df, sc_path)
    print(f"   - Saved {ts_path.name}")
    print(f"   - Saved {sc_path.name}")

    # ── Step 6: save tables ──────────────────────────────────────────────────
    print("\n6. Saving tables...")
    results_df.to_csv(OUTPUTS_DIR / "noise_comparison_results.csv", index=False)
    EW.to_csv(OUTPUTS_DIR / "noise_comparison_EW.csv")
    VW.to_csv(OUTPUTS_DIR / "noise_comparison_VW.csv")
    print(f"   - Saved noise_comparison_results.csv, _EW.csv, _VW.csv")

    with open(OUTPUTS_DIR / "noise_comparison_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("QUARTERLY VARIANCE DECOMPOSITION — NOISE METHODOLOGY COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write("Method A — residual    : Noise = max(Var(r) - MktInfo - FirmInfo, 0)\n")
        f.write("Method B — BN Delta-s  : Noise = Var(Delta s_t)\n\n")
        f.write(f"Sample: {len(results_df):,} stock-quarter cells\n")
        f.write(f"Lags = {MAX_LAGS}   |   Winsor = "
                f"{int(WINSOR_BOUNDS[0]*100)}%-{int(WINSOR_BOUNDS[1]*100)}%\n\n")
        f.write("Overall noise share (% across quarters)\n")
        f.write("-" * 80 + "\n")
        for k, v in overall.items():
            f.write(f"  {k:35s} {v:6.2f}%\n")
        f.write("\nPer-cell diagnostics\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Cross-cell correlation A vs B :  {corr_cell:6.4f}\n")
        f.write(f"  Mean (A - B)                  : {diff.mean():+6.3f} pp\n")
        f.write(f"  Median (A - B)                : {diff.median():+6.3f} pp\n")
        f.write(f"  Std  (A - B)                  :  {diff.std():6.3f} pp\n")
        f.write("\nEqual-weighted shares by quarter\n")
        f.write("-" * 80 + "\n")
        f.write(EW[["NoiseShare_A", "NoiseShare_B"]].to_string())
        f.write("\n\nVariance-weighted shares by quarter\n")
        f.write("-" * 80 + "\n")
        f.write(VW[["NoiseShare_A", "NoiseShare_B"]].to_string())
    print(f"   - Saved noise_comparison_summary.txt")

    print("\n" + "=" * 80)
    print("DONE.")
    print("=" * 80)

    return results_df, EW, VW


if __name__ == "__main__":
    results_df, EW, VW = main()
