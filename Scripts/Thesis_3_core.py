"""
Import-safe core functions for Thesis_3 4-way variance decomposition.

This module contains only reusable functions and no top-level execution.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

WINSOR_BOUNDS = (0.05, 0.95)
MIN_VALID_OBS = 20


def decompose_variance_single_stock(
    df_stock,
    market_ret_col="rm",
    stock_ret_col="r",
    volume_col="volume",
    price_col="price",
    n_lags=5,
):
    """Decompose one stock-year into 4 variance components."""

    # Copy input to avoid mutating caller data.
    df = df_stock.copy()

    # Build signed dollar volume used by the Brogaard Appendix A setup.
    df["x"] = np.sign(df[stock_ret_col]) * df[price_col] * df[volume_col]

    # Keep variables in required VAR order: market return, signed volume, stock return.
    var_data = df[[market_ret_col, "x", stock_ret_col]].dropna()

    # Winsorize each VAR input series for robust estimation.
    for col in [market_ret_col, "x", stock_ret_col]:
        q_low = var_data[col].quantile(WINSOR_BOUNDS[0])
        q_high = var_data[col].quantile(WINSOR_BOUNDS[1])
        var_data[col] = var_data[col].clip(lower=q_low, upper=q_high)

    if len(var_data) < MIN_VALID_OBS:
        return None

    try:
        var_result = VAR(var_data).fit(maxlags=n_lags, trend="c")
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

    # Structural mapping from reduced-form residuals.
    b10 = np.cov(e_x, e_rm)[0, 1] / np.var(e_rm, ddof=1)
    X_mat = np.column_stack([e_rm, e_x])
    c10, c20 = np.linalg.lstsq(X_mat, e_r, rcond=None)[0]

    sigma2_eps_rm = sigma2_erm
    sigma2_eps_x = sigma2_ex - b10**2 * sigma2_erm
    sigma2_eps_r = sigma2_er - (c10**2 + 2 * c10 * c20 * b10) * sigma2_erm - c20**2 * sigma2_ex

    if sigma2_eps_x < 0 or sigma2_eps_r < 0:
        return None

    # Build reduced-form lag polynomial sum for stationarity checks.
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

    # Long-run structural multipliers.
    try:
        LR_matrix = np.linalg.inv(np.eye(n_vars) - A_sum)
    except np.linalg.LinAlgError:
        return None

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

    # Information components.
    mkt_info = theta_rm**2 * sigma2_eps_rm
    private_info = theta_x**2 * sigma2_eps_x
    public_info = theta_r**2 * sigma2_eps_r

    # Noise from residual model fit.
    eps_rm = e_rm.values
    eps_x = e_x.values - b10 * e_rm.values
    eps_r = e_r.values - c10 * e_rm.values - c20 * e_x.values

    a0 = var_result.params.iloc[0, 2]
    fitted_info_return = theta_rm * eps_rm + theta_x * eps_x + theta_r * eps_r

    actual_returns = var_data[stock_ret_col].iloc[used_lags:].values
    if len(actual_returns) != len(fitted_info_return):
        return None

    noise_returns = actual_returns - a0 - fitted_info_return
    noise = max(np.var(noise_returns, ddof=1), 0)

    return {
        "MktInfo": mkt_info,
        "PrivateInfo": private_info,
        "PublicInfo": public_info,
        "Noise": noise,
        "k_ar": used_lags,
        "max_eigenvalue": max_eigenvalue,
    }


def winsorize_by_period(df, component_cols, period_col="year", bounds=(0.05, 0.95)):
    """Winsorize component levels by year."""

    df_winsorized = df.copy()
    for period_value in df_winsorized[period_col].unique():
        period_mask = df_winsorized[period_col] == period_value
        for col in component_cols:
            q_low = df_winsorized.loc[period_mask, col].quantile(bounds[0])
            q_high = df_winsorized.loc[period_mask, col].quantile(bounds[1])
            df_winsorized.loc[period_mask, col] = df_winsorized.loc[period_mask, col].clip(lower=q_low, upper=q_high)
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

    # Winsorize component levels by year.
    component_cols = ["MktInfo", "PrivateInfo", "PublicInfo", "Noise"]
    results_df = winsorize_by_period(results_df, component_cols=component_cols, period_col="year", bounds=WINSOR_BOUNDS)

    # Convert levels to shares.
    results_df["VarTotal"] = results_df["MktInfo"] + results_df["PrivateInfo"] + results_df["PublicInfo"] + results_df["Noise"]
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
    """Adapter entry point matching the Thesis_2 adapter return format."""

    required_cols = {"stock", "date", "price", "volume", "stock_ret", "market_ret"}
    missing_cols = required_cols - set(daily_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for Thesis_3 adapter: {sorted(missing_cols)}")

    # Map canonical names to function-specific names.
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
        return {"results": results_df, "ew": pd.DataFrame(), "vw": pd.DataFrame(), "diagnostics": pd.DataFrame()}

    share_cols = ["MktInfoShare", "PrivateInfoShare", "PublicInfoShare", "NoiseShare"]
    ew_df, vw_df = aggregate_variance_shares_fixed(results_df, share_cols=share_cols)
    diagnostics_df = build_yearly_diagnostics(results_df, share_cols, period_col="year", total_col="VarTotal")

    return {"results": results_df, "ew": ew_df, "vw": vw_df, "diagnostics": diagnostics_df}
