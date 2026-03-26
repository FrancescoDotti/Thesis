"""
Essay I panel builder pipeline.

This script builds a firm-year panel for 2015-2025 by combining:
1) Daily variance decomposition outputs (Thesis_3 default or Thesis_2 optional)
2) Quarterly firm controls aggregated to yearly values

It saves a merged panel and simple diagnostics to Thesis/Outputs/essay1_pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data_interface import parse_data_workbook, summarize_parsed_panel
from Thesis_2 import run_thesis2_from_daily_panel
from Thesis_3_core import run_thesis3_from_daily_panel


# Resolve project paths from the Thesis root directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
OUTPUTS_DIR = PROJECT_DIR / "Outputs" / "essay1_pipeline"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric and keep invalid values as NaN."""

    return pd.to_numeric(series, errors="coerce")


def aggregate_quarterly_controls_to_year(quarterly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate quarterly controls into a yearly firm-level table.

    Deterministic yearly rules used in this MVP:
    - esg_score_y: yearly mean of ESG_SCORE
    - size_y: yearly mean of log(CUR_MKT_CAP)
    - book_to_market_y: yearly mean of HEADLINE_BVPS (proxy in current data)
    - leverage_y: yearly mean of FNCL_LVRG
    - analyst_coverage_y: yearly mean of TOT_ANALYST_REC
    - gics_industry_y: first non-null GICS_INDUSTRY in year
    """

    df = quarterly_df.copy()

    # Convert control fields to numeric so aggregation is robust.
    for col in [
        "ESG_SCORE",
        "CUR_MKT_CAP",
        "HEADLINE_BVPS",
        "FNCL_LVRG",
        "TOT_ANALYST_REC",
    ]:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])

    # Build size using log market cap and protect against non-positive values.
    if "CUR_MKT_CAP" in df.columns:
        positive_mkt_cap = df["CUR_MKT_CAP"].where(df["CUR_MKT_CAP"] > 0)
        df["size_y_raw"] = np.log(positive_mkt_cap)
    else:
        df["size_y_raw"] = np.nan

    # Build yearly controls with explicit column checks to avoid hidden fallbacks.
    grouped = df.groupby(["stock", "year"])
    yearly = grouped.size().reset_index(name="n_quarters")

    def add_yearly_mean(out_df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
        """Add yearly mean of a source column if present, otherwise fill with NaN."""

        if source_col in df.columns:
            stats = grouped[source_col].mean().reset_index(name=target_col)
            out_df = out_df.merge(stats, on=["stock", "year"], how="left")
        else:
            out_df[target_col] = np.nan
        return out_df

    yearly = add_yearly_mean(yearly, "ESG_SCORE", "esg_score_y")
    yearly = add_yearly_mean(yearly, "size_y_raw", "size_y")
    yearly = add_yearly_mean(yearly, "HEADLINE_BVPS", "book_to_market_y")
    yearly = add_yearly_mean(yearly, "FNCL_LVRG", "leverage_y")
    yearly = add_yearly_mean(yearly, "TOT_ANALYST_REC", "analyst_coverage_y")

    # Add a simple yearly industry code from the first available quarter in the year.
    if "GICS_INDUSTRY" in df.columns:
        gics = (
            df.dropna(subset=["GICS_INDUSTRY"])
            .sort_values(["stock", "year", "date"])
            .groupby(["stock", "year"], as_index=False)
            .first()[["stock", "year", "GICS_INDUSTRY"]]
            .rename(columns={"GICS_INDUSTRY": "gics_industry_y"})
        )
        yearly = yearly.merge(gics, on=["stock", "year"], how="left")
    else:
        yearly["gics_industry_y"] = np.nan

    return yearly


def standardize_decomposition_output(raw_results: pd.DataFrame, decomp_model: str) -> pd.DataFrame:
    """
    Map decomposition outputs into a common schema while preserving source columns.

    Notes:
    - Thesis_3 keeps PrivateInfoShare and PublicInfoShare as separate outcomes.
    - No aggregate information-share column is created for Thesis_3.
    """

    df = raw_results.copy()

    # Harmonize year key across both decomposition scripts.
    if "period" in df.columns and "year" not in df.columns:
        df["year"] = pd.to_numeric(df["period"], errors="coerce")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["decomp_model"] = decomp_model

    # Create a shared noise proxy used as baseline informativeness complement.
    if "NoiseShare" in df.columns:
        df["noise_share_proxy"] = pd.to_numeric(df["NoiseShare"], errors="coerce")
    else:
        df["noise_share_proxy"] = np.nan

    # Build a stable total variance column regardless of source naming.
    if "VarTotal" in df.columns:
        df["total_var"] = pd.to_numeric(df["VarTotal"], errors="coerce")
    elif "TotalVar" in df.columns:
        df["total_var"] = pd.to_numeric(df["TotalVar"], errors="coerce")
    else:
        df["total_var"] = np.nan

    # Add optional harmonized columns without dropping original source columns.
    for col in [
        "MktInfoShare",
        "FirmInfoShare",
        "PrivateInfoShare",
        "PublicInfoShare",
        "NoiseShare",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep a clear, panel-first column order.
    base_cols = [
        "stock",
        "year",
        "decomp_model",
        "noise_share_proxy",
        "total_var",
        "MktInfoShare",
        "FirmInfoShare",
        "PrivateInfoShare",
        "PublicInfoShare",
        "NoiseShare",
    ]
    other_cols = [col for col in df.columns if col not in base_cols]

    return df[base_cols + other_cols].copy()


def build_decomposition_panel(daily_df: pd.DataFrame, decomp_model: str) -> Dict[str, pd.DataFrame]:
    """Run the selected decomposition adapter and return standardized outputs."""

    if decomp_model == "thesis3":
        outputs = run_thesis3_from_daily_panel(daily_df)
    elif decomp_model == "thesis2":
        outputs = run_thesis2_from_daily_panel(daily_df)
    else:
        raise ValueError(f"Unknown decomposition model: {decomp_model}")

    outputs["results"] = standardize_decomposition_output(outputs["results"], decomp_model=decomp_model)
    return outputs


def apply_sample_filters(panel_df: pd.DataFrame, start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    """Apply panel filters for target year window and valid variance-share ranges."""

    df = panel_df.copy()

    # Keep only the target sample window.
    df = df[df["year"].between(start_year, end_year, inclusive="both")]

    # Ensure key variance-share fields are in valid 0-100 range when present.
    share_cols = ["MktInfoShare", "FirmInfoShare", "PrivateInfoShare", "PublicInfoShare", "NoiseShare"]
    for col in share_cols:
        valid_mask = df[col].isna() | ((df[col] >= 0) & (df[col] <= 100))
        df = df[valid_mask]

    # Keep only rows with a valid noise proxy and positive/known total variance.
    df = df[df["noise_share_proxy"].notna()]
    if "total_var" in df.columns:
        df = df[(df["total_var"].isna()) | (df["total_var"] > 0)]

    return df.reset_index(drop=True)


def build_merge_diagnostics(pre_merge: pd.DataFrame, post_merge: pd.DataFrame) -> pd.DataFrame:
    """Create simple coverage and merge-rate diagnostics by year."""

    pre = pre_merge.groupby("year", as_index=False).agg(decomp_rows=("stock", "count"), decomp_stocks=("stock", "nunique"))
    post = post_merge.groupby("year", as_index=False).agg(merged_rows=("stock", "count"), merged_stocks=("stock", "nunique"))
    out = pre.merge(post, on="year", how="left")

    out["merged_rows"] = out["merged_rows"].fillna(0)
    out["merged_stocks"] = out["merged_stocks"].fillna(0)
    out["merge_retention_pct"] = np.where(out["decomp_rows"] > 0, 100 * out["merged_rows"] / out["decomp_rows"], np.nan)

    return out.sort_values("year").reset_index(drop=True)


def _fit_clustered_fe_ols(df: pd.DataFrame, outcome_col: str) -> pd.Series:
    """
    Fit OLS with firm and year fixed effects and stock-clustered standard errors.

    Model:
        outcome ~ ESG + controls + C(stock) + C(year)
    """

    # Keep regression variables explicit and readable.
    reg_df = df[
        [
            "stock",
            "year",
            outcome_col,
            "esg_score_y",
            "size_y",
            "book_to_market_y",
            "leverage_y",
            "analyst_coverage_y",
        ]
    ].copy()

    # Enforce numeric regressors before model estimation.
    for col in [
        outcome_col,
        "esg_score_y",
        "size_y",
        "book_to_market_y",
        "leverage_y",
        "analyst_coverage_y",
    ]:
        reg_df[col] = pd.to_numeric(reg_df[col], errors="coerce")

    # Drop incomplete rows for a clean estimation sample.
    reg_df = reg_df.dropna()

    # Guard against empty or degenerate samples.
    if len(reg_df) == 0:
        return pd.Series(
            {
                "outcome": outcome_col,
                "n_obs": 0,
                "n_stocks": 0,
                "coef_esg": np.nan,
                "se_esg": np.nan,
                "t_esg": np.nan,
                "p_esg": np.nan,
                "r2": np.nan,
            }
        )

    if reg_df["stock"].nunique() < 2 or reg_df["year"].nunique() < 2:
        return pd.Series(
            {
                "outcome": outcome_col,
                "n_obs": int(len(reg_df)),
                "n_stocks": int(reg_df["stock"].nunique()),
                "coef_esg": np.nan,
                "se_esg": np.nan,
                "t_esg": np.nan,
                "p_esg": np.nan,
                "r2": np.nan,
            }
        )

    # Estimate fixed-effects model via dummies and stock-clustered covariance.
    formula = (
        f"{outcome_col} ~ esg_score_y + size_y + book_to_market_y + leverage_y + "
        f"analyst_coverage_y + C(stock) + C(year)"
    )

    try:
        fitted = smf.ols(formula=formula, data=reg_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": reg_df["stock"]},
        )

        return pd.Series(
            {
                "outcome": outcome_col,
                "n_obs": int(len(reg_df)),
                "n_stocks": int(reg_df["stock"].nunique()),
                "coef_esg": float(fitted.params.get("esg_score_y", np.nan)),
                "se_esg": float(fitted.bse.get("esg_score_y", np.nan)),
                "t_esg": float(fitted.tvalues.get("esg_score_y", np.nan)),
                "p_esg": float(fitted.pvalues.get("esg_score_y", np.nan)),
                "r2": float(fitted.rsquared),
            }
        )
    except Exception:
        return pd.Series(
            {
                "outcome": outcome_col,
                "n_obs": int(len(reg_df)),
                "n_stocks": int(reg_df["stock"].nunique()),
                "coef_esg": np.nan,
                "se_esg": np.nan,
                "t_esg": np.nan,
                "p_esg": np.nan,
                "r2": np.nan,
            }
        )


def run_fe_regressions(panel_df: pd.DataFrame, decomp_model: str) -> pd.DataFrame:
    """Run baseline FE regressions and return a tidy coefficient table."""

    # Baseline outcomes include market and noise shares for both decomposition choices.
    outcomes = ["MktInfoShare", "noise_share_proxy"]

    # For Thesis_3, keep disaggregated information shares as separate outcomes.
    if decomp_model == "thesis3":
        outcomes.extend(["PrivateInfoShare", "PublicInfoShare"])
    elif decomp_model == "thesis2":
        outcomes.extend(["FirmInfoShare"])

    # Keep only outcomes that exist in the panel schema.
    outcomes = [col for col in outcomes if col in panel_df.columns]

    rows = []
    for outcome_col in outcomes:
        rows.append(_fit_clustered_fe_ols(panel_df, outcome_col))

    if len(rows) == 0:
        return pd.DataFrame(
            columns=["outcome", "n_obs", "n_stocks", "coef_esg", "se_esg", "t_esg", "p_esg", "r2"]
        )

    return pd.DataFrame(rows)


def _regression_table_to_markdown(reg_df: pd.DataFrame) -> str:
    """Format the regression results table as simple markdown."""

    if len(reg_df) == 0:
        return "No regression results were produced."

    # Build table header.
    lines = []
    lines.append("| outcome | n_obs | n_stocks | coef_esg | se_esg | t_esg | p_esg | r2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    # Add one row per specification with concise rounding.
    for _, row in reg_df.iterrows():
        lines.append(
            "| {outcome} | {n_obs} | {n_stocks} | {coef_esg:.4f} | {se_esg:.4f} | {t_esg:.3f} | {p_esg:.4f} | {r2:.4f} |".format(
                outcome=row["outcome"],
                n_obs=int(row["n_obs"]) if pd.notna(row["n_obs"]) else 0,
                n_stocks=int(row["n_stocks"]) if pd.notna(row["n_stocks"]) else 0,
                coef_esg=row["coef_esg"] if pd.notna(row["coef_esg"]) else np.nan,
                se_esg=row["se_esg"] if pd.notna(row["se_esg"]) else np.nan,
                t_esg=row["t_esg"] if pd.notna(row["t_esg"]) else np.nan,
                p_esg=row["p_esg"] if pd.notna(row["p_esg"]) else np.nan,
                r2=row["r2"] if pd.notna(row["r2"]) else np.nan,
            )
        )

    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run full panel assembly and save outputs."""

    # 1) Parse workbook with robust date handling.
    parsed = parse_data_workbook(
        excel_path=args.data_file,
        daily_sheet=args.daily_sheet,
        quarterly_sheet=args.quarterly_sheet,
        market_ticker=args.market_ticker,
    )

    daily = parsed["daily"].copy()
    quarterly = parsed["quarterly"].copy()

    # Optional stock cap for quick smoke tests.
    if args.max_stocks is not None:
        keep_stocks = daily["stock"].drop_duplicates().head(args.max_stocks)
        daily = daily[daily["stock"].isin(keep_stocks)].copy()
        quarterly = quarterly[quarterly["stock"].isin(keep_stocks)].copy()

    # 2) Run selected decomposition and standardize result schema.
    decomp_outputs = build_decomposition_panel(daily_df=daily, decomp_model=args.decomp)
    decomp_results = decomp_outputs["results"].copy()

    # 3) Build yearly controls from quarterly data.
    controls_yearly = aggregate_quarterly_controls_to_year(quarterly)

    # 4) Merge decomposition results with controls on (stock, year).
    merged = decomp_results.merge(controls_yearly, on=["stock", "year"], how="left")

    # 5) Apply sample filters and build diagnostics.
    filtered_panel = apply_sample_filters(merged, start_year=args.start_year, end_year=args.end_year)
    merge_diag = build_merge_diagnostics(decomp_results, filtered_panel)

    # 6) Run FE regressions on the filtered firm-year panel.
    regression_table = run_fe_regressions(filtered_panel, decomp_model=args.decomp)

    # 7) Save outputs to deterministic file names.
    filtered_panel.to_csv(OUTPUTS_DIR / f"essay1_panel_{args.decomp}.csv", index=False)
    merge_diag.to_csv(OUTPUTS_DIR / f"essay1_merge_diagnostics_{args.decomp}.csv", index=False)
    regression_table.to_csv(OUTPUTS_DIR / f"essay1_regression_results_{args.decomp}.csv", index=False)

    # Save decomposition diagnostics from adapters for transparency.
    decomp_outputs["diagnostics"].to_csv(OUTPUTS_DIR / f"essay1_decomp_diagnostics_{args.decomp}.csv", index=False)

    # Save parser summary to a 1-row CSV for quick validation.
    parser_summary = pd.DataFrame([summarize_parsed_panel(parsed["daily"], parsed["quarterly"])])
    parser_summary.to_csv(OUTPUTS_DIR / f"essay1_parser_summary_{args.decomp}.csv", index=False)

    # Save concise markdown report.
    with open(OUTPUTS_DIR / f"essay1_panel_report_{args.decomp}.md", "w", encoding="utf-8") as report:
        report.write("# Essay I Panel Build Report\n\n")
        report.write(f"- Decomposition model: `{args.decomp}`\n")
        report.write(f"- Year window: {args.start_year}-{args.end_year}\n")
        report.write(f"- Market ticker: `{args.market_ticker}`\n")
        report.write(f"- Daily rows parsed: {len(parsed['daily'])}\n")
        report.write(f"- Quarterly rows parsed: {len(parsed['quarterly'])}\n")
        report.write(f"- Final panel rows: {len(filtered_panel)}\n")
        report.write(f"- Final unique stocks: {filtered_panel['stock'].nunique() if len(filtered_panel) > 0 else 0}\n\n")

        if args.decomp == "thesis3":
            report.write("## Decomposition outcome notes\n")
            report.write("- `PrivateInfoShare` and `PublicInfoShare` are kept separate.\n")
            report.write("- No aggregate information-share metric is constructed.\n")

        report.write("\n## Fixed-effects regressions\n")
        report.write("- Specification: firm and year fixed effects with stock-clustered standard errors.\n")
        report.write("- Core regressor: `esg_score_y`. Controls: size, book-to-market, leverage, analyst coverage.\n\n")
        report.write("- Reported outcomes include `MktInfoShare` and `noise_share_proxy` for all models.\n")
        if args.decomp == "thesis3":
            report.write("- For `thesis3`, `PrivateInfoShare` and `PublicInfoShare` are also estimated separately.\n\n")
        elif args.decomp == "thesis2":
            report.write("- For `thesis2`, `FirmInfoShare` is also estimated separately.\n\n")
        else:
            report.write("\n")
        report.write(_regression_table_to_markdown(regression_table))
        report.write("\n")

    # Print short execution summary.
    print("Pipeline complete.")
    print(f"Saved panel: {OUTPUTS_DIR / f'essay1_panel_{args.decomp}.csv'}")
    print(f"Saved merge diagnostics: {OUTPUTS_DIR / f'essay1_merge_diagnostics_{args.decomp}.csv'}")
    print(f"Saved regression results: {OUTPUTS_DIR / f'essay1_regression_results_{args.decomp}.csv'}")


def parse_args() -> argparse.Namespace:
    """Parse command line options."""

    parser = argparse.ArgumentParser(description="Build Essay I merged panel from daily and quarterly workbook sheets.")
    parser.add_argument("--decomp", choices=["thesis3", "thesis2"], default="thesis3", help="Decomposition model to use.")
    parser.add_argument("--data-file", default=str(DATA_DIR / "data.xlsx"), help="Path to workbook with daily_ and quarterly_ sheets.")
    parser.add_argument("--daily-sheet", default="daily_", help="Daily sheet name.")
    parser.add_argument("--quarterly-sheet", default="quarterly_", help="Quarterly sheet name.")
    parser.add_argument("--market-ticker", default="SXXP Index", help="Market index ticker in daily price columns.")
    parser.add_argument("--start-year", type=int, default=2015, help="Sample start year.")
    parser.add_argument("--end-year", type=int, default=2025, help="Sample end year.")
    parser.add_argument("--max-stocks", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
