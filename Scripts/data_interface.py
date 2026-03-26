"""
Shared data interface utilities for Essay I thesis scripts.

This module centralizes parsing logic for Bloomberg-style Excel sheets that use
2-row headers (ticker on row 1, field name on row 2).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _resolve_excel_path(excel_path: Union[str, Path]) -> Path:
    """Resolve input path to an absolute path."""

    path = Path(excel_path)
    if path.is_absolute():
        return path

    # Keep relative paths stable by resolving from this script's Thesis root.
    thesis_root = Path(__file__).resolve().parent.parent
    return thesis_root / path


def _parse_excel_date_column(date_series: pd.Series) -> pd.Series:
    """Parse date values handling both Excel serial dates and datetime strings."""

    # First attempt: normal datetime parsing for string/date-like cells.
    parsed_default = pd.to_datetime(date_series, errors="coerce")

    # Detect whether the raw column is mostly numeric Excel serial dates.
    numeric_values = pd.to_numeric(date_series, errors="coerce")
    numeric_ratio = numeric_values.notna().mean() if len(date_series) > 0 else 0.0

    # If numeric dates dominate, parse as Excel serial day counts.
    if numeric_ratio > 0.8:
        parsed_excel = pd.to_datetime(numeric_values, unit="D", origin="1899-12-30", errors="coerce")
        return parsed_excel

    return parsed_default


def read_bloomberg_two_row_sheet(
    excel_path: Union[str, Path],
    sheet_name: str,
    date_col: int = 0,
) -> pd.DataFrame:
    """
    Parse a Bloomberg-style sheet with 2-row headers into a MultiIndex DataFrame.

    The expected layout is:
    - Row 1: ticker labels (can have blanks that continue previous ticker)
    - Row 2: field labels (PRICE, VOLUME, ESG_SCORE, etc.)
    - Row 3+: data

    Returns
    -------
    pd.DataFrame
        Index is parsed datetime, columns are MultiIndex (Ticker, Field)
    """

    file_path = _resolve_excel_path(excel_path)
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    if raw.shape[0] < 3:
        raise ValueError(f"Sheet '{sheet_name}' does not have enough rows for 2-row headers.")

    # Read the first two header rows used to build (ticker, field) pairs.
    ticker_row = raw.iloc[0]
    field_row = raw.iloc[1]
    value_block = raw.iloc[2:].copy()

    # Parse the date column and keep only rows with valid dates.
    parsed_dates = _parse_excel_date_column(value_block.iloc[:, date_col])
    value_block = value_block.loc[parsed_dates.notna()].copy()
    parsed_dates = parsed_dates.loc[parsed_dates.notna()]

    # Remove the date column from values and set parsed dates as index.
    value_block = value_block.drop(columns=value_block.columns[date_col])
    value_block.index = parsed_dates
    value_block.index.name = "date"

    # Build the MultiIndex columns, forward-filling blank ticker cells.
    tuples: list[Tuple[str, str]] = []
    last_ticker: Optional[str] = None

    # Shift because we dropped the date column from the value block.
    source_cols = [col for col in raw.columns if col != raw.columns[date_col]]
    for col in source_cols:
        raw_ticker = ticker_row[col]
        raw_field = field_row[col]

        if pd.notna(raw_ticker) and str(raw_ticker).strip() != "":
            last_ticker = str(raw_ticker).strip()

        if last_ticker is None or pd.isna(raw_field):
            continue

        field_name = str(raw_field).strip()
        if field_name == "":
            continue

        tuples.append((last_ticker, field_name))

    if len(tuples) != value_block.shape[1]:
        raise ValueError(
            f"Header parse mismatch in sheet '{sheet_name}': "
            f"built {len(tuples)} columns for {value_block.shape[1]} values."
        )

    value_block.columns = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Field"])

    # Convert each column to numeric when possible and keep original values otherwise.
    for col in value_block.columns:
        try:
            value_block[col] = pd.to_numeric(value_block[col])
        except Exception:
            pass

    return value_block.sort_index()


def split_price_volume(
    data: pd.DataFrame,
    price_field_candidates: Optional[Tuple[str, ...]] = None,
    volume_field_candidates: Optional[Tuple[str, ...]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Split parsed MultiIndex daily data into PRICE and VOLUME wide frames.

    Returns a dict with keys: 'prices', 'volume'.
    """

    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns with levels (Ticker, Field).")

    if price_field_candidates is None:
        price_field_candidates = ("PRICE", "PX_LAST")
    if volume_field_candidates is None:
        volume_field_candidates = ("VOLUME",)

    fields = set(data.columns.get_level_values("Field"))

    price_field = next((field for field in price_field_candidates if field in fields), None)
    volume_field = next((field for field in volume_field_candidates if field in fields), None)

    if price_field is None or volume_field is None:
        raise ValueError(
            "Daily sheet must contain price and volume fields. "
            f"Found fields: {sorted(fields)}"
        )

    prices = data.xs(price_field, axis=1, level="Field")
    volume = data.xs(volume_field, axis=1, level="Field")

    # Remove duplicate ticker columns if any exist in the source.
    prices = prices.loc[:, ~prices.columns.duplicated()].copy()
    volume = volume.loc[:, ~volume.columns.duplicated()].copy()

    return {"prices": prices, "volume": volume}


def build_daily_canonical_panel(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    market_ticker: str,
) -> pd.DataFrame:
    """
    Convert wide PRICE/VOLUME data into canonical daily long format.

    Output columns:
    - stock, date, year
    - price, volume
    - stock_ret, market_ret
    """

    if market_ticker not in prices.columns:
        raise ValueError(f"Market ticker '{market_ticker}' not found in PRICE columns.")

    market_price = prices[market_ticker].copy()
    market_ret = market_price.pct_change(fill_method=None)

    rows = []
    stock_tickers = [ticker for ticker in prices.columns if ticker != market_ticker]

    for ticker in stock_tickers:
        stock_price = prices[ticker]
        stock_volume = volume[ticker] if ticker in volume.columns else np.nan
        stock_ret = stock_price.pct_change(fill_method=None)

        ticker_df = pd.DataFrame(
            {
                "stock": ticker,
                "date": prices.index,
                "price": stock_price.values,
                "volume": stock_volume.values if isinstance(stock_volume, pd.Series) else np.nan,
                "stock_ret": stock_ret.values,
                "market_ret": market_ret.values,
            }
        )
        rows.append(ticker_df)

    daily = pd.concat(rows, ignore_index=True)
    daily["year"] = pd.to_datetime(daily["date"]).dt.year

    # Keep only rows with core values needed by decomposition routines.
    daily = daily.dropna(subset=["date", "stock", "stock_ret", "market_ret"])
    return daily.sort_values(["stock", "date"]).reset_index(drop=True)


def build_quarterly_canonical_panel(quarterly_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert parsed quarterly MultiIndex data into canonical long format.

    Output columns:
    - stock, date, year
    - one column per quarterly field (e.g., ESG_SCORE, CUR_MKT_CAP)
    """

    if not isinstance(quarterly_data.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns with levels (Ticker, Field).")

    rows = []
    tickers = quarterly_data.columns.get_level_values("Ticker").unique()

    for ticker in tickers:
        one_ticker = quarterly_data[ticker].copy()
        one_ticker = one_ticker.reset_index().rename(columns={"date": "date"})
        one_ticker["stock"] = ticker
        rows.append(one_ticker)

    quarterly = pd.concat(rows, ignore_index=True)
    quarterly["year"] = pd.to_datetime(quarterly["date"]).dt.year

    # Put canonical ID columns first.
    id_cols = ["stock", "date", "year"]
    other_cols = [col for col in quarterly.columns if col not in id_cols]
    quarterly = quarterly[id_cols + other_cols]

    return quarterly.sort_values(["stock", "date"]).reset_index(drop=True)


def parse_data_workbook(
    excel_path: Union[str, Path],
    daily_sheet: str = "daily_",
    quarterly_sheet: str = "quarterly_",
    market_ticker: str = "SXXP Index",
    price_field_candidates: Optional[Tuple[str, ...]] = None,
    volume_field_candidates: Optional[Tuple[str, ...]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Parse daily and quarterly sheets from the thesis workbook.

    Returns
    -------
    dict with keys:
    - daily_raw: MultiIndex daily table
    - daily: canonical daily panel
    - quarterly_raw: MultiIndex quarterly table
    - quarterly: canonical quarterly panel
    """

    daily_raw = read_bloomberg_two_row_sheet(excel_path, sheet_name=daily_sheet)
    daily_parts = split_price_volume(
        daily_raw,
        price_field_candidates=price_field_candidates,
        volume_field_candidates=volume_field_candidates,
    )
    daily = build_daily_canonical_panel(
        prices=daily_parts["prices"],
        volume=daily_parts["volume"],
        market_ticker=market_ticker,
    )

    quarterly_raw = read_bloomberg_two_row_sheet(excel_path, sheet_name=quarterly_sheet)
    quarterly = build_quarterly_canonical_panel(quarterly_raw)

    return {
        "daily_raw": daily_raw,
        "daily": daily,
        "quarterly_raw": quarterly_raw,
        "quarterly": quarterly,
    }


def summarize_parsed_panel(daily: pd.DataFrame, quarterly: pd.DataFrame) -> Dict[str, object]:
    """Return parser diagnostics for logging and validation."""

    daily_summary = {
        "daily_rows": int(len(daily)),
        "daily_stocks": int(daily["stock"].nunique()) if "stock" in daily else 0,
        "daily_date_min": pd.to_datetime(daily["date"]).min() if len(daily) > 0 else pd.NaT,
        "daily_date_max": pd.to_datetime(daily["date"]).max() if len(daily) > 0 else pd.NaT,
    }

    quarterly_summary = {
        "quarterly_rows": int(len(quarterly)),
        "quarterly_stocks": int(quarterly["stock"].nunique()) if "stock" in quarterly else 0,
        "quarterly_date_min": pd.to_datetime(quarterly["date"]).min() if len(quarterly) > 0 else pd.NaT,
        "quarterly_date_max": pd.to_datetime(quarterly["date"]).max() if len(quarterly) > 0 else pd.NaT,
    }

    return {**daily_summary, **quarterly_summary}
