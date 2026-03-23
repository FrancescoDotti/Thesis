"""
Variance Decomposition Script
Based on the methodology in "What Moves Stock Prices"

This script decomposes the variance of stock returns into three components:
1. Market Information: Variance driven by market-wide shocks
2. Firm-Specific Information: Variance driven by firm-specific shocks
3. Noise: Residual unexplained variance

The methodology uses a VAR model and long-run multipliers to identify
structural shocks and their contribution to price movements.
"""

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.api import VAR

warnings.filterwarnings('ignore')


# Resolve paths from the Thesis project root instead of the shell working directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'Data'
OUTPUTS_DIR = PROJECT_DIR / 'Outputs' / 'thesis_2_outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(data_path="./Data/BigSmall_NYA.xlsx", use_processed=False):
    """
    Load stock data from Excel file or processed pickle file.
    
    Parameters
    ----------
    data_path : str
        Path to BigSmall_NYA.xlsx file
    use_processed : bool
        If True, load from processed_data.pkl instead
        
    Returns
    -------
    dict with 'prices', 'volume', 'market_ret', 'index'
    """
    
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = DATA_DIR / data_path.name

    if use_processed:
        processed_path = DATA_DIR / 'processed_data.pkl'
        with open(processed_path, 'rb') as f:
            return pickle.load(f)
    
    # Read the Excel file
    raw = pd.read_excel(data_path, sheet_name="BigCap")
    index_name = 'NYA Index'
    
    # Parse dates and set as index
    raw1 = raw.rename(columns={'Unnamed: 0': 'Date'})
    dates = pd.to_datetime(raw1['Date'].iloc[1:])
    
    raw1 = raw1.iloc[1:].copy()
    raw1['Date'] = dates
    raw1 = raw1.set_index('Date')
    
    # Get second header row for field names (PRICE, VOLUME)
    second_header = raw.iloc[0]
    
    data = raw1.copy()
    
    # Build MultiIndex for columns: (ticker, field)
    tuples = []
    for col in data.columns:
        field = second_header[col]
        ticker = col
        if col.startswith('Unnamed:'):
            idx = data.columns.get_loc(col)
            ticker = data.columns[idx - 1]
        tuples.append((ticker, field))
    
    multi_cols = pd.MultiIndex.from_tuples(tuples, names=['Ticker', 'Field'])
    data.columns = multi_cols
    
    # Extract PRICE and VOLUME data
    prices = data.xs('PRICE', axis=1, level='Field')
    volume = data.xs('VOLUME', axis=1, level='Field')
    market = prices[[index_name]].copy()
    
    # Apply date range filter
    start_date = '1997-01-01'
    end_date = '2015-12-31'
    date_mask = (prices.index >= start_date) & (prices.index <= end_date)
    
    prices = prices.loc[date_mask]
    volume = volume.loc[date_mask]
    market = market.loc[date_mask]
    
    # Handle missing values
    prices = prices.ffill(limit=5)
    volume = volume.ffill(limit=5)
    market = market.ffill(limit=5)
    
    # Calculate market returns
    market_ret = market[index_name].pct_change(fill_method=None)
    
    return {
        'prices': prices,
        'volume': volume,
        'market_ret': market_ret,
        'index': index_name
    }


def prepare_var_data(prices, market_ret, window_days=252, min_obs=100):
    """
    Prepare data for VAR estimation: convert to daily returns within windows.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Stock prices with dates as index
    market_ret : pd.Series
        Market returns with dates as index
    window_days : int
        Number of trading days per window (default: 252 = 1 year)
    min_obs : int
        Minimum number of observations required
        
    Returns
    -------
    dict with windows of VAR data
    """
    
    # Calculate returns
    stock_ret = prices.pct_change()
    
    # Align dates
    common_idx = stock_ret.index.intersection(market_ret.index)
    stock_ret = stock_ret.loc[common_idx]
    market_ret_aligned = market_ret.loc[common_idx]
    
    # Create date-based windows
    windows = {}
    dates = stock_ret.index
    
    for i in range(len(dates) - window_days):
        window_start = dates[i]
        window_end = dates[i + window_days - 1]
        window_key = f"{window_start.strftime('%Y-%m-%d')}_{window_end.strftime('%Y-%m-%d')}"
        
        window_data = stock_ret.loc[window_start:window_end].copy()
        window_market = market_ret_aligned.loc[window_start:window_end].copy()
        
        if len(window_data) >= min_obs:
            windows[window_key] = {
                'stock_ret': window_data,
                'market_ret': window_market,
                'start_date': window_start,
                'end_date': window_end
            }
    
    return windows


def decompose_variance_single_period(market_ret, stock_ret, stock_name, n_lags=5):
    """
    Decompose variance of a single stock in a single period using VAR.
    
    Follows Brogaard et al. (2022) methodology with noise calculated as
    residual unexplained variance (not as a component).
    
    Parameters
    ----------
    market_ret : pd.Series
        Market returns for the period
    stock_ret : pd.Series
        Stock returns for the period
    stock_name : str
        Name/ticker of the stock
    n_lags : int
        Number of lags in VAR model
        
    Returns
    -------
    dict with variance decomposition results or None if estimation fails
    """
    
    # Prepare data for VAR
    var_data = pd.DataFrame({
        'market_ret': market_ret,
        'stock_ret': stock_ret
    }).dropna()
    
    if len(var_data) < max(20, 2 * n_lags + 5):
        return None
    
    try:
        # Estimate reduced-form VAR
        var_model = VAR(var_data)
        var_result = var_model.fit(maxlags=n_lags, trend='c')
        
    except Exception as e:
        return None
    
    # Get reduced-form residuals
    resids = var_result.resid
    e_market = resids.iloc[:, 0].values
    e_stock = resids.iloc[:, 1].values
    
    # Reduced-form residual variances
    sigma2_e_market = np.var(e_market, ddof=1)
    sigma2_e_stock = np.var(e_stock, ddof=1)
    
    if sigma2_e_market <= 0:
        return None
    
    # Identify structural parameters
    cov_stock_market = np.cov(e_stock, e_market)[0, 1]
    b10 = cov_stock_market / sigma2_e_market
    
    # Structural shock variances
    sigma2_eps_market = sigma2_e_market
    sigma2_eps_stock = sigma2_e_stock - b10**2 * sigma2_e_market
    
    if sigma2_eps_stock < 0:
        sigma2_eps_stock = 0
    
    # Calculate long-run multipliers
    params = var_result.params.values
    n_vars = 2
    
    A_sum = np.zeros((n_vars, n_vars))
    for lag in range(n_lags):
        start_row = 1 + lag * n_vars
        end_row = 1 + (lag + 1) * n_vars
        A_lag = params[start_row:end_row, :].T
        A_sum += A_lag
    
    I = np.eye(n_vars)
    try:
        LR_matrix = np.linalg.inv(I - A_sum)
    except np.linalg.LinAlgError:
        return None
    
    # Calculate eigenvalues (for diagnostics, but don't filter)
    # Note: Unlike some VAR approaches, we do NOT filter based on stationarity
    # to match Thesis_2.ipynb methodology
    eigenvalues = np.linalg.eigvals(A_sum)
    
    # Structural impact matrix
    B0_inv = np.array([[1.0, 0.0], [b10, 1.0]])
    LR_structural = LR_matrix @ B0_inv
    
    theta_market = LR_structural[1, 0]
    theta_stock = LR_structural[1, 1]
    
    # ========================================
    # Variance decomposition (Brogaard approach)
    # ========================================
    
    # Information components from structural shocks
    MktInfo = theta_market**2 * sigma2_eps_market
    FirmInfo = theta_stock**2 * sigma2_eps_stock
    
    # Noise is calculated as RESIDUAL unexplained variance (following Thesis_2.ipynb)
    # This is the key difference from component-based decomposition
    
    # Get constant term from VAR (drift)
    a0 = var_result.params.iloc[0, 1]
    
    # Structural shocks (inverted from reduced form)
    eps_market = e_market
    eps_stock = e_stock - b10 * e_market
    
    # Fitted information component
    fitted_info_return = theta_market * eps_market + theta_stock * eps_stock
    
    # Actual returns (aligned with residuals, after n_lags observations)
    actual_returns = var_data['stock_ret'].iloc[n_lags:].values
    
    # Alignment check
    if len(actual_returns) != len(fitted_info_return):
        return None
    
    # Noise = unexplained variance after fitting information model
    noise_returns = actual_returns - a0 - fitted_info_return
    Noise = np.var(noise_returns, ddof=1)
    
    if Noise < 0:
        Noise = 0
    
    # Total variance
    TotalVar = MktInfo + FirmInfo + Noise
    
    if TotalVar <= 0:
        return None
    
    return {
        'stock': stock_name,
        'n_obs': len(var_data),
        'max_eigenvalue': np.max(np.abs(eigenvalues)),
        # Structural parameters
        'b10': b10,
        'theta_market': theta_market,
        'theta_stock': theta_stock,
        # Structural shock variances
        'sigma2_eps_market': sigma2_eps_market,
        'sigma2_eps_stock': sigma2_eps_stock,
        # Variance components
        'MktInfo': MktInfo,
        'FirmInfo': FirmInfo,
        'Noise': Noise,
        'TotalVar': TotalVar,
        # Variance shares (%)
        'MktInfoShare': 100 * MktInfo / TotalVar if TotalVar > 0 else np.nan,
        'FirmInfoShare': 100 * FirmInfo / TotalVar if TotalVar > 0 else np.nan,
        'NoiseShare': 100 * Noise / TotalVar if TotalVar > 0 else np.nan,
    }


def decompose_all_stocks_period(market_ret, stock_prices, period_label, n_lags=5):
    """
    Decompose variance for all stocks in a given period.
    
    Parameters
    ----------
    market_ret : pd.Series
        Market returns for the period
    stock_prices : pd.DataFrame
        Stock prices for the period (columns = tickers)
    period_label : str
        Label for the period (e.g., '2000')
    n_lags : int
        Number of lags in VAR model
        
    Returns
    -------
    pd.DataFrame with variance decomposition for all stocks
    """
    
    # Calculate stock returns
    stock_ret = stock_prices.pct_change()
    
    # Align data
    common_idx = stock_ret.index.intersection(market_ret.index)
    stock_ret = stock_ret.loc[common_idx]
    market_ret = market_ret.loc[common_idx]
    
    results = []
    
    for ticker in stock_prices.columns:
        if ticker in stock_ret.columns and market_ret.notna().sum() > 20:
            result = decompose_variance_single_period(
                market_ret,
                stock_ret[ticker],
                ticker,
                n_lags=n_lags
            )
            
            if result is not None:
                result['period'] = period_label
                results.append(result)
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    return results_df


def winsorize_by_period(df, component_cols, bounds=(0.05, 0.95)):
    """
    Winsorize variance COMPONENTS (not shares) by period to handle outliers.
    
    This matches the Brogaard et al. (2022) methodology:
    1. Winsorize variance components (MktInfo, FirmInfo, Noise) at 5%-95%
    2. Then recalculate shares from winsorized components
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe with 'period' column
    component_cols : list
        Variance component columns to winsorize (MktInfo, FirmInfo, Noise)
    bounds : tuple
        (lower_percentile, upper_percentile)
        
    Returns
    -------
    pd.DataFrame with winsorized components and recalculated shares
    """
    
    df_winsorized = df.copy()
    
    # Step 1: Winsorize variance COMPONENTS (levels, not shares)
    for period in df_winsorized['period'].unique():
        mask = df_winsorized['period'] == period
        
        for col in component_cols:
            if col in df_winsorized.columns:
                q_low = df_winsorized.loc[mask, col].quantile(bounds[0])
                q_high = df_winsorized.loc[mask, col].quantile(bounds[1])
                df_winsorized.loc[mask, col] = df_winsorized.loc[mask, col].clip(q_low, q_high)
    
    # Step 2: Recalculate shares from winsorized components
    df_winsorized['TotalVar'] = (df_winsorized['MktInfo'] + 
                                  df_winsorized['FirmInfo'] + 
                                  df_winsorized['Noise'])
    
    df_winsorized['MktInfoShare'] = 100 * df_winsorized['MktInfo'] / df_winsorized['TotalVar']
    df_winsorized['FirmInfoShare'] = 100 * df_winsorized['FirmInfo'] / df_winsorized['TotalVar']
    df_winsorized['NoiseShare'] = 100 * df_winsorized['Noise'] / df_winsorized['TotalVar']
    
    return df_winsorized


def calculate_aggregate_shares(results_df, share_cols=None):
    """
    Calculate equal-weighted and variance-weighted average shares by period.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from variance decomposition
    share_cols : list
        Share columns to aggregate
        
    Returns
    -------
    tuple of (EW_df, VW_df) - equal-weighted and variance-weighted aggregates
    """
    
    if share_cols is None:
        share_cols = ['MktInfoShare', 'FirmInfoShare', 'NoiseShare']
    
    # Equal-weighted average
    EW = results_df.groupby('period')[share_cols].mean()
    
    # Variance-weighted average
    def vw_aggregate(group):
        clean = group.dropna(subset=share_cols + ['TotalVar'])
        clean = clean[clean['TotalVar'] > 0]
        
        if len(clean) == 0:
            return pd.Series({col: np.nan for col in share_cols})
        
        weights = clean['TotalVar'].values
        result = {}
        
        for col in share_cols:
            values = clean[col].values
            result[col] = np.average(values, weights=weights)
        
        return pd.Series(result)
    
    VW = results_df.groupby('period').apply(vw_aggregate)
    
    return EW, VW


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("VARIANCE DECOMPOSITION: What Moves Stock Prices")
    print("=" * 80)
    print("\nMethodology: Brogaard, Nguyen, Putnins & Wu (2022)")
    print("Winsorization: Components at 5%-95% (before calculating shares)")
    print("=" * 80)
    
    # ========================================
    # 1. Load data
    # ========================================
    print("\n1. Loading data...")
    # Always rebuild from the same raw input file so this script matches Thesis_3.py.
    data = load_data(data_path=DATA_DIR / 'BigSmall_NYA.xlsx', use_processed=False)
    
    prices = data['prices']
    market_ret = data['market_ret']
    index_name = data['index']
    
    print(f"   - Loaded {len(prices)} dates")
    print(f"   - {len(prices.columns)} stocks")
    print(f"   - Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # ========================================
    # 2. Decompose variance by year
    # ========================================
    print("\n2. Computing variance decomposition by year...")
    
    all_results = []
    years = prices.index.year.unique()
    
    for year in sorted(years):
        print(f"   Processing {year}...", end=" ")
        
        # Get data for this year
        year_mask = prices.index.year == year
        year_prices = prices[year_mask]
        year_market = market_ret[year_mask]
        
        # Decompose
        year_results = decompose_all_stocks_period(
            year_market,
            year_prices,
            str(year),
            n_lags=5
        )
        
        if year_results is not None:
            all_results.append(year_results)
            print(f"✓ {len(year_results)} stocks")
        else:
            print("✗ No valid results")
    
    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\n   Total stock-year observations: {len(results_df)}")
    
    # ========================================
    # 3. Winsorize variance COMPONENTS and recalculate shares
    # ========================================
    print("\n3. Winsorizing variance components at 5%-95% bounds...")
    print("   (Following Brogaard et al. 2022 methodology)")
    
    component_cols = ['MktInfo', 'FirmInfo', 'Noise']
    results_df = winsorize_by_period(results_df, component_cols)
    
    # ========================================
    # 4. Aggregate results
    # ========================================
    print("\n4. Aggregating results...")
    
    share_cols = ['MktInfoShare', 'FirmInfoShare', 'NoiseShare']
    EW, VW = calculate_aggregate_shares(results_df, share_cols)
    
    # ========================================
    # 5. Summary Statistics
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS: Variance Share Decomposition")
    print("=" * 80)
    
    print("\nVariance-Weighted Average (%):")
    print("-" * 80)
    print(VW)
    
    print("\n\nEqual-Weighted Average (%):")
    print("-" * 80)
    print(EW)
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL SAMPLE STATISTICS")
    print("=" * 80)
    
    vw_mean = VW[share_cols].mean()
    ew_mean = EW[share_cols].mean()
    
    print("\nVariance-Weighted Mean Across Full Period:")
    print(f"  Market Information:    {vw_mean['MktInfoShare']:6.2f}%")
    print(f"  Firm-Specific Info:    {vw_mean['FirmInfoShare']:6.2f}%")
    print(f"  Noise:                 {vw_mean['NoiseShare']:6.2f}%")
    
    print("\nEqual-Weighted Mean Across Full Period:")
    print(f"  Market Information:    {ew_mean['MktInfoShare']:6.2f}%")
    print(f"  Firm-Specific Info:    {ew_mean['FirmInfoShare']:6.2f}%")
    print(f"  Noise:                 {ew_mean['NoiseShare']:6.2f}%")
    
    # ========================================
    # 6. Diagnostic checks
    # ========================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CHECKS")
    print("=" * 80)
    
    print(f"\nSample size by year:")
    year_counts = results_df.groupby('period').size()
    print(year_counts)
    
    print(f"\nTotal stock-year observations: {len(results_df)}")
    print(f"Observations with valid variance shares: {results_df[share_cols].notna().all(axis=1).sum()}")
    
    print(f"\nStructural parameter statistics:")
    print(f"  b10 (contemporaneous response):")
    print(f"    Mean: {results_df['b10'].mean():8.4f}")
    print(f"    Median: {results_df['b10'].median():8.4f}")
    print(f"    Std Dev: {results_df['b10'].std():8.4f}")
    
    print(f"\n  theta_market (long-run market multiplier):")
    print(f"    Mean: {results_df['theta_market'].mean():8.4f}")
    print(f"    Median: {results_df['theta_market'].median():8.4f}")
    
    print(f"\n  theta_stock (long-run firm multiplier):")
    print(f"    Mean: {results_df['theta_stock'].mean():8.4f}")
    print(f"    Median: {results_df['theta_stock'].median():8.4f}")
    
    print(f"\nVAR stationarity (max eigenvalue):")
    print(f"  Mean: {results_df['max_eigenvalue'].mean():8.4f}")
    print(f"  Max: {results_df['max_eigenvalue'].max():8.4f}")
    print(f"  % with max_eig < 0.99: {(results_df['max_eigenvalue'] < 0.99).sum() / len(results_df) * 100:.1f}%")
    
    # ========================================
    # 7. Visualizations
    # ========================================
    print("\n5. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Time series of variance-weighted shares
    ax = axes[0, 0]
    for col, color in zip(share_cols, ['#1f77b4', '#ff7f0e', '#2ca02c']):
        ax.plot(VW.index, VW[col], marker='o', label=col.replace('Share', ''), linewidth=2, color=color)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Share (%)', fontsize=11)
    ax.set_title('Variance-Weighted Shares Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Plot 2: Time series of equal-weighted shares
    ax = axes[0, 1]
    for col, color in zip(share_cols, ['#1f77b4', '#ff7f0e', '#2ca02c']):
        ax.plot(EW.index, EW[col], marker='s', label=col.replace('Share', ''), linewidth=2, color=color)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Share (%)', fontsize=11)
    ax.set_title('Equal-Weighted Shares Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Plot 3: Cross-sectional distribution of shares (VW)
    ax = axes[1, 0]
    vw_data = VW[share_cols].values
    x = np.arange(len(vw_data))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, col in enumerate(share_cols):
        ax.bar(x + i*width, VW[col], width, label=col.replace('Share', ''), color=colors[i])
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Share (%)', fontsize=11)
    ax.set_title('Variance-Weighted Shares (Stacked View)', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(VW.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Distribution across stocks (latest year)
    ax = axes[1, 1]
    latest_year = str(results_df['period'].max())
    latest_data = results_df[results_df['period'] == latest_year][share_cols]
    
    bp = ax.boxplot([latest_data[col].dropna() for col in share_cols],
                     labels=['Market Info', 'Firm Info', 'Noise'],
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Share (%)', fontsize=11)
    ax.set_title(f'Cross-Sectional Distribution ({latest_year})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'variance_decomposition.png', dpi=300, bbox_inches='tight')
    print("   - Saved: variance_decomposition.png")
    
    # ========================================
    # 8. Save detailed results
    # ========================================
    print("\n6. Saving detailed results...")
    
    # Save all stock-year results
    results_df.to_csv(OUTPUTS_DIR / 'variance_decomposition_results.csv', index=False)
    print("   - Saved: variance_decomposition_results.csv")
    
    # Save aggregated results
    VW.to_csv(OUTPUTS_DIR / 'variance_decomposition_VW.csv')
    EW.to_csv(OUTPUTS_DIR / 'variance_decomposition_EW.csv')
    print("   - Saved: variance_decomposition_VW.csv")
    print("   - Saved: variance_decomposition_EW.csv")
    
    # Save summary statistics
    with open(OUTPUTS_DIR / 'variance_decomposition_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VARIANCE DECOMPOSITION: What Moves Stock Prices\n")
        f.write("Brogaard, Nguyen, Putnins & Wu (2022) Methodology\n")
        f.write("=" * 80 + "\n\n")
        f.write("WINSORIZATION METHOD: Variance components at 5%-95%\n")
        f.write("(Components winsorized BEFORE calculating shares, matching Brogaard et al.)\n\n")
        
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
    
    print("   - Saved: variance_decomposition_summary.txt")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return results_df, EW, VW


if __name__ == "__main__":
    results_df, EW, VW = main()
