#!/usr/bin/env python
# coding: utf-8

# Summary of the five-step procedure from Appendix A of "*What Moves Stock Prices? The Roles of News, Noise, and Information*":
# 
# 1.   Estimate the reduced-form VAR in Equation (A1), saving the residuals and variance/covariance matrix of residuals;
# 
# 2.   Estimate the parameters $b_{1,0}$, $c_{1,0}$, and $c_{2,0}$ from regressions of the reduced-form residuals (second and third equations in [A2]);
# 
# 3.   Estimate the variances of the structural innovations using Equation (A3);
#     
# 4.   Estimate the long-run (permanent) cumulative return responses to unit shocks of the structural-model innovations, $\theta_{rm}$, $\theta_x$, and $\theta_r$, using reduced-form-model impulse response functions with the shocks given in Equations (A4); and
#     
# 5.   Combine the estimated variances of the structural innovations from step (iii) with the long-run return responses from step (iv) to get the variance components and variance shares following Equations (9) and (10) in the paper.

# In[ ]:


from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Resolve paths from the Thesis project root instead of the shell working directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
OUTPUTS_DIR = PROJECT_DIR / "Outputs" / "thesis_3_outputs"
WINSOR_BOUNDS = (0.05, 0.95)
MIN_VALID_OBS = 20
RETURN_SCALE = 10000

# Make sure the output folder exists before any figures are written to disk.
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# In[ ]:


# Always read the same raw input file used by Thesis_2.py.
file_path = DATA_DIR / "BigSmall_NYA.xlsx"
raw = pd.read_excel(file_path, sheet_name="BigCap")
index = 'NYA Index'

# Parse dates and set them as the index before processing the table.
raw1 = raw.rename(columns={'Unnamed: 0': 'Date'})
dates = pd.to_datetime(raw1['Date'].iloc[1:])

raw1 = raw1.iloc[1:].copy()
raw1['Date'] = dates
raw1 = raw1.set_index('Date')

# Read the second header row so each column can be tagged as PRICE or VOLUME.
second_header = raw.iloc[0]
data = raw1.copy()

# Build a MultiIndex for columns: (ticker, field).
arrays = []
for col in data.columns:
    field = second_header[col]
    ticker = col
    if col.startswith('Unnamed:'):
        idx = data.columns.get_loc(col)
        ticker = data.columns[idx - 1]
    arrays.append((ticker, field))

tuples = [(ticker, field) for ticker, field in arrays]
multi_cols = pd.MultiIndex.from_tuples(tuples, names=['Ticker', 'Field'])
data.columns = multi_cols

# Split the table into prices and volumes.
prices = data.xs('PRICE', axis=1, level='Field')
volume = data.xs('VOLUME', axis=1, level='Field')
market = prices[[index]].copy()

# Keep the same sample window as the script version.
start_date = '1997-01-01'
end_date = '2015-12-31'
date_mask = (prices.index >= start_date) & (prices.index <= end_date)
prices = prices.loc[date_mask]
volume = volume.loc[date_mask]
market = market.loc[date_mask]

# Fill short gaps so returns and volume can be computed consistently.
prices = prices.ffill(limit=5)
volume = volume.ffill(limit=5)
market = market.ffill(limit=5)

# Compute the market return series used later in the notebook.
market_ret = market[index].pct_change(fill_method=None)

print(f"Data loaded from {file_path}")


# In[4]:


# Calculate stock returns for all tickers (excluding market index)
stock_tickers = [col for col in prices.columns if col != index]
returns = prices[stock_tickers].pct_change(fill_method=None)

# Build long-format DataFrame
df_list = []

for ticker in stock_tickers:
    # Get data for this ticker
    ticker_df = pd.DataFrame({
        'date': prices.index,
        'ticker': ticker,
        'price': prices[ticker].values,
        'volume': volume[ticker].values,
        'r': returns[ticker].values * RETURN_SCALE,   # Convert to basis points
        'rm': market_ret.values * RETURN_SCALE        # Convert to basis points
    })

    df_list.append(ticker_df)

# Concatenate all stocks
df = pd.concat(df_list, ignore_index=True)

# Add year column
df['year'] = df['date'].dt.year

# Remove missing values
df = df.dropna(subset=['r', 'rm', 'price', 'volume'])

# Filter: minimum valid daily observations per stock-year
df = df.groupby(['ticker', 'year']).filter(lambda x: len(x) >= MIN_VALID_OBS)

# Keep only necessary columns in correct order
df = df[['ticker', 'year', 'date', 'rm', 'r', 'volume', 'price']]

print(f"Final dataset shape: {df.shape}")
print(f"Stocks: {df['ticker'].nunique()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")
print(f"\nFirst few rows:\n{df.head()}")


# In[5]:


def decompose_variance_single_stock(df_stock, market_ret_col='rm',
                                   stock_ret_col='r', volume_col='volume',
                                   price_col='price', n_lags=5):
    """
    Decompose variance for a single stock-year following Brogaard et al. (2022).

    Returns dict with variance components and shares, or None if estimation fails.
    """

    # ========================================
    # 1. Prepare data
    # ========================================
    df = df_stock.copy()

    # Calculate signed dollar volume: sign(return) * price * volume
    df['x'] = np.sign(df[stock_ret_col]) * df[price_col] * df[volume_col]

    # Select variables in correct order: [rm, x, r]
    var_data = df[[market_ret_col, 'x', stock_ret_col]].dropna()

    # Winsorize raw VAR input series before estimation for extra robustness.
    # This clips extreme daily observations in market return, signed volume, and stock return.
    for col in [market_ret_col, 'x', stock_ret_col]:
        q_low = var_data[col].quantile(WINSOR_BOUNDS[0])
        q_high = var_data[col].quantile(WINSOR_BOUNDS[1])
        var_data[col] = var_data[col].clip(lower=q_low, upper=q_high)

    if len(var_data) < MIN_VALID_OBS:  # Minimum observations check
        return None

    # ========================================
    # 2. Estimate reduced-form VAR
    # ========================================
    try:
        var_model = VAR(var_data)
        var_result = var_model.fit(maxlags=n_lags, trend='c')
    except:
        return None

    # Track the lag count actually used by the fitted VAR.
    used_lags = var_result.k_ar

    # Get reduced-form residuals
    resids = var_result.resid
    e_rm = resids.iloc[:, 0]
    e_x = resids.iloc[:, 1]
    e_r = resids.iloc[:, 2]

    # Reduced-form residual variances
    sigma2_erm = np.var(e_rm, ddof=1)
    sigma2_ex = np.var(e_x, ddof=1)
    sigma2_er = np.var(e_r, ddof=1)

    # Market residual variance must be positive for structural identification.
    if sigma2_erm <= 0:
        return None

    # ========================================
    # 3. Identify structural parameters
    # ========================================
    # From Equation (A2): e_x = b1,0 * e_rm + eps_x
    b10 = np.cov(e_x, e_rm)[0,1] / np.var(e_rm, ddof=1)

    # From Equation (A2): e_r = c1,0 * e_rm + c2,0 * e_x + eps_r
    X_mat = np.column_stack([e_rm, e_x])
    coeffs = np.linalg.lstsq(X_mat, e_r, rcond=None)[0]
    c10, c20 = coeffs

    # ========================================
    # 4. Calculate structural shock variances (Equation A3)
    # ========================================
    sigma2_eps_rm = sigma2_erm
    sigma2_eps_x = sigma2_ex - b10**2 * sigma2_erm
    sigma2_eps_r = (sigma2_er
                    - (c10**2 + 2*c10*c20*b10) * sigma2_erm  # Fixed: added cross-term
                    - c20**2 * sigma2_ex)

    # Ensure non-negative variances
    if sigma2_eps_x < 0 or sigma2_eps_r < 0:
        return None

    # Build reduced-form coefficient sum for stationarity diagnostics.
    params = var_result.params.values
    n_vars = 3
    A_sum = np.zeros((n_vars, n_vars))
    for lag in range(used_lags):
        start_row = 1 + lag * n_vars
        end_row = 1 + (lag + 1) * n_vars
        A_lag = params[start_row:end_row, :].T
        A_sum += A_lag

    eigenvalues = np.linalg.eigvals(A_sum)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # Filter out non-stationary/explosive VAR systems.
    if max_eigenvalue >= 1.0:
        return None

    # ========================================
    # 5. Calculate long-run multipliers (closed-form BN representation)
    # ========================================
    # Compute the analytically exact long-run multiplier matrix:
    # (I - A1 - A2 - ... - Ap)^(-1)
    I = np.eye(n_vars)
    try:
        LR_matrix = np.linalg.inv(I - A_sum)
    except np.linalg.LinAlgError:
        return None

    # Build the structural impact matrix from Appendix A residual equations.
    # [e_rm, e_x, e_r]' = B0_inv * [eps_rm, eps_x, eps_r]'
    B0_inv = np.array([
        [1.0, 0.0, 0.0],
        [b10, 1.0, 0.0],
        [c10 + c20 * b10, c20, 1.0],
    ])

    # Map reduced-form long-run effects into structural long-run effects.
    LR_structural = LR_matrix @ B0_inv

    # Long-run return multipliers for each structural shock.
    theta_rm = LR_structural[2, 0]
    theta_x = LR_structural[2, 1]
    theta_r = LR_structural[2, 2]

    # ========================================
    # 6. Calculate variance components (Equation 9)
    # ========================================
    MktInfo = theta_rm**2 * sigma2_eps_rm
    PrivateInfo = theta_x**2 * sigma2_eps_x
    PublicInfo = theta_r**2 * sigma2_eps_r

    # Information variance
    sigma2_w = MktInfo + PrivateInfo + PublicInfo

    # Calculate structural innovations from reduced-form residuals.
    eps_rm = e_rm.values
    eps_x = e_x.values - b10 * e_rm.values
    eps_r = e_r.values - c10 * e_rm.values - c20 * e_x.values

    # Calculate noise from structural information model fit.
    a0 = var_result.params.iloc[0, 2]  # Constant from return equation (drift)
    fitted_info_return = theta_rm * eps_rm + theta_x * eps_x + theta_r * eps_r

    actual_returns = var_data[stock_ret_col].iloc[used_lags:].values
    if len(actual_returns) != len(fitted_info_return):
        return None

    noise_returns = actual_returns - a0 - fitted_info_return
    sigma2_s = np.var(noise_returns, ddof=1)

    # Ensure non-negative noise variance
    if sigma2_s < 0:
        sigma2_s = 0

    return {
        'MktInfo': MktInfo,
        'PrivateInfo': PrivateInfo,
        'PublicInfo': PublicInfo,
        'Noise': sigma2_s,
        'k_ar': used_lags,
        'max_eigenvalue': max_eigenvalue,
        'sigma2_w': sigma2_w,
        'sigma2_s': sigma2_s,
        'theta_rm': theta_rm,
        'theta_x': theta_x,
        'theta_r': theta_r,
        'b10': b10,
        'c10': c10,
        'c20': c20
    }


def winsorize_by_period(df, component_cols, period_col='year', bounds=(0.05, 0.95)):
    """
    Winsorize variance components by period and keep Appendix A share logic.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    component_cols : list
        Columns with variance component levels
    period_col : str
        Period column (year)
    bounds : tuple
        Lower/upper winsorization quantiles

    Returns
    -------
    pd.DataFrame with winsorized component levels
    """

    df_winsorized = df.copy()

    for period_value in df_winsorized[period_col].unique():
        period_mask = df_winsorized[period_col] == period_value
        for col in component_cols:
            q_low = df_winsorized.loc[period_mask, col].quantile(bounds[0])
            q_high = df_winsorized.loc[period_mask, col].quantile(bounds[1])
            df_winsorized.loc[period_mask, col] = df_winsorized.loc[period_mask, col].clip(lower=q_low, upper=q_high)

    return df_winsorized


def build_yearly_diagnostics(results_df, share_cols, period_col='year', total_col='VarTotal'):
    """
    Build a harmonized yearly diagnostics table used by both Thesis_2 and Thesis_3 outputs.

    Parameters
    ----------
    results_df : pd.DataFrame
        Stock-year decomposition results
    share_cols : list
        Share column names
    period_col : str
        Year column name
    total_col : str
        Total variance column name

    Returns
    -------
    pd.DataFrame with diagnostics by year
    """

    diagnostics = []

    for period_value, group in results_df.groupby(period_col):
        valid_share_mask = group[share_cols].notna().all(axis=1)
        positive_total_mask = group[total_col] > 0
        valid_mask = valid_share_mask & positive_total_mask

        share_sum = group.loc[valid_mask, share_cols].sum(axis=1)

        row = {
            'period': str(period_value),
            'n_stock_years': len(group),
            'n_unique_stocks': group['stock'].nunique() if 'stock' in group.columns else np.nan,
            'n_valid_shares': int(valid_mask.sum()),
            'pct_valid_shares': 100 * valid_mask.mean() if len(group) > 0 else np.nan,
            'total_var_mean': group[total_col].mean(),
            'total_var_median': group[total_col].median(),
            'share_sum_mean': share_sum.mean() if len(share_sum) > 0 else np.nan,
            'share_sum_std': share_sum.std(ddof=1) if len(share_sum) > 1 else np.nan,
            'max_eigenvalue_mean': group['max_eigenvalue'].mean() if 'max_eigenvalue' in group.columns else np.nan,
            'max_eigenvalue_max': group['max_eigenvalue'].max() if 'max_eigenvalue' in group.columns else np.nan,
            'pct_stationary_lt_0_99': (
                100 * (group['max_eigenvalue'] < 0.99).mean()
                if 'max_eigenvalue' in group.columns and len(group) > 0
                else np.nan
            ),
            'k_ar_mean': group['k_ar'].mean() if 'k_ar' in group.columns else np.nan,
        }
        diagnostics.append(row)

    diagnostics_df = pd.DataFrame(diagnostics).sort_values('period')
    return diagnostics_df


# In[6]:


def process_stock_year_data(df, stock_col='ticker', year_col='year',
                            market_ret_col='rm', stock_ret_col='r',
                            volume_col='volume', price_col='price'):
    """
    Process all stock-years and calculate variance decomposition.

    Returns DataFrame with variance components and shares.
    """

    results = []

    for (stock, year), group in df.groupby([stock_col, year_col]):
        res = decompose_variance_single_stock(
            group, market_ret_col, stock_ret_col, volume_col, price_col
        )

        if res is not None:
            res['stock'] = stock
            res['year'] = year
            res['n_obs'] = len(group)
            results.append(res)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return results_df

    # ========================================
    # Winsorize variance components by year (5% and 95%)
    # ========================================
    component_cols = ['MktInfo', 'PrivateInfo', 'PublicInfo', 'Noise']
    results_df = winsorize_by_period(
        results_df,
        component_cols=component_cols,
        period_col='year',
        bounds=WINSOR_BOUNDS,
    )

    # ========================================
    # Calculate variance shares (Equation 10)
    # ========================================
    results_df['VarTotal'] = (results_df['MktInfo'] + results_df['PrivateInfo'] +
                              results_df['PublicInfo'] + results_df['Noise'])

    results_df['MktInfoShare'] = 100 * results_df['MktInfo'] / results_df['VarTotal']
    results_df['PrivateInfoShare'] = 100 * results_df['PrivateInfo'] / results_df['VarTotal']
    results_df['PublicInfoShare'] = 100 * results_df['PublicInfo'] / results_df['VarTotal']
    results_df['NoiseShare'] = 100 * results_df['Noise'] / results_df['VarTotal']

    return results_df


def run_thesis3_from_daily_panel(daily_df):
    """
    Run Thesis_3 decomposition from a canonical daily panel.

    Expected input columns
    ----------------------
    daily_df : pd.DataFrame
        Must include: stock, date, price, volume, stock_ret, market_ret

    Returns
    -------
    dict
        {
            'results': stock-year decomposition table,
            'ew': equal-weighted yearly shares,
            'vw': variance-weighted yearly shares,
            'diagnostics': yearly diagnostics table
        }
    """

    required_cols = {'stock', 'date', 'price', 'volume', 'stock_ret', 'market_ret'}
    missing_cols = required_cols - set(daily_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for Thesis_3 adapter: {sorted(missing_cols)}")

    # Map canonical names to the function's expected names.
    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df.rename(
        columns={
            'stock': 'ticker',
            'stock_ret': 'r',
            'market_ret': 'rm',
        }
    )

    # Keep only rows with all variables needed by the 4-way model.
    df = df.dropna(subset=['ticker', 'year', 'date', 'rm', 'r', 'volume', 'price'])

    results_df = process_stock_year_data(
        df,
        stock_col='ticker',
        year_col='year',
        market_ret_col='rm',
        stock_ret_col='r',
        volume_col='volume',
        price_col='price',
    )

    if len(results_df) == 0:
        return {
            'results': results_df,
            'ew': pd.DataFrame(),
            'vw': pd.DataFrame(),
            'diagnostics': pd.DataFrame(),
        }

    share_cols = ['MktInfoShare', 'PrivateInfoShare', 'PublicInfoShare', 'NoiseShare']
    ew_df, vw_df = aggregate_variance_shares_fixed(results_df, share_cols=share_cols)
    diagnostics_df = build_yearly_diagnostics(results_df, share_cols, period_col='year', total_col='VarTotal')

    return {
        'results': results_df,
        'ew': ew_df,
        'vw': vw_df,
        'diagnostics': diagnostics_df,
    }


def aggregate_variance_shares_fixed(results_df, share_cols=None):
    """
    Calculate equal-weighted and variance-weighted yearly averages.
    """
    if share_cols is None:
        share_cols = ['MktInfoShare', 'PrivateInfoShare', 'PublicInfoShare', 'NoiseShare']

    # Equal-weighted
    EW = results_df.groupby('year')[share_cols].mean()

    # Variance-weighted - robust version
    def vw_avg(group):
        # Remove rows with NaN in shares or weights
        clean_group = group.dropna(subset=share_cols + ['VarTotal'])

        # Remove rows with zero or negative weights
        clean_group = clean_group[clean_group['VarTotal'] > 0]

        if len(clean_group) == 0:
            return pd.Series({col: np.nan for col in share_cols})

        weights = clean_group['VarTotal'].values
        result = {}

        for col in share_cols:
            values = clean_group[col].values
            result[col] = np.average(values, weights=weights)

        return pd.Series(result)

    VW = results_df.groupby('year').apply(vw_avg)

    return EW, VW

# Assuming your data is in df with columns: ticker, year, date, rm, r, volume, price
# where rm = market return, r = stock return

results_df = process_stock_year_data(df)

# Keep the 4-way decomposition disaggregated (Private vs Public Info separate)
results_df['VarTotal'] = (results_df['MktInfo'] + results_df['PrivateInfo'] +
                          results_df['PublicInfo'] + results_df['Noise'])
results_df['MktInfoShare'] = 100 * results_df['MktInfo'] / results_df['VarTotal']
results_df['PrivateInfoShare'] = 100 * results_df['PrivateInfo'] / results_df['VarTotal']
results_df['PublicInfoShare'] = 100 * results_df['PublicInfo'] / results_df['VarTotal']
results_df['NoiseShare'] = 100 * results_df['Noise'] / results_df['VarTotal']

# Recalculate with all 4 components
EW, VW = aggregate_variance_shares_fixed(
    results_df,
    share_cols=['MktInfoShare', 'PrivateInfoShare', 'PublicInfoShare', 'NoiseShare'],
)

# Define share_cols globally for the plot and print statements
share_cols = ['MktInfoShare', 'PrivateInfoShare', 'PublicInfoShare', 'NoiseShare']

# Check for issues
print("VW head:\n", VW.head(10))
print("\nVW tail:\n", VW.tail(10))
print("\nVW info:")
print(VW.info())

# Check source data quality
print("\n=== Data Quality Checks ===")
print(f"Total stock-years: {len(results_df)}")
print(f"Stock-years with NaN in shares: {results_df[share_cols].isna().any(axis=1).sum()}")
print(f"Stock-years with NaN in VarTotal: {results_df['VarTotal'].isna().sum()}")
print(f"Stock-years with VarTotal <= 0: {(results_df['VarTotal'] <= 0).sum()}")

# Build harmonized diagnostics so yearly paths can be compared directly with Thesis_2 outputs.
diagnostics_df = build_yearly_diagnostics(results_df, share_cols, period_col='year', total_col='VarTotal')
print("\nHarmonized yearly diagnostics (first 10 rows):")
print(diagnostics_df.head(10))

print("\nStocks per year:")
print(results_df.groupby('year').size())

print("\nSample of VarTotal by year:")
print(results_df.groupby('year')['VarTotal'].agg(['count', 'mean', 'min', 'max']))

print("\nVAR stationarity (max eigenvalue):")
print(f"  Mean: {results_df['max_eigenvalue'].mean():8.4f}")
print(f"  Max: {results_df['max_eigenvalue'].max():8.4f}")
print(f"  % with max_eig < 0.99: {(results_df['max_eigenvalue'] < 0.99).sum() / len(results_df) * 100:.1f}%")


# In[7]:


# Plotting code - 4-way decomposition visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Define colors for 4 components
colors_4way = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot 1: Time series of variance-weighted shares
ax = axes[0, 0]
for col, color in zip(share_cols, colors_4way):
    label_text = col.replace('Share', '').replace('MktInfo', 'Market').replace('PrivateInfo', 'Private').replace('PublicInfo', 'Public')
    ax.plot(VW.index, VW[col], marker='o', label=label_text, linewidth=2, color=color)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Share (%)', fontsize=11)
ax.set_title('Variance-Weighted Shares Over Time (4-way)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

# Plot 2: Time series of equal-weighted shares
ax = axes[0, 1]
for col, color in zip(share_cols, colors_4way):
    label_text = col.replace('Share', '').replace('MktInfo', 'Market').replace('PrivateInfo', 'Private').replace('PublicInfo', 'Public')
    ax.plot(EW.index, EW[col], marker='s', label=label_text, linewidth=2, color=color)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Share (%)', fontsize=11)
ax.set_title('Equal-Weighted Shares Over Time (4-way)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

# Plot 3: Stacked bar chart (VW) showing 4 components
ax = axes[1, 0]
x = np.arange(len(VW))
width = 0.8
bottom = np.zeros(len(VW))

for col, color in zip(share_cols, colors_4way):
    label_text = col.replace('Share', '').replace('MktInfo', 'Market').replace('PrivateInfo', 'Private').replace('PublicInfo', 'Public')
    ax.bar(x, VW[col], width, bottom=bottom, label=label_text, color=color)
    bottom += VW[col].values

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Share (%)', fontsize=11)
ax.set_title('Variance-Weighted Shares (Stacked, 4-way)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(VW.index, rotation=45, fontsize=9)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 100])

# Plot 4: Distribution across stocks (latest year) - 4 components
ax = axes[1, 1]
latest_year = int(results_df['year'].max())
latest_data = results_df[results_df['year'] == latest_year][share_cols]

labels_short = ['Market', 'Private', 'Public', 'Noise']
bp = ax.boxplot(
    [latest_data[col].dropna() for col in share_cols],
    labels=labels_short,
    patch_artist=True,
)

for patch, color in zip(bp['boxes'], colors_4way):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Share (%)', fontsize=11)
ax.set_title(f'Cross-Sectional Distribution ({latest_year})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'variance_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print("   - Saved: variance_decomposition.png")

# Save detailed results
print("\n6. Saving detailed results...")
results_df.to_csv(OUTPUTS_DIR / 'variance_decomposition_results.csv', index=False)
print("   - Saved: variance_decomposition_results.csv")

# Save aggregated results
VW.to_csv(OUTPUTS_DIR / 'variance_decomposition_VW.csv')
EW.to_csv(OUTPUTS_DIR / 'variance_decomposition_EW.csv')
print("   - Saved: variance_decomposition_VW.csv")
print("   - Saved: variance_decomposition_EW.csv")
diagnostics_df.to_csv(OUTPUTS_DIR / 'variance_decomposition_diagnostics.csv', index=False)
print("   - Saved: variance_decomposition_diagnostics.csv")

# Save summary statistics
vw_mean = VW.mean()
ew_mean = EW.mean()

with open(OUTPUTS_DIR / 'variance_decomposition_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("VARIANCE DECOMPOSITION (4-way): What Moves Stock Prices\n")
    f.write("Brogaard, Nguyen, Putnins & Wu (2022) Methodology\n")
    f.write("Extended 3-Variable VAR: Market Return, Signed Dollar Volume, Stock Return\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"WINSORIZATION METHOD: Variance components at {int(WINSOR_BOUNDS[0]*100)}%-{int(WINSOR_BOUNDS[1]*100)}%\n")
    f.write("(Components winsorized BEFORE calculating shares, matching Appendix A implementation)\n\n")
    f.write("COMPONENTS (Disaggregated):\n")
    f.write("  1. Market Information: Variance driven by market shocks\n")
    f.write("  2. Private Information: Variance driven by private information (via volume)\n")
    f.write("  3. Public Information: Variance driven by public information (own residuals)\n")
    f.write("  4. Noise: Residual unexplained variance\n\n")
    
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
    f.write("HARMONIZED YEARLY DIAGNOSTICS (for cross-script comparability)\n")
    f.write("-" * 80 + "\n")
    f.write(diagnostics_df.to_string(index=False))

print("   - Saved: variance_decomposition_summary.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Print summary statistics to console
print("\n=== VARIANCE-WEIGHTED MEANS (Full Period) ===")
print(VW.mean().round(2))
print("\n=== EQUAL-WEIGHTED MEANS (Full Period) ===")
print(EW.mean().round(2))

