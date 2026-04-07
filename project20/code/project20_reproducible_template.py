"""
===============================================================================
PROJECT 20: Reproducible Research Template for Sustainable Finance
===============================================================================
PURPOSE:
    A meta-project demonstrating best practices for reproducible empirical 
    research. Includes a complete pipeline: data → analysis → report.
METHOD:
    Demonstrates: Makefile automation, config files, modular code,
    documentation, and reproducibility checks.
DATA:
    Uses Yahoo Finance as example (easily swappable for any dataset)
===============================================================================
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import json, os, warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data/raw','data/processed','docs']:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# STEP 0: Load configuration
# =============================================================================
# In a real project, parameters come from config.yaml
config = {
    'tickers': ['ICLN', 'XLE', 'SPY'],
    'start_date': '2018-01-01',
    'end_date': '2025-12-31',
    'risk_free_rate': 0.04,
    'random_seed': 42,
    'analysis_version': '1.0.0'
}

with open('data/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("STEP 0: Configuration loaded")

np.random.seed(config['random_seed'])

# =============================================================================
# STEP 1: Data Collection (01_collect_data.py)
# =============================================================================
print("\nSTEP 1: Collecting data...")

prices = {}
for t in config['tickers']:
    df = yf.download(t, start=config['start_date'], end=config['end_date'],
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    prices[t] = df['Close']
    print(f"  {t}: {len(df)} observations")

prices = pd.DataFrame(prices).dropna()
prices.to_csv('data/raw/daily_prices.csv')
print(f"  Saved: data/raw/daily_prices.csv ({len(prices)} rows)")

# =============================================================================
# STEP 2: Data Cleaning (02_clean_data.py)
# =============================================================================
print("\nSTEP 2: Cleaning and transforming data...")

returns = np.log(prices / prices.shift(1)).dropna() * 100
excess_returns = returns.copy()
for col in excess_returns.columns:
    excess_returns[col] = excess_returns[col] - config['risk_free_rate'] / 252 * 100

returns.to_csv('data/processed/daily_returns.csv')
excess_returns.to_csv('data/processed/excess_returns.csv')

# Data quality report
quality = pd.DataFrame({
    'observations': returns.count(),
    'missing': returns.isnull().sum(),
    'mean': returns.mean().round(4),
    'std': returns.std().round(4),
    'min': returns.min().round(4),
    'max': returns.max().round(4),
    'skewness': returns.skew().round(4),
    'kurtosis': returns.kurtosis().round(4)
})
quality.to_csv('output/tables/data_quality_report.csv')
print("  Saved: data quality report")

# =============================================================================
# STEP 3: Analysis (03_analyze.py)
# =============================================================================
print("\nSTEP 3: Running analysis...")

# Performance metrics
def compute_metrics(ret_series, name, rf=config['risk_free_rate']):
    annual_ret = ret_series.mean() * 252
    annual_vol = ret_series.std() * np.sqrt(252)
    sharpe = (annual_ret - rf) / annual_vol if annual_vol > 0 else 0
    cum = (1 + ret_series/100).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return {
        'Asset': name,
        'Annual Return (%)': round(annual_ret, 2),
        'Annual Volatility (%)': round(annual_vol, 2),
        'Sharpe Ratio': round(sharpe, 4),
        'Max Drawdown (%)': round(max_dd, 2),
        'Skewness': round(ret_series.skew(), 4),
        'Kurtosis': round(ret_series.kurtosis(), 4)
    }

metrics = pd.DataFrame([compute_metrics(returns[col], col) for col in returns.columns])
metrics.to_csv('output/tables/performance_metrics.csv', index=False)
print(metrics.to_string(index=False))

# Correlation analysis
corr = returns.corr()
corr.to_csv('output/tables/correlation_matrix.csv')

# Rolling beta regression (ICLN vs SPY)
window = 60
betas = []
for i in range(window, len(returns)):
    y = returns['ICLN'].iloc[i-window:i]
    x = add_constant(returns['SPY'].iloc[i-window:i])
    try:
        model = OLS(y, x).fit()
        betas.append({'date': returns.index[i], 'beta': model.params.iloc[1], 'r2': model.rsquared})
    except:
        pass
beta_df = pd.DataFrame(betas)
beta_df.to_csv('output/tables/rolling_betas.csv', index=False)

# =============================================================================
# STEP 4: Visualizations (04_visualize.py)
# =============================================================================
print("\nSTEP 4: Creating publication-quality figures...")

# Fig 1: Cumulative returns
fig, ax = plt.subplots(figsize=(12, 6))
cum = (1 + returns/100).cumprod()
colors = {'ICLN':'#2ecc71','XLE':'#e74c3c','SPY':'#3498db'}
for col in cum.columns:
    ax.plot(cum.index, cum[col], label=col, color=colors[col], linewidth=1.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Cumulative Returns', fontweight='bold')
ax.set_ylabel('Growth of $1')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('output/figures/fig1_cumulative_returns.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Rolling beta
fig, ax = plt.subplots(figsize=(12, 5))
if len(beta_df) > 0:
    ax.plot(beta_df['date'], beta_df['beta'], color='#2ecc71', linewidth=1.2)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(beta_df['date'], beta_df['beta'], 1, alpha=0.2, color='#2ecc71')
ax.set_title('Rolling 60-Day Beta: ICLN vs SPY', fontweight='bold')
ax.set_ylabel('Beta')
plt.tight_layout()
plt.savefig('output/figures/fig2_rolling_beta.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax, 
            linewidths=1, square=True)
ax.set_title('Return Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: Return distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(returns.columns):
    axes[i].hist(returns[col], bins=60, density=True, alpha=0.7, color=colors[col])
    axes[i].set_title(f'{col} Returns', fontweight='bold')
    axes[i].set_xlabel('Daily Return (%)')
plt.tight_layout()
plt.savefig('output/figures/fig4_return_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# STEP 5: Generate reproducibility artifacts
# =============================================================================
print("\nSTEP 5: Creating reproducibility documentation...")

# Codebook
codebook = """# Data Codebook

## daily_prices.csv
| Variable | Type | Description |
|----------|------|-------------|
| Date | datetime | Trading date |
| ICLN | float | Adjusted closing price of iShares Global Clean Energy ETF |
| XLE | float | Adjusted closing price of Energy Select Sector SPDR |
| SPY | float | Adjusted closing price of S&P 500 ETF |

## daily_returns.csv
| Variable | Type | Description |
|----------|------|-------------|
| Date | datetime | Trading date |
| ICLN | float | Daily log return (%) |
| XLE | float | Daily log return (%) |
| SPY | float | Daily log return (%) |

## Data Sources
- Yahoo Finance (via yfinance Python library)
- All prices are adjusted for splits and dividends (auto_adjust=True)
"""

with open('docs/codebook.md', 'w') as f:
    f.write(codebook)

# Methods documentation
methods = """# Methodology

## Performance Metrics
- **Annual Return**: Mean daily return × 252
- **Annual Volatility**: Daily std × √252
- **Sharpe Ratio**: (Annual return - Risk-free rate) / Annual volatility
- **Max Drawdown**: Maximum peak-to-trough decline

## Rolling Beta
- Window: 60 trading days
- Model: R_ICLN = α + β × R_SPY + ε
- Estimated via OLS

## Configuration
- Risk-free rate: 4.0% annual
- Random seed: 42
"""

with open('docs/methods.md', 'w') as f:
    f.write(methods)

print("  Saved: docs/codebook.md, docs/methods.md")
print("\n  COMPLETE!")
