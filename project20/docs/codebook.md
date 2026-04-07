# Data Codebook

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
