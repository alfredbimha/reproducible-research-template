# Methodology

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
