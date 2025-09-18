# NSE Monthly-ish Forecast Ranker

Python script `nse_monthly_forecast.py` builds a ranked list of NIFTY 100 stocks using a short-horizon (≈21 trading days) ensemble forecast plus momentum, relative strength, and risk-adjusted return components.

## Key Features
- Fetches NIFTY 50 + NIFTY NEXT 50 constituents from NSE JSON endpoints (falls back to Wikipedia).
- Historical prices via `nsepy` (preferred) else `yfinance` fallback.
- Historical prices via `nsepy` (preferred; see flag below) else `yfinance` fallback.
- Ensemble forecast: Holt-Winters (log) + drift (average recent log return).
- Momentum (≈1 quarter), relative strength vs synthetic equal‑weight NIFTY50, volatility (21d & 63d), and risk-adjusted return metrics.
- Composite score with configurable weights (expected return, momentum, relative strength, risk-adjusted return).
- CLI arguments to tune horizons and limits.
- Outputs ranked CSV (e.g. `nse_forecast_rank_YYYYMMDD.csv`) and prints top candidates.

## Installation
Create / activate a virtual environment (recommended) then install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
```
> NOTE: The file is named `requirement.txt` (singular). If you prefer the conventional name, copy/rename it to `requirements.txt`.

If you want faster HTML parsing, ensure `lxml` is installed (already listed).

## Usage
Basic run with defaults:
```
python nse_monthly_forecast.py
```
Use nsepy explicitly (experimental; disables by default due to upstream threading bug):
```
python nse_monthly_forecast.py --nsepy
```
Common overrides:
```
python nse_monthly_forecast.py --years 4 --horizon 30 --mom 84 --rs 84 --max 90
```
Silence debug output:
```
python nse_monthly_forecast.py --quiet
```
### Arguments
| Flag | Meaning | Default |
|------|---------|---------|
| `--years` | History years to download | 3 |
| `--horizon` | Forecast horizon (trading days) | 21 |
| `--mom` | Momentum window (days) | 63 |
| `--rs` | Relative strength window | 63 |
| `--max` | Max tickers to process | 100 |
| `--quiet` | Disable debug prints | False |
| `--nsepy` | Force use `nsepy` for history (experimental) | False |

## Output Columns (CSV)
- `ticker` – NSE symbol (.NS suffix when using Yahoo).
- `last_close` – Last available close.
- `forecast_21d_price` – Ensemble forecast price (mean of components).
- `expected_return_pct_21d` – Ensemble expected % return.
- `hw_exp_ret_pct` / `drift_exp_ret_pct` – Component expected returns.
- `momentum_pct_60d` – Momentum (% change over momentum window; label kept for backward compatibility if window changed to 63).
- `rel_strength_pct_90d` – Out/under‑performance vs synthetic equal‑weight NIFTY50 over RS window (label similarly legacy if window changed).
- `vol_pct_21d_ann` / `vol_pct_63d_ann` – Annualized log-return volatility (short / long window) in %.
- `risk_adj_return` – Expected return divided by short volatility (approx. information ratio proxy).
- `score` – Weighted composite (see weights in script: `W_EXP_RET`, `W_MOM`, `W_RS`, `W_RISK_ADJ`).
- `obs` – Number of price observations used.
- `rank` – Rank by score.

## Adjusting Weights
Inside the script near the config you can edit:
```
W_EXP_RET, W_MOM, W_RS, W_RISK_ADJ
```
Sum does not need to be 1.0; weights are linear multipliers.

## Improving Predictive Quality (Ideas)
- Add walk‑forward backtest to validate score stability.
- Include additional factors (earnings revision, volume trend) if data available.
- Add regime detection (volatility state) to dynamically switch weights.
- Replace / augment Holt‑Winters with Prophet, ARIMA, or gradient boosted models (requires more engineering & validation).

## Limitations / Disclaimers
- NSE endpoints are not an officially supported public API; may break or rate limit. Use sparingly and cache results.
- Forecasts are simplistic; not investment advice. Validate before using for trading.
- Yahoo and nsepy data can have corporate action adjustments; inspect edge cases (splits, dividends) yourself.

## Troubleshooting
| Issue | Cause | Action |
|-------|-------|--------|
| 403 errors on constituents | Blocked by NSE/Wikipedia | Retry later, ensure User-Agent intact, or reduce frequency |
| Empty CSV | All scores NaN (insufficient data) | Increase history years, reduce MIN_OBS, verify connectivity |
| Many NaN forecasts | Model failed or extreme returns filtered | Inspect individual series; reduce filter threshold |
| ValueError: "identically-labeled Series" during sort | Non-scalar values (e.g., Series) got into DataFrame columns | Update to latest `predict_options_with_sl.py`; ensure you coerce scalars before sorting and flatten yfinance MultiIndex columns |

Enable debug (default True) to see extra diagnostics. Set `--quiet` to suppress.

## Example Interpretation
If `expected_return_pct_21d` = 2.5 and `vol_pct_21d_ann` = 25, risk_adj_return ≈ 0.10 (2.5 / 25). Higher composite score indicates favorable blend of expected return, momentum, relative strength, and risk-adjusted efficiency.

## License
Personal / research use only; ensure compliance with NSE data terms for any redistribution.

---
Feel free to request backtest module, additional factors, or weight CLI switches.
