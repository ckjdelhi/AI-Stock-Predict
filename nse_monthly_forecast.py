"""Monthly-esque (21 trading day) forecast ranking for NIFTY 100.

Rewritten to prefer NSE data sources for better fidelity vs pure Yahoo Finance.
Features:
 1. Attempts to pull constituents from NSE's public index JSON endpoints.
 2. Falls back to Wikipedia if NSE endpoints blocked.
 3. Historical price retrieval via nsepy (if installed) else yfinance fallback.
 4. Forecast uses Holt-Winters on log prices + simple drift sanity check.
 5. Composite score blends expected return, 60d momentum, and relative strength.

NOTE: NSE website endpoints are not an official public API; excessive automated
requests may be blocked. Use respectful intervals. For production research,
license data from an authorized vendor.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import math
import time
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import requests
import pandas as pd
import numpy as np

try:  # Optional dependency
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # fallback later

try:  # Optional dependency
    from nsepy import get_history  # type: ignore
except Exception:  # pragma: no cover
    get_history = None

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------------- Config ----------------
"""Tuning guidance (updated defaults):

Rationale:
 - YEARS_HISTORY: shorter (3) captures current regime; too long can dilute.
 - MIN_OBS: 252 = 1 trading year ensures model stability.
 - MOMENTUM_WINDOW / REL_STRENGTH_WINDOW: align to ~quarter (63 trading days).
 - MAX_TICKERS: restrict to 100 for speed + data quality.
 - Scoring weights adjusted (0.5 exp_ret, 0.3 momentum, 0.2 rel_strength).
"""

YEARS_HISTORY = 3              # Use last ~3Y for regime relevance
FORECAST_HORIZON = 21          # Trading days (~1 month)
MIN_OBS = 252                  # Require at least 1Y of data
MOMENTUM_WINDOW = 63           # ~Quarter momentum
REL_STRENGTH_WINDOW = 63       # Align RS with momentum horizon
VOL_WINDOW_SHORT = 21          # 1M volatility window
VOL_WINDOW_LONG = 63           # 1Q volatility window
MAX_TICKERS = 100              # Safety cap
MAX_WORKERS = 8                # Parallel downloads
NSE_PAUSE = 0.2                # polite pause between NSE calls (seconds)
DEBUG = True                   # set False to silence debug prints

# Scoring weights (can be overridden via CLI later if desired)
W_EXP_RET = 0.45
W_MOM = 0.25
W_RS = 0.15
W_RISK_ADJ = 0.15   # risk-adjusted component weight

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/",
}

INDEX_ENDPOINT = "https://www.nseindia.com/api/equity-stockIndices?index={index}"
NIFTY50_NAME = "NIFTY 50"
NIFTY_NEXT50_NAME = "NIFTY NEXT 50"


def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA_HEADERS)
    # Prime cookies
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return s


def _fetch_index_constituents(session: requests.Session, index_name: str) -> List[str]:
    url = INDEX_ENDPOINT.format(index=index_name.replace(" ", "%20"))
    r = session.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    stocks = data.get("data", [])
    symbols = [row.get("symbol") for row in stocks if row.get("symbol")]
    return symbols


def _fallback_wikipedia() -> List[str]:
    """Fallback scrape (previous approach)."""
    out = []
    try:
        for wiki_url in [
            "https://en.wikipedia.org/wiki/NIFTY_50",
            "https://en.wikipedia.org/wiki/NIFTY_Next_50",
        ]:
            html = requests.get(wiki_url, headers=UA_HEADERS, timeout=20).text
            tables = pd.read_html(html)
            for t in tables:
                if {"Symbol"}.issubset(t.columns):
                    out.extend(t["Symbol"].dropna().astype(str).tolist())
                    break
    except Exception:
        return []
    return sorted(set(out))


def get_nifty_100_symbols() -> tuple[List[str], List[str]]:
    """Return (all_100_with_suffix, nifty50_with_suffix).

    Keeps an explicit NIFTY50 list to build benchmark accurately instead of
    assuming first 50 items of combined list.
    """
    session = _nse_session()
    s50: List[str] = []
    snext: List[str] = []
    try:
        s50 = _fetch_index_constituents(session, NIFTY50_NAME)
        snext = _fetch_index_constituents(session, NIFTY_NEXT50_NAME)
    except Exception:
        # Fallback merges both; we cannot distinguish which are 50 vs next 50
        merged = _fallback_wikipedia()
        if len(merged) >= 50:
            s50, snext = merged[:50], merged[50:100]
        else:
            s50 = merged
            snext = []
    symbols = sorted(set(s50 + snext))[:MAX_TICKERS]
    return [s + ".NS" for s in symbols], [s + ".NS" for s in s50]


def _fetch_history_single(symbol: str, start: datetime, end: datetime) -> pd.Series | None:
    base_symbol = symbol.replace(".NS", "")
    if get_history is not None:
        try:
            df = get_history(symbol=base_symbol, start=start.date(), end=end.date())
            if not df.empty and "Close" in df:
                s = df["Close"].astype(float)
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
        except Exception:
            pass
    if yf is not None:
        try:
            df = yf.download(symbol, start=start, end=end + timedelta(days=1), progress=False, interval="1d", auto_adjust=False, threads=False)
            if not df.empty and "Adj Close" in df:
                s = df["Adj Close"].dropna().astype(float)
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
        except Exception:
            pass
    return None


def _coerce_to_series(obj) -> pd.Series | None:
    """Coerce various return shapes to a clean float Series (prices).

    Accepts:
      - Series: returns as float Series
      - DataFrame with column Close / Adj Close / close / last / Close Price
      - DataFrame with exactly 1 numeric column (uses it)
    Returns None if cannot confidently extract.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        return obj.astype(float)
    if isinstance(obj, pd.DataFrame):
        # prefer common price column names (case insensitive)
        preferred = ["Adj Close", "Close", "Close Price", "close", "last"]
        cols_lower = {c.lower(): c for c in obj.columns}
        for name in preferred:
            if name.lower() in cols_lower:
                s = obj[cols_lower[name.lower()]].dropna()
                return s.astype(float)
        # single numeric column fallback
        numeric_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        if len(numeric_cols) == 1:
            return obj[numeric_cols[0]].dropna().astype(float)
    return None


def download_prices(tickers: List[str], years: int = YEARS_HISTORY) -> Dict[str, pd.Series]:
    end = datetime.today()
    start = end - timedelta(days=int(years * 365.25))
    out: Dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_history_single, t, start, end): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Hist"):
            t = futures[fut]
            try:
                raw = fut.result()
                s = _coerce_to_series(raw)
                if s is None or s.empty:
                    continue
                # Remove duplicate index
                s = s[~s.index.duplicated(keep="last")]
                out[t] = s
            except Exception:
                continue
    return out


def momentum_pct(series: pd.Series, window: int = MOMENTUM_WINDOW) -> float:
    if len(series) < window + 1:
        return math.nan
    return (series.iloc[-1] / series.iloc[-window - 1]) * 100.0 - 100.0


def relative_strength_pct(series: pd.Series, benchmark: pd.Series, window: int = REL_STRENGTH_WINDOW) -> float:
    if len(series) < window + 1 or len(benchmark) < window + 1:
        return math.nan
    s_ret = series.iloc[-1] / series.iloc[-window - 1] - 1
    b_ret = benchmark.iloc[-1] / benchmark.iloc[-window - 1] - 1
    if b_ret == 0:
        return math.nan
    return (s_ret - b_ret) * 100.0


def hw_forecast(series: pd.Series, horizon: int = FORECAST_HORIZON) -> Tuple[float, float]:
    """Forecast terminal price after horizon days.

    Steps:
      1. Resample to business daily, forward-fill.
      2. Work on log prices for smoother additive structure.
      3. Holt-Winters (trend only, damped) on log prices.
      4. Convert back to level; compute expected % return.
    """
    try:
        s = series.asfreq("B").ffill()
        if len(s) < MIN_OBS:
            return math.nan, math.nan
        log_s = np.log(s)
        model = ExponentialSmoothing(log_s, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        fcst_log = fit.forecast(horizon)
        f_last = float(np.exp(fcst_log.iloc[-1]))
        last = float(s.iloc[-1])
        exp_ret = (f_last / last - 1.0) * 100.0
        # Sanity: if extreme (>50% in a month) clamp to NaN
        if abs(exp_ret) > 50:
            return math.nan, math.nan
        return f_last, exp_ret
    except Exception:
        return math.nan, math.nan


def drift_forecast(series: pd.Series, horizon: int = FORECAST_HORIZON, window: int = MOMENTUM_WINDOW) -> Tuple[float, float]:
    """Simple log-return drift forecast using recent average daily log return.
    Acts as a lightweight, robust alternative component to Holt-Winters.
    """
    try:
        if len(series) < window + 5:
            return math.nan, math.nan
        s = series.asfreq("B").ffill()
        log_ret = np.log(s / s.shift(1)).dropna()
        mu = log_ret.tail(window).mean()
        last = float(s.iloc[-1])
        f_last = float(np.exp(np.log(last) + mu * horizon))
        exp_ret = (f_last / last - 1.0) * 100.0
        if abs(exp_ret) > 50:
            return math.nan, math.nan
        return f_last, exp_ret
    except Exception:
        return math.nan, math.nan


def ensemble_forecast(series: pd.Series, horizon: int = FORECAST_HORIZON) -> Tuple[float, float, float, float]:
    """Combine Holt-Winters and drift forecasts.

    Returns (final_price, exp_ret_pct, hw_exp_ret, drift_exp_ret)
    where final is the mean of available component forecasts.
    """
    hw_price, hw_ret = hw_forecast(series, horizon)
    d_price, d_ret = drift_forecast(series, horizon)
    rets = [r for r in [hw_ret, d_ret] if not math.isnan(r)]
    prices = [p for p in [hw_price, d_price] if not math.isnan(p)]
    if not rets or not prices:
        return math.nan, math.nan, hw_ret, d_ret
    return float(np.mean(prices)), float(np.mean(rets)), hw_ret, d_ret


def realized_vol(series: pd.Series, window: int) -> float:
    if len(series) < window + 2:
        return math.nan
    log_ret = np.log(series / series.shift(1)).dropna().tail(window)
    if log_ret.empty:
        return math.nan
    # annualize (assuming 252 trading days)
    return float(log_ret.std(ddof=0) * math.sqrt(252) * 100.0)  # in %


def build_index_series(prices: Dict[str, pd.Series], tickers50: List[str]) -> pd.Series:
    """Equal-weight synthetic index from explicit NIFTY50 tickers.

    Avoids column length mismatch by constructing DataFrame via dict.
    """
    avail = {t: prices[t] for t in tickers50 if t in prices and isinstance(prices[t], pd.Series)}
    if not avail:
        return pd.Series(dtype=float)
    try:
        df = pd.DataFrame(avail)
    except ValueError as e:  # debug aid
        if DEBUG:
            print("[DEBUG] build_index_series ValueError:", e)
            print("[DEBUG] avail keys:", list(avail.keys())[:5], "count=", len(avail))
            for k,v in list(avail.items())[:2]:
                print("[DEBUG] sample series", k, type(v), getattr(v, 'shape', None))
        return pd.Series(dtype=float)
    eq = df.pct_change().mean(axis=1)
    idx = (1 + eq.fillna(0)).cumprod()
    base = 100 * idx / idx.iloc[0]
    return base


def score_row(exp_ret: float, momentum: float, rel_strength: float, risk_adj: float) -> float:
    def _coerce(x):
        try:
            if isinstance(x, (pd.Series, list, tuple, np.ndarray)):
                if len(x) == 0:
                    return math.nan
                # flatten and take first value
                x = np.asarray(x).flatten()[0]
            return float(x)
        except Exception:
            return math.nan
    exp_ret_f = _coerce(exp_ret)
    momentum_f = _coerce(momentum)
    rel_strength_f = _coerce(rel_strength)
    comps = []
    if not math.isnan(exp_ret_f):
        comps.append(W_EXP_RET * exp_ret_f)
    if not math.isnan(momentum_f):
        comps.append(W_MOM * momentum_f)
    if not math.isnan(rel_strength_f):
        comps.append(W_RS * rel_strength_f)
    ra_f = _coerce(risk_adj)
    if not math.isnan(ra_f):
        comps.append(W_RISK_ADJ * ra_f)
    return sum(comps) if comps else math.nan


def run_pipeline():
    print("Fetching NIFTY 100 universe …")
    tickers, nifty50_list = get_nifty_100_symbols()
    if not tickers:
        raise RuntimeError("No tickers fetched.")
    print(f"Universe size: {len(tickers)} (NIFTY50={len(nifty50_list)})")

    print("Downloading price history …")
    prices = download_prices(tickers)
    if not prices:
        raise RuntimeError("Failed to download any price series.")

    bench = build_index_series(prices, nifty50_list)

    rows = []
    for t in tqdm(tickers, desc="Model"):
        s = prices.get(t)
        if s is None or s.empty:
            continue
        # safety: ensure Series (earlier coercion should guarantee)
        if isinstance(s, pd.DataFrame):
            s = _coerce_to_series(s)
            if s is None or s.empty:
                continue
        f_price, exp_ret, hw_ret, drift_ret = ensemble_forecast(s)
        mom = momentum_pct(s)
        rs = relative_strength_pct(s, bench) if not bench.empty else math.nan
        vol_s = realized_vol(s, VOL_WINDOW_SHORT)
        vol_l = realized_vol(s, VOL_WINDOW_LONG)
        # Risk-adjusted return uses short vol (avoid divide by zero)
        risk_adj = exp_ret / vol_s if (vol_s and not math.isnan(vol_s) and vol_s > 0) else math.nan
        sc = score_row(exp_ret, mom, rs, risk_adj)
        rows.append({
            "ticker": t,
            "last_close": float(s.iloc[-1]),
            "forecast_21d_price": f_price,
            "expected_return_pct_21d": exp_ret,
            "hw_exp_ret_pct": hw_ret,
            "drift_exp_ret_pct": drift_ret,
            "momentum_pct_60d": mom,
            "rel_strength_pct_90d": rs,
            "vol_pct_21d_ann": vol_s,
            "vol_pct_63d_ann": vol_l,
            "risk_adj_return": risk_adj,
            "score": sc,
            "obs": len(s),
        })

    try:
        df = pd.DataFrame.from_records(rows)  # safer for list-of-dicts
    except ValueError as e:
        if DEBUG:
            print("[DEBUG] DataFrame creation failed:", e)
            print("[DEBUG] rows type:", type(rows), "len:", (len(rows) if hasattr(rows,'__len__') else 'n/a'))
            if rows:
                print("[DEBUG] first row keys:", list(rows[0].keys()) if isinstance(rows, list) else list(rows.keys()))
        raise
    df = df.dropna(subset=["score"]).sort_values("score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)

    out_csv = f"nse_forecast_rank_{datetime.today().strftime('%Y%m%d')}.csv"
    df.to_csv(out_csv, index=False)

    top = df[df["expected_return_pct_21d"] > 0].head(15)
    print("\n========== TOP (Research Candidates) ==========")
    if top.empty:
        print("No positives. Review full CSV.")
    else:
        print(top[[
            "rank","ticker","last_close","forecast_21d_price",
            "expected_return_pct_21d","momentum_pct_60d","rel_strength_pct_90d",
            "risk_adj_return","score"
        ]].to_string(index=False))
    print(f"\nSaved: {out_csv}")


def main():  # entrypoint
    # Declare globals before any usage (needed because we assign later)
    global YEARS_HISTORY, FORECAST_HORIZON, MOMENTUM_WINDOW, REL_STRENGTH_WINDOW, MAX_TICKERS, DEBUG

    parser = argparse.ArgumentParser(description="NSE monthly-ish forecast ranker")
    parser.add_argument("--years", type=int, default=YEARS_HISTORY, help="History years to download")
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON, help="Forecast horizon trading days")
    parser.add_argument("--mom", type=int, default=MOMENTUM_WINDOW, help="Momentum window")
    parser.add_argument("--rs", type=int, default=REL_STRENGTH_WINDOW, help="Relative strength window")
    parser.add_argument("--max", type=int, default=MAX_TICKERS, help="Max tickers")
    parser.add_argument("--quiet", action="store_true", help="Silence debug output")
    args = parser.parse_args()

    YEARS_HISTORY = args.years
    FORECAST_HORIZON = args.horizon
    MOMENTUM_WINDOW = args.mom
    REL_STRENGTH_WINDOW = args.rs
    MAX_TICKERS = args.max
    if args.quiet:
        DEBUG = False

    start_time = time.time()
    run_pipeline()
    print(f"\nDone in {time.time() - start_time:0.1f}s")


if __name__ == "__main__":
    main()