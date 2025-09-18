import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import argparse
import requests

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
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return s

def _fetch_index_constituents(session: requests.Session, index_name: str) -> list[str]:
    url = INDEX_ENDPOINT.format(index=index_name.replace(" ", "%20"))
    r = session.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    rows = data.get("data", [])
    return [row.get("symbol") for row in rows if row.get("symbol")]

def _fallback_wikipedia() -> list[str]:
    out: list[str] = []
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

def get_universe(max_tickers: int = 100, include_next50: bool = True) -> list[str]:
    s = _nse_session()
    s50: list[str] = []
    snext: list[str] = []
    try:
        s50 = _fetch_index_constituents(s, NIFTY50_NAME)
        if include_next50:
            snext = _fetch_index_constituents(s, NIFTY_NEXT50_NAME)
    except Exception:
        merged = _fallback_wikipedia()
        if len(merged) >= 50:
            s50, snext = merged[:50], merged[50:100]
        else:
            s50 = merged
            snext = []
    symbols = s50 + (snext if include_next50 else [])
    # Deduplicate and cut to max
    uniq = []
    seen = set()
    for sym in symbols:
        if sym and sym not in seen:
            seen.add(sym)
            uniq.append(sym)
        if len(uniq) >= max_tickers:
            break
    return [s + ".NS" for s in uniq]

def fetch_data(ticker, period="6mo"):
    try:
        # Disable progress bar and threads for consistency; fetch daily bars
        data = yf.download(ticker, period=period, interval="1d", progress=False, threads=False, auto_adjust=False)
        if data.empty:
            return None
        # If yfinance returns MultiIndex columns (e.g., level 0: OHLCV, level 1: ticker),
        # drop the ticker level when it's a single symbol to get flat columns.
        if isinstance(data.columns, pd.MultiIndex):
            try:
                # If only one unique ticker on second level, drop it
                lvl1 = data.columns.get_level_values(1)
                if len(set(lvl1)) == 1:
                    data.columns = data.columns.droplevel(1)
            except Exception:
                # Fallback: use the top level names
                data.columns = data.columns.get_level_values(0)

        # Ensure we have a closing price column
        if "Close" not in data.columns and "Adj Close" in data.columns:
            data["Close"] = data["Adj Close"]

        # Enforce numeric types
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        data["Return"] = data["Close"].pct_change()
        # Compute ATR(14) for target sizing
        if all(col in data.columns for col in ["High", "Low", "Close"]):
            prev_close = data["Close"].shift(1)
            tr1 = data["High"] - data["Low"]
            tr2 = (data["High"] - prev_close).abs()
            tr3 = (data["Low"] - prev_close).abs()
            data["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            data["ATR14"] = data["TR"].rolling(window=14, min_periods=14).mean()
        else:
            data["ATR14"] = np.nan
        data["Target"] = (data["Return"].shift(-1) > 0).astype(int)
        data = data.dropna()
        return data
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def train_and_predict(data):
    features = ["Open", "High", "Low", "Close", "Volume"]
    # Keep only features that exist (some feeds may lack Volume)
    feats = [c for c in features if c in data.columns]
    X = data[feats].copy()
    y = data["Target"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train on all but last row; ensure 2D input when predicting
    model.fit(X.iloc[:-1], y.iloc[:-1])
    pred = model.predict(X.iloc[[-1]])[0]
    
    # Define stoploss and target (simple heuristic: 2% SL, 4% Target)
    last_close_obj = data["Close"].iloc[-1]
    # Coerce potential 0-dim array/Series to scalar float
    if isinstance(last_close_obj, (pd.Series, pd.DataFrame, np.ndarray)):
        try:
            last_close = float(np.asarray(last_close_obj).squeeze().item())
        except Exception:
            last_close = float(np.asarray(last_close_obj).squeeze()[()])
    else:
        last_close = float(last_close_obj)
    stoploss = round(last_close * 0.98, 2)
    target = round(last_close * 1.04, 2)
    # ATR-based targets
    last_atr = float(np.asarray(data.get("ATR14", pd.Series([np.nan])).iloc[-1]).squeeze()) if "ATR14" in data.columns else np.nan
    if not np.isfinite(last_atr) or last_atr <= 0:
        # Fallback: use recent volatility to estimate ATR-like move
        daily_vol = float(data["Return"].rolling(20, min_periods=5).std().iloc[-1]) if "Return" in data.columns else 0.01
        est_move = last_close * daily_vol
        last_atr = est_move if np.isfinite(est_move) and est_move > 0 else last_close * 0.01
    daily_target = round(last_close + last_atr, 2)
    weekly_target = round(last_close + last_atr * np.sqrt(5), 2)
    monthly_target = round(last_close + last_atr * np.sqrt(21), 2)
    
    return pred, last_close, stoploss, target, daily_target, weekly_target, monthly_target

def main():
    parser = argparse.ArgumentParser(description="Next-day bullish picks with ATR targets")
    parser.add_argument("--max", type=int, default=100, help="Max tickers to scan (<=100)")
    parser.add_argument("--nifty50-only", action="store_true", help="Use only NIFTY 50 (exclude Next 50)")
    parser.add_argument("--period", default="6mo", help="History period for features (e.g., 3mo, 6mo, 1y)")
    args = parser.parse_args()

    results = []
    universe = get_universe(max_tickers=args.max, include_next50=(not args.nifty50_only))
    for stock in universe:
        data = fetch_data(stock, period=args.period)
        if data is not None and len(data) > 30:
            pred, close, sl, tgt, d_tgt, w_tgt, m_tgt = train_and_predict(data)
            if pred == 1:
                results.append((stock, close, sl, tgt, d_tgt, w_tgt, m_tgt))
    
    df = pd.DataFrame(results, columns=["Stock", "Last Close", "Stoploss", "Target", "Daily Target", "Weekly Target", "Monthly Target"])
    # Ensure values are numeric scalars for safe sorting
    df["Last Close"] = pd.to_numeric(df["Last Close"], errors="coerce")
    df["Stoploss"] = pd.to_numeric(df["Stoploss"], errors="coerce")
    df["Target"] = pd.to_numeric(df["Target"], errors="coerce")
    df["Daily Target"] = pd.to_numeric(df["Daily Target"], errors="coerce")
    df["Weekly Target"] = pd.to_numeric(df["Weekly Target"], errors="coerce")
    df["Monthly Target"] = pd.to_numeric(df["Monthly Target"], errors="coerce")
    df = df.sort_values(by="Last Close", ascending=False, kind="mergesort").head(10)
    df = df.reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    # Create a formatted view with percentages relative to Last Close
    def fmt_price_with_pct(base, val):
        if pd.isna(base) or pd.isna(val) or base == 0:
            return "-"
        pct = (val / base - 1.0) * 100.0
        sign = "+" if pct >= 0 else ""
        return f"{val:.2f} ({sign}{pct:.2f}%)"

    display_cols = [
        "Stock",
        "Last Close",
        "Stoploss",
        "Target",
        "Daily Target",
        "Weekly Target",
        "Monthly Target",
    ]

    df_display = df.copy()
    # Keep a numeric Last Close column for potential downstream use, but also add formatted columns
    df_display["Last Close"] = df["Last Close"].map(lambda x: f"{x:.2f}")
    df_display["Stoploss"] = [fmt_price_with_pct(b, v) for b, v in zip(df["Last Close"], df["Stoploss"])]
    df_display["Target"] = [fmt_price_with_pct(b, v) for b, v in zip(df["Last Close"], df["Target"])]
    df_display["Daily Target"] = [fmt_price_with_pct(b, v) for b, v in zip(df["Last Close"], df["Daily Target"])]
    df_display["Weekly Target"] = [fmt_price_with_pct(b, v) for b, v in zip(df["Last Close"], df["Weekly Target"])]
    df_display["Monthly Target"] = [fmt_price_with_pct(b, v) for b, v in zip(df["Last Close"], df["Monthly Target"])]

    header = "\nTop 10 bullish stock picks with SL, fixed Target, and ATR-based Daily/Weekly/Monthly targets:"
    table_str = df_display[["Rank"] + display_cols].to_string(index=False)
    print(header)
    print(table_str)

    # Save to a text file with date stamp
    from datetime import datetime as _dt
    out_txt = f"options_picks_{_dt.today().strftime('%Y%m%d')}.txt"
    try:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(header.strip() + "\n")
            f.write(table_str + "\n")
        print(f"\nSaved text: {out_txt}")
    except Exception as e:
        print(f"[WARN] Failed to write {out_txt}: {e}")

if __name__ == "__main__":
    main()
