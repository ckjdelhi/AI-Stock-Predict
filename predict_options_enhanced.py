import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import argparse
import requests
import warnings
warnings.filterwarnings('ignore')

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
    
    # Deduplicate and cut to max, with validation
    uniq = []
    seen = set()
    invalid_patterns = ['NIFTY', '50', 'NEXT', 'INDEX']
    
    for sym in symbols:
        if not sym or not isinstance(sym, str):
            continue
        sym = sym.strip().upper()
        if any(pattern in sym for pattern in invalid_patterns):
            continue
        if sym.isdigit() or len(sym) < 2:
            continue
        if sym in seen:
            continue
        
        seen.add(sym)
        uniq.append(sym)
        
        if len(uniq) >= max_tickers:
            break
    
    return [s + ".NS" for s in uniq]

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line"""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_stochastic(high, low, close, period=14):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period).mean()
    
    return adx

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def add_technical_indicators(data):
    """Add comprehensive technical indicators to the dataset"""
    df = data.copy()
    data_len = len(df)
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages - only add periods that fit in the data
    ma_periods = [5, 10, 20, 50]
    if data_len >= 200:
        ma_periods.append(200)
    
    for period in ma_periods:
        if data_len >= period:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # Price position relative to moving averages (only if they exist)
    if 'SMA_20' in df.columns:
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
    if 'SMA_50' in df.columns:
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        df['SMA20_to_SMA50'] = df['SMA_20'] / df['SMA_50']
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['RSI_SMA'] = df['RSI'].rolling(window=14).mean()
    
    # MACD
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Stochastic
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df['High'], df['Low'], df['Close'])
    
    # ADX (Trend Strength)
    df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
    
    # ATR (Volatility)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['ATR_Percent'] = df['ATR'] / df['Close'] * 100
    
    # Volume indicators
    volume_period = min(20, data_len//2)
    df['Volume_SMA'] = df['Volume'].rolling(window=volume_period).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    df['OBV'] = calculate_obv(df['Close'], df['Volume'])
    df['OBV_SMA'] = df['OBV'].rolling(window=volume_period).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=min(20, data_len//2)).std()
    if data_len >= 50:
        df['Volatility_Ratio'] = df['Volatility'] / df['Volatility'].rolling(window=50).mean()
    
    # Price momentum
    momentum_periods = [5, 10, 20]
    for period in momentum_periods:
        if data_len >= period:
            df[f'Momentum_{period}'] = df['Close'].pct_change(period)
    
    # Rate of Change
    if data_len >= 10:
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Williams %R
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['Williams_R'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    # Trend indicators
    df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
    df['Uptrend'] = ((df['Higher_High'] == 1) & (df['Higher_Low'] == 1)).astype(int)
    
    # Gap detection
    df['Gap_Up'] = ((df['Low'] > df['High'].shift(1))).astype(int)
    df['Gap_Down'] = ((df['High'] < df['Low'].shift(1))).astype(int)
    
    return df

def fetch_data(ticker, period="1y", verbose=False):
    """Fetch and prepare stock data with technical indicators"""
    try:
        if not ticker or not isinstance(ticker, str) or len(ticker) < 3:
            return None
        
        if verbose:
            print(f"\nFetching {ticker}...")
            
        data = yf.download(ticker, period=period, interval="1d", progress=False, threads=False, auto_adjust=False)
        if data is None or data.empty:
            if verbose:
                print(f"  No data returned for {ticker}")
            return None
        
        if verbose:
            print(f"  Downloaded {len(data)} rows")
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            try:
                lvl1 = data.columns.get_level_values(1)
                if len(set(lvl1)) == 1:
                    data.columns = data.columns.droplevel(1)
            except Exception:
                data.columns = data.columns.get_level_values(0)
        
        # Ensure we have required columns
        if "Close" not in data.columns and "Adj Close" in data.columns:
            data["Close"] = data["Adj Close"]
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_cols):
            if verbose:
                print(f"  Missing columns. Available: {data.columns.tolist()}")
            return None
        
        # Enforce numeric types
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        
        if verbose:
            print(f"  Adding technical indicators...")
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Create target: 1 if next day's close > today's close
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Drop NaN values
        data = data.dropna()
        
        if verbose:
            print(f"  Final data: {len(data)} rows after cleaning")
        
        return data
    except Exception as e:
        if verbose:
            print(f"Error fetching {ticker}: {e}")
        return None

def train_and_evaluate(data, min_train_size=40):
    """Train model with walk-forward validation"""
    # Select features for model
    feature_cols = [col for col in data.columns if col not in 
                   ['Target', 'Date', 'Dividends', 'Stock Splits', 'Capital Gains']]
    
    # Remove any remaining non-numeric columns
    numeric_data = data[feature_cols].select_dtypes(include=[np.number])
    feature_cols = numeric_data.columns.tolist()
    
    X = data[feature_cols].values
    y = data['Target'].values
    
    if len(X) < min_train_size:
        return None
    
    # Walk-forward validation: train on 70-80%, test on rest
    # Use 70% for smaller datasets to have enough test data
    train_ratio = 0.7 if len(X) < 100 else 0.8
    split_point = int(len(X) * train_ratio)
    
    # Ensure minimum sizes
    if split_point < 30 or (len(X) - split_point) < 10:
        return None
        
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble of models - adjust complexity based on data size
    n_estimators_rf = min(200, max(50, len(X_train) // 2))
    n_estimators_gb = min(100, max(30, len(X_train) // 3))
    max_depth = 5 if len(X_train) < 100 else 10
    
    rf_model = RandomForestClassifier(n_estimators=n_estimators_rf, max_depth=max_depth, 
                                     min_samples_split=10, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=n_estimators_gb, max_depth=4, 
                                         learning_rate=0.1, random_state=42)
    
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    
    # Ensemble prediction (average)
    ensemble_pred = ((rf_pred + gb_pred) / 2) >= 0.5
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, ensemble_pred)
    precision = precision_score(y_test, ensemble_pred, zero_division=0)
    recall = recall_score(y_test, ensemble_pred, zero_division=0)
    f1 = f1_score(y_test, ensemble_pred, zero_division=0)
    
    # Predict for latest data point
    X_latest = scaler.transform(X[-1:])
    rf_prob = rf_model.predict_proba(X_latest)[0][1]
    gb_prob = gb_model.predict_proba(X_latest)[0][1]
    confidence = (rf_prob + gb_prob) / 2
    prediction = 1 if confidence >= 0.5 else 0
    
    # Feature importance (from Random Forest)
    feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'top_features': top_features,
        'data': data
    }

def calculate_dynamic_targets(data, confidence):
    """Calculate dynamic stop-loss and targets based on volatility and confidence"""
    last_close = float(data['Close'].iloc[-1])
    atr = float(data['ATR'].iloc[-1])
    volatility = float(data['Volatility'].iloc[-1])
    
    # Dynamic stop-loss based on ATR (1.5x to 2.5x ATR)
    atr_multiplier = 1.5 + (1 - confidence)  # Lower confidence = wider stop
    stoploss = round(last_close - (atr * atr_multiplier), 2)
    stoploss_pct = ((stoploss / last_close) - 1) * 100
    
    # Dynamic targets based on ATR and confidence
    # Higher confidence = more aggressive targets
    target_multiplier = 2.0 + (confidence * 2)  # 2x to 4x ATR
    daily_target = round(last_close + (atr * target_multiplier), 2)
    weekly_target = round(last_close + (atr * target_multiplier * np.sqrt(5)), 2)
    monthly_target = round(last_close + (atr * target_multiplier * np.sqrt(21)), 2)
    
    # Calculate risk-reward ratio
    risk = last_close - stoploss
    reward = daily_target - last_close
    risk_reward = reward / risk if risk > 0 else 0
    
    return {
        'last_close': last_close,
        'stoploss': stoploss,
        'stoploss_pct': stoploss_pct,
        'daily_target': daily_target,
        'weekly_target': weekly_target,
        'monthly_target': monthly_target,
        'atr': atr,
        'volatility_pct': volatility * 100,
        'risk_reward': risk_reward
    }

def main():
    parser = argparse.ArgumentParser(description="Enhanced stock prediction with technical analysis")
    parser.add_argument("--max", type=int, default=100, help="Max tickers to scan")
    parser.add_argument("--nifty50-only", action="store_true", help="Use only NIFTY 50")
    parser.add_argument("--period", default="6mo", help="History period (e.g., 6mo, 1y, 2y)")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum confidence (0-1)")
    parser.add_argument("--min-accuracy", type=float, default=0.52, help="Minimum backtest accuracy")
    parser.add_argument("--top", type=int, default=10, help="Number of top picks to show")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for first stock")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("ENHANCED STOCK PREDICTION WITH TECHNICAL ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Min Confidence: {args.min_confidence:.0%}")
    print(f"  • Min Accuracy: {args.min_accuracy:.0%}")
    print(f"  • History Period: {args.period}")
    print(f"  • Top Picks: {args.top}")
    
    results = []
    universe = get_universe(max_tickers=args.max, include_next50=(not args.nifty50_only))
    print(f"\n⏳ Scanning {len(universe)} stocks...")
    
    if args.debug and len(universe) > 0:
        print(f"\n🔍 DEBUG MODE - Testing first stock: {universe[0]}")
        test_data = fetch_data(universe[0], period=args.period, verbose=True)
        if test_data is not None:
            print(f"✓ Successfully fetched {len(test_data)} rows")
            print(f"  Columns: {len(test_data.columns)}")
            print(f"  Sample features: {list(test_data.columns[:10])}")
        else:
            print("✗ Failed to fetch data")
        print("\nContinuing with full scan...\n")
    
    print()
    
    for i, stock in enumerate(universe, 1):
        print(f"[{i:3d}/{len(universe)}] {stock:15s}", end=" ", flush=True)
        
        data = fetch_data(stock, period=args.period, verbose=False)
        if data is None or len(data) < 60:
            print(f"✗ Insufficient data ({len(data) if data is not None else 0} rows)")
            continue
        
        result = train_and_evaluate(data)
        if result is None:
            print("✗ Training failed")
            continue
        
        # Filter by accuracy and confidence
        if result['accuracy'] < args.min_accuracy:
            print(f"✗ Low accuracy ({result['accuracy']:.1%})")
            continue
        
        if result['prediction'] == 1 and result['confidence'] >= args.min_confidence:
            targets = calculate_dynamic_targets(result['data'], result['confidence'])
            
            # Only include trades with good risk-reward ratio
            if targets['risk_reward'] >= 1.5:
                results.append({
                    'stock': stock,
                    'confidence': result['confidence'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'f1_score': result['f1_score'],
                    **targets,
                    'top_features': result['top_features']
                })
                print(f"✓ BULLISH (Conf: {result['confidence']:.0%}, Acc: {result['accuracy']:.0%}, R:R {targets['risk_reward']:.1f})")
            else:
                print(f"✗ Poor R:R ({targets['risk_reward']:.1f})")
        else:
            conf_str = f"{result['confidence']:.0%}" if result['prediction'] == 1 else "bearish"
            print(f"✗ {conf_str}")
    
    # Sort by confidence * accuracy (combined score)
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\n⚠️  No stocks met the criteria. Try lowering --min-confidence or --min-accuracy")
        return
    
    results_df['score'] = results_df['confidence'] * results_df['accuracy']
    results_df = results_df.sort_values(by='score', ascending=False).head(args.top)
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, 'Rank', results_df.index + 1)
    
    # Display results
    print("\n" + "="*80)
    print(f"TOP {len(results_df)} BULLISH PICKS")
    print("="*80 + "\n")
    
    for idx, row in results_df.iterrows():
        print(f"\n{'─'*80}")
        print(f"#{row['Rank']} {row['stock']} - Last Close: ₹{row['last_close']:.2f}")
        print(f"{'─'*80}")
        print(f"  Confidence:    {row['confidence']:.1%}  │  Accuracy:  {row['accuracy']:.1%}  │  Precision: {row['precision']:.1%}")
        print(f"  Stop Loss:     ₹{row['stoploss']:.2f} ({row['stoploss_pct']:+.1f}%)")
        print(f"  Daily Target:  ₹{row['daily_target']:.2f} ({(row['daily_target']/row['last_close']-1)*100:+.1f}%)")
        print(f"  Weekly Target: ₹{row['weekly_target']:.2f} ({(row['weekly_target']/row['last_close']-1)*100:+.1f}%)")
        print(f"  Monthly Target:₹{row['monthly_target']:.2f} ({(row['monthly_target']/row['last_close']-1)*100:+.1f}%)")
        print(f"  Risk:Reward:   1:{row['risk_reward']:.2f}")
        print(f"  ATR (14):      ₹{row['atr']:.2f}  │  Volatility: {row['volatility_pct']:.2f}%")
        print(f"\n  Key Indicators:")
        for feat, importance in row['top_features'][:3]:
            print(f"    • {feat}: {importance:.3f}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80 + "\n")
    
    summary_df = results_df[['Rank', 'stock', 'last_close', 'confidence', 'accuracy', 
                             'stoploss', 'daily_target', 'weekly_target', 'monthly_target', 'risk_reward']].copy()
    summary_df.columns = ['Rank', 'Stock', 'Price', 'Conf%', 'Acc%', 'StopLoss', 'Daily', 'Weekly', 'Monthly', 'R:R']
    summary_df['Conf%'] = summary_df['Conf%'].apply(lambda x: f"{x:.0%}")
    summary_df['Acc%'] = summary_df['Acc%'].apply(lambda x: f"{x:.0%}")
    summary_df['Price'] = summary_df['Price'].apply(lambda x: f"₹{x:.2f}")
    summary_df['StopLoss'] = summary_df['StopLoss'].apply(lambda x: f"₹{x:.2f}")
    summary_df['Daily'] = summary_df['Daily'].apply(lambda x: f"₹{x:.2f}")
    summary_df['Weekly'] = summary_df['Weekly'].apply(lambda x: f"₹{x:.2f}")
    summary_df['Monthly'] = summary_df['Monthly'].apply(lambda x: f"₹{x:.2f}")
    summary_df['R:R'] = summary_df['R:R'].apply(lambda x: f"1:{x:.1f}")
    
    print(summary_df.to_string(index=False))
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_txt = f"enhanced_picks_{timestamp}.txt"
    
    try:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("ENHANCED STOCK PREDICTIONS WITH TECHNICAL ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Min Confidence: {args.min_confidence:.0%}\n")
            f.write(f"Min Accuracy: {args.min_accuracy:.0%}\n")
            f.write(f"Period: {args.period}\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
            f.write("DETAILED ANALYSIS\n")
            f.write("="*80 + "\n")
            
            for idx, row in results_df.iterrows():
                f.write(f"\n{'─'*80}\n")
                f.write(f"#{row['Rank']} {row['stock']} - ₹{row['last_close']:.2f}\n")
                f.write(f"{'─'*80}\n")
                f.write(f"Confidence: {row['confidence']:.1%} | Accuracy: {row['accuracy']:.1%} | R:R: 1:{row['risk_reward']:.2f}\n")
                f.write(f"Stop Loss: ₹{row['stoploss']:.2f} | Daily Target: ₹{row['daily_target']:.2f}\n")
                f.write(f"Top Indicators: {', '.join([f[0] for f in row['top_features'][:3]])}\n")
        
        print(f"\n✓ Results saved to: {out_txt}")
    except Exception as e:
        print(f"\n✗ Failed to save file: {e}")
    
    print("\n" + "="*80)
    print("⚠️  DISCLAIMER: Past performance does not guarantee future results.")
    print("    Always do your own research and manage risk appropriately.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
