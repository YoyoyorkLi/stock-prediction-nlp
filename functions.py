import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"  # Set up personal Alpaca API key before running the script
SECRET_KEY = "YOUR_SECRET_KEY"
BASE_URL = "https://paper-api.alpaca.markets"

def get_data(tickers, days):
    # Authenticate
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    request_days = days * 2  # Request 2 times training days for non-trading days
    training_days = days + 21
    #end_date = datetime(2025, 7, 3)  # Bullish day
    #end_date = datetime(2024, 8, 4) # Bearish day
    #end_date = datetime.now() # Predict tomorrow's market
    end_date = datetime.now() - timedelta(days=1)  # Predict yesterday for evaluation
    start_date = end_date - timedelta(days=request_days)

    if isinstance(tickers, str):
        tickers = [tickers]

    stocks_data = {}

    for ticker in tickers:
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = client.get_stock_bars(request_params)
        df = bars.df.reset_index()

        if df.empty:
            print(f"No data for {ticker}")
            continue

        if "timestamp" not in df.columns:
            print(f"'timestamp' missing for {ticker}: {df.columns}")
            continue

        df = df.sort_values("timestamp")
        df = (
            df.groupby("symbol", group_keys=False)
            .tail(training_days)
            .reset_index(drop=True)
        )
        stocks_data[ticker] = df

    return stocks_data

def add_lagged_features_regression(df):
    # All features should have shift(1), except for target
    # Lagged close, open and volume
    for col in ['close', 'open', 'volume']:
        for lag in range(1, 4):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Returns and gaps
    df['return_1d'] = df['close'].pct_change(1).shift(1)
    df['return_5d'] = df['close'].pct_change(5).shift(1)
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1)

    # Rolling statistics
    df['sma_5'] = df['close'].rolling(window=5).mean().shift(1)
    df['sma_10'] = df['close'].rolling(window=10).mean().shift(1)
    df['std_5'] = df['close'].rolling(window=5).std().shift(1)
    df['std_10'] = df['close'].rolling(window=10).std().shift(1)
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean().shift(1)

    # Price ratios
    df['price_vs_sma5'] = df['close'].shift(1) / df['sma_5']
    df['price_vs_sma10'] = df['close'].shift(1) / df['sma_10']
    df['price_vs_vwap'] = df['close'].shift(1) / df['vwap'].shift(1)

    # Volume features
    df['volume_avg_10'] = df['volume'].rolling(window=10).mean().shift(1)
    df['volume_spike'] = df['volume'].shift(1) / df['volume_avg_10']
    df['volume_change_pct'] = df['volume'].pct_change(1).shift(1)

    # Volatility and range
    df['range'] = df['high'] - df['low']
    df['range_lag1'] = df['range'].shift(1)

    # MACD and signal
    ema_12_raw = df['close'].ewm(span=12, adjust=False).mean()
    ema_26_raw = df['close'].ewm(span=26, adjust=False).mean()
    macd_raw = ema_12_raw - ema_26_raw
    df['MACD'] = macd_raw.shift(1)
    df['MACD_signal'] = macd_raw.shift(1).ewm(span=9, adjust=False).mean()

    # RSI 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = (100 - (100 / (1 + rs))).shift(1)

    # Longer return
    df['return_10d'] = df['close'].pct_change(10).shift(1)

    # Target: next day's return
    df['target'] = df['close'].pct_change().shift(-1)

    # Relative movement and return compared to the market (S&P500)
    df['relative_return_1d'] = df['return_1d'] - df['target_sp500']
    df['rolling_corr_10'] = df['return_1d'].rolling(10).corr(df['target_sp500']).shift(1)
    df['return_vs_sp500'] = df['return_1d'] / (df['target_sp500'] + 1e-6)
    cov = df['return_1d'].rolling(20).cov(df['target_sp500']).shift(1)
    var = df['target_sp500'].rolling(20).var().shift(1)

    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)

    # How far price is from the bands (normalized distance)
    df['price_vs_upper_band'] = df['close'].shift(1) / upper_band.shift(1)
    df['price_vs_lower_band'] = df['close'].shift(1) / lower_band.shift(1)

    # Rolling Skewness of 1-day returns
    df['return_skew_5d'] = df['return_1d'].rolling(window=5).skew().shift(1)

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14 + 1e-6)).shift(1)

    # Price Acceleration (second derivative of close price)
    df['price_accel'] = df['close'].shift(1) - 2 * df['close'].shift(2) + df['close'].shift(3)

    # Rolling correlation with S&P 500
    if 'target_sp500' in df.columns:
        df['rolling_corr_10'] = df['return_1d'].rolling(10).corr(df['target_sp500']).shift(1)

    df.drop(columns=[
        'open',  # current-day open (used in gap)
        'high',  # current-day high
        'low',  # current-day low
        'close',  # current-day close (used in target, return, etc.)
        'volume',  # current-day volume
        'trade_count',  # not used in any feature
        'vwap',  # raw current-day vwap (you use vwap.shift(1) instead)
        'range',  # current-day high-low range (already lagged as range_lag1)
        'target_sp500'
    ], errors='ignore', inplace=True)

    # Drop all rows with any NaN values except the last row (for prediction)
    last_row = df.iloc[[-1]].copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add one last row with latest available features (but NaN target)
    df.loc[len(df)] = last_row.iloc[0]

    return df

def add_lagged_features_classification(df):
    # All features should have shift(1), except for target
    # Lagged close, open and volume
    for col in ['close', 'open', 'volume']:
        for lag in range(1, 4):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Returns and gaps
    df['return_1d'] = df['close'].pct_change(1).shift(1)
    df['return_5d'] = df['close'].pct_change(5).shift(1)
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1)

    # Rolling statistics
    df['sma_5'] = df['close'].rolling(window=5).mean().shift(1)
    df['sma_10'] = df['close'].rolling(window=10).mean().shift(1)
    df['std_5'] = df['close'].rolling(window=5).std().shift(1)
    df['std_10'] = df['close'].rolling(window=10).std().shift(1)
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean().shift(1)

    # Price ratios
    df['price_vs_sma5'] = df['close'].shift(1) / df['sma_5']
    df['price_vs_sma10'] = df['close'].shift(1) / df['sma_10']
    df['price_vs_vwap'] = df['close'].shift(1) / df['vwap'].shift(1)

    # Volume features
    df['volume_avg_10'] = df['volume'].rolling(window=10).mean().shift(1)
    df['volume_spike'] = df['volume'].shift(1) / df['volume_avg_10']
    df['volume_change_pct'] = df['volume'].pct_change(1).shift(1)

    # Volatility and range
    df['range'] = df['high'] - df['low']
    df['range_lag1'] = df['range'].shift(1)

    # MACD and signal
    ema_12_raw = df['close'].ewm(span=12, adjust=False).mean()
    ema_26_raw = df['close'].ewm(span=26, adjust=False).mean()
    macd_raw = ema_12_raw - ema_26_raw
    df['MACD'] = macd_raw.shift(1)
    df['MACD_signal'] = macd_raw.shift(1).ewm(span=9, adjust=False).mean()

    # RSI 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = (100 - (100 / (1 + rs))).shift(1)

    # Longer return
    df['return_10d'] = df['close'].pct_change(10).shift(1)

    # Target: next day's return
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Relative movement and return compared to the market (S&P500)
    df['relative_return_1d'] = df['return_1d'] - df['target_sp500']
    df['rolling_corr_10'] = df['return_1d'].rolling(10).corr(df['target_sp500']).shift(1)
    df['return_vs_sp500'] = df['return_1d'] / (df['target_sp500'] + 1e-6)
    cov = df['return_1d'].rolling(20).cov(df['target_sp500']).shift(1)
    var = df['target_sp500'].rolling(20).var().shift(1)

    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)

    # How far price is from the bands (normalized distance)
    df['price_vs_upper_band'] = df['close'].shift(1) / upper_band.shift(1)
    df['price_vs_lower_band'] = df['close'].shift(1) / lower_band.shift(1)

    # Rolling Skewness of 1-day returns
    df['return_skew_5d'] = df['return_1d'].rolling(window=5).skew().shift(1)

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14 + 1e-6)).shift(1)

    # Price Acceleration (second derivative of close price)
    df['price_accel'] = df['close'].shift(1) - 2 * df['close'].shift(2) + df['close'].shift(3)

    # Rolling correlation with S&P 500
    if 'target_sp500' in df.columns:
        df['rolling_corr_10'] = df['return_1d'].rolling(10).corr(df['target_sp500']).shift(1)

    df.drop(columns=[
        'open',  # current-day open (used in gap)
        'high',  # current-day high
        'low',  # current-day low
        'close',  # current-day close (used in target, return, etc.)
        'volume',  # current-day volume
        'trade_count',  # not used in any feature
        'vwap',  # raw current-day vwap (you use vwap.shift(1) instead)
        'range',  # current-day high-low range (already lagged as range_lag1)
        'target_sp500'
    ], errors='ignore', inplace=True)

    # Drop all rows with any NaN values except the last row (for prediction)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Make last row target N/A for prediction/evaluation
    df.loc[df.index[-1], 'target'] = np.nan

    return df

def save_target(data):
    stocks_target = {}  # Set up data dictionary

    for key, df in data.items():
        pct_change = df['close'].pct_change()
        target = pct_change.iloc[-1]
        stocks_target[key] = target

    return stocks_target


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def predict_stocks_random_forest(data, days, threshold):
    stock_predictions = {}  # Set up data dictionary
    # Loop through data dictionary and predict each stock
    for key, df in data.items():
        # Exclude irrelevant features
        features = [col for col in data[key].columns if col not in ['symbol', 'timestamp', 'target']]

        X = data[key][features].values
        y = data[key]['target'].values

        # Train on days-1 amount of instances, predict one day's rerturn
        X_train = X[:days]
        y_train = y[:days]
        X_test = X[days:days + 1]
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        predicted_return = model.predict(X_test)

        if predicted_return >= threshold:
            adjusted_return = abs(predicted_return)  # "up" → positive magnitude
        else:
            adjusted_return = predicted_return  # "down" → negative magnitude

        stock_predictions[key] = adjusted_return

    return stock_predictions


from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # Import XGBoost
import pandas as pd
import numpy as np


def predict_stocks_xgboost(data, days):
    stock_predictions = {}

    # Loop through and predict each stock
    for key, df in data.items():
        # Exclude irrelevant features
        features = [col for col in df.columns if col not in ['symbol', 'timestamp', 'target']]

        X = df[features].values
        y = df['target'].values

        # Train on days-1 amount of instances, predict one day's rerturn
        X_train = X[:days]
        y_train = y[:days]
        X_test = X[days:days + 1]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBRegressor(
            objective='reg:squarederror',  # Standard objective for regression with squared loss
            n_estimators=100,  # Number of boosting rounds (trees)
            random_state=42,  # For reproducibility
            n_jobs=-1  # Use all available CPU cores
        )

        model.fit(X_train_scaled, y_train)
        stock_predictions[key] = model.predict(X_test)

    return stock_predictions
