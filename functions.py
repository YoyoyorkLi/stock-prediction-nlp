import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
API_KEY = "YOUR_API_KEY"  # Set up personal Alpaca API key before running the script
SECRET_KEY = "YOUR_SECRET_KEY"
BASE_URL = "https://paper-api.alpaca.markets"

def add_technical_features (df, window):

    # Log daily price change
    df['log_return'] = np.nan
    valid_mask = (df['close'] > 0) & (df['close'].shift(1) > 0)
    df.loc[valid_mask, 'log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Intraday move percentage
    df['pct_move_intraday'] = (df['close'] - df['open']) / df['open']

    # Daily average true range(ATR)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda row: max(
            row['high'] - row['low'],
            abs (row['high'] - row['prev_close']),
            abs (row['low'] - row['prev_close']),
        ), axis = 1
    )
    df['atr'] = df['tr'].rolling(window).mean()

    df.drop(columns = ['prev_close', 'tr'], inplace = True) # Delete columns used for calculation

    # Volatility: Rolling std dev of close price
    df[f'volatility_{window}'] = df['close'].rolling(window).std()

    # Relative Strength index(RSI)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD line and signal line
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Simple moving average (SMA)
    df[f'sma_{window}'] = df['close'].rolling(window).mean()

    # Exponential Moving Average (EMA)
    df[f'ema_{window}'] = df['close'].ewm(span = window, adjust = False).mean()

    # Volume spike ratio: volume vs rolling average
    df[f'volume_avg_{window}'] = df['volume'].rolling(window).mean()
    df['volume_spike'] = df['volume'] / (df[f'volume_avg_{window}'] + 1e-10) #avoid division by 0

    # Daily volume change
    df['volume_change_pct'] = df['volume'].pct_change()

    # VWAP deviation from close
    df['vwap_deviation'] = df['close'] - df['vwap']

    # Target column(percentage change from previous day)
    df['target'] = df['close'].pct_change()

    # Drop N/A and useless columns
    df.drop(columns = ['trade_count', 'symbol', 'timestamp'], inplace = True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

def add_lagged_features (df):
        # Lagged raw features
        for col in ['close', 'open', 'volume', 'vwap']:
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
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean().shift(1)
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean().shift(1)

        # Price ratios
        df['price_vs_sma5'] = df['close'] / df['sma_5']
        df['price_vs_sma10'] = df['close'] / df['sma_10']
        df['price_vs_vwap'] = df['close'] / df['vwap'].shift(1)

        # Volume features
        df['volume_avg_10'] = df['volume'].rolling(window=10).mean().shift(1)
        df['volume_spike'] = df['volume'] / df['volume_avg_10']
        df['volume_change_pct'] = df['volume'].pct_change(1).shift(1)

        # Volatility and range
        df['range'] = df['high'] - df['low']
        df['range_lag1'] = df['range'].shift(1)

        # MACD and signal
        ema_12_raw = df['close'].ewm(span=12, adjust=False).mean()
        ema_26_raw = df['close'].ewm(span=26, adjust=False).mean()
        macd_raw = ema_12_raw - ema_26_raw
        df['MACD'] = macd_raw
        df['MACD_signal'] = macd_raw.ewm(span=9, adjust=False).mean().shift(1)

        # RSI 14
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # Target: next day's return
        df['target'] = df['close'].pct_change().shift(-1)
        last_row = df.iloc[[-1]].copy()

        # Drop all rows with any NaN values except the last row (for prediction)
        df.dropna(inplace=True)
        df.reset_index(drop = True, inplace=True)

        # Add one last row with latest available features (but NaN target)
        df.loc[len(df)] = last_row.iloc[0]
    
        return df


def get_data(tickers, days):
    # Authenticate
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # Set up timeframe
    request_days = days*2
    training_days = days + 15
    end_date = datetime.now() - timedelta(days=1)  # Make end date yesterday to predict today
    start_date = end_date - timedelta(days=request_days)  # Request 500 days to ensure 250 days of trading day data

    # Set up Dataframe dictionary

    stocks_data = {}

    # Loop through the tickers and retrieve data separately

    for ticker in tickers:
        symbol = [ticker]  # Stock ticker
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,  # TimeFrame.Minute, .Hour, .Day, .Week, etc.
            start=start_date,  # Start date
            end=end_date  # End date
        )

        # Fetch the data

        bars = client.get_stock_bars(request_params)

        # Convert to DataFrame

        df = bars.df.reset_index()
        df = df.sort_values("timestamp")
        df = (
            df.groupby("symbol", group_keys=False)
            .tail(training_days)  # <-- Only save amount of training days as instances
            .reset_index(drop=True)
        )
        stocks_data[ticker] = df

    return stocks_data


def save_target(data):
    stocks_target = {}  # Set up data dictionary

    for key, df in data.items():
        pct_change = df['close'].pct_change()
        target = pct_change.iloc[-1]
        stocks_target[key] = target

    return stocks_target

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
def predict_stocks_random_forest(data, days):
    stock_predictions = {}  # Set up data dictionary
    # Loop through and predict each stock
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

        stock_predictions[key] = model.predict(X_test)

    return stock_predictions

from sklearn.preprocessing import StandardScaler
import xgboost as xgb # Import XGBoost
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
            objective='reg:squarederror', # Standard objective for regression with squared loss
            n_estimators=100,             # Number of boosting rounds (trees)
            random_state=42,              # For reproducibility
            n_jobs=-1                     # Use all available CPU cores
        )

        model.fit(X_train_scaled, y_train)
        stock_predictions[key] = model.predict(X_test)

    return stock_predictions
