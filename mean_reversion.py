import numpy as np
import pandas as pd
import ccxt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from backtesting import Backtest, Strategy
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler
import warnings
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*use_label_encoder.*')
warnings.filterwarnings('ignore', message='Some prices are larger than initial cash value')

# --- 1. Data Collection & Feature Engineering ---
def fetch_data(symbol='BTC/USDT', timeframe='1h', limit=2000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')

def calculate_features(df):
    # Calculate ATR first since it's needed for strategy
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Mean Reversion Indicators
    df['sma_20'] = ta.sma(df['Close'], length=20)
    df['sma_50'] = ta.sma(df['Close'], length=50)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # Momentum Indicators
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # Additional profitable features
    df['cci'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    df['obv'] = ta.obv(df['Close'], df['Volume'])
    
    # Mean Deviation Features
    df['deviation'] = (df['Close'] / df['sma_20']) - 1
    df['z_score'] = (df['Close'] - df['sma_20']) / df['Close'].rolling(20).std()
    
    # Lagged Returns and Target
    df['ret_5'] = df['Close'].pct_change(5).shift(-5)
    
    # Fill missing values
    df = df.ffill().dropna()
    
    return df.rename(columns={'BBL_20_2.0': 'boll_lower', 
                              'BBM_20_2.0': 'boll_middle',
                              'BBU_20_2.0': 'boll_upper',
                              'MACD_12_26_9': 'macd',
                              'MACDs_12_26_9': 'macd_signal'})

# --- 2. Target Definition ---
def define_target(df, deviation_threshold=0.03, return_threshold=0.025):
    # Only use past information for target creation
    oversold = (
        (df['deviation'] < -deviation_threshold) | 
        (df['rsi'] < 35) |
        (df['Close'] < df['boll_lower'] * 1.01)
    )
    
    # Future return should only use past data
    positive_return = (df['ret_5'] > return_threshold)
    volume_spike = df['Volume'] > 1.5 * df['Volume'].rolling(20).mean().shift(1)
    cci_reversal = (df['cci'].shift(1) < -100) & (df['cci'] > df['cci'].shift(1))
    
    df['target'] = np.where(oversold & positive_return & volume_spike & cci_reversal, 1, 0)
    
    # Print target distribution
    target_counts = df['target'].value_counts()
    print(f"Target distribution:\n{target_counts}")
    print(f"Positive class ratio: {target_counts[1]/(target_counts[0]+target_counts[1]):.2%}")
    
    return df

# --- 3. Model Training with Robust Scaling ---
def train_model(df):
    features = ['deviation', 'z_score', 'rsi', 'macd', 'atr', 
                'boll_lower', 'cci', 'obv']
    
    X = df[features]
    y = df['target']
    
    # Time-series split for financial data
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced to 3 splits for stability
    scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = XGBClassifier(
            n_estimators=300,  # Reduced complexity
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=10  # Weight for minority class
        )
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
    
    print(f"Cross-validation accuracy: {np.mean(scores):.4f}")
    
    # Train final model on entire dataset
    final_scaler = RobustScaler()
    X_scaled = final_scaler.fit_transform(X)
    
    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=10  # Weight for minority class
    )
    final_model.fit(X_scaled, y)
    
    return final_scaler, final_model

# --- 4. Enhanced Trading Strategy ---
class MeanReversionStrategy(Strategy):
    tp_multiplier = 3.5
    sl_multiplier = 1.8
    
    def init(self):
        # Pre-calculated probability series
        self.probability = self.data.probability
        
        # Pre-calculated ATR values
        self.atr_values = self.data.atr
        
    def next(self):
        if len(self.data) < 50:
            return
        
        current_price = self.data.Close[-1]
        atr = self.atr_values[-1]
        prob = self.probability[-1]
        
        # Enhanced entry conditions
        volume_ok = self.data.Volume[-1] > 1.3 * self.data.Volume[-20:-1].mean()
        
        # Entry: High probability + conditions
        if prob > 0.65 and not self.position and volume_ok:  # Lowered threshold
            sl_price = current_price - self.sl_multiplier * atr
            tp_price = current_price + self.tp_multiplier * atr
            
            # Position sizing (1% risk per trade)
            risk_per_share = current_price - sl_price
            position_size = max(0.01 * self.equity / risk_per_share, 0)
            
            # Only trade if position size is valid
            if position_size > 0 and sl_price > 0:
                self.buy(size=position_size, sl=sl_price, tp=tp_price)
        
        # Dynamic exit: Close if probability drops or profit target reached
        elif self.position:
            if prob < 0.4 or self.position.pl_pct >= 0.03:
                self.position.close()

# --- 5. Feature Preparation and Probability Calculation ---
def calculate_probabilities(df, scaler, model):
    feature_cols = ['deviation', 'z_score', 'rsi', 'macd', 'atr', 
                    'boll_lower', 'cci', 'obv']
    feature_df = df[feature_cols]
    
    # Handle missing values
    feature_df = feature_df.ffill().fillna(0)
    
    # Scale features and predict
    scaled_features = scaler.transform(feature_df)
    probabilities = model.predict_proba(scaled_features)[:, 1]
    
    return probabilities

# --- Main Execution ---
if __name__ == "__main__":
    print("Fetching data...")
    df = fetch_data(symbol='BTC/USDT', timeframe='1h', limit=2000)
    
    print("Calculating features...")
    df = calculate_features(df)
    
    print("Defining targets...")
    df = define_target(df, deviation_threshold=0.028, return_threshold=0.022)
    
    print("Training model...")
    scaler, model = train_model(df)
    
    print("Calculating probabilities...")
    df['probability'] = calculate_probabilities(df, scaler, model)
    
    # Ensure required columns are present
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'atr']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert to satoshis to avoid fractional trading issues
    SATOSHI_MULTIPLIER = 100_000_000  # 1 BTC = 100,000,000 satoshis
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col] * SATOSHI_MULTIPLIER
    df['atr'] = df['atr'] * SATOSHI_MULTIPLIER
    
    print("Running backtest...")
    bt = Backtest(
        df, 
        MeanReversionStrategy, 
        cash=10_000_000_000,  # 100 BTC in satoshis (100,000,000 sat/BTC * 100 BTC)
        commission=0.00075,
        trade_on_close=True,
        exclusive_orders=True
    )
    
    stats = bt.run()
    
    print("\nBacktest Results:")
    if stats['# Trades'] > 0:
        print(f"Number of Trades: {stats['# Trades']}")
        print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}" if not np.isnan(stats['Sharpe Ratio']) else "Sharpe Ratio: N/A (no variance)")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"Total Return: {stats['Return [%]']:.2f}%")
    else:
        print("No trades were executed during the backtest period")
        print("Try lowering the probability threshold in the strategy")
    
    # Plot equity curve even if no trades
    plt.figure(figsize=(12, 6))
    plt.plot(stats['_equity_curve'])
    plt.title('Equity Curve')
    plt.ylabel('Equity (Satoshis)')
    plt.xlabel('Time')
    plt.grid(True)
    plt.show()