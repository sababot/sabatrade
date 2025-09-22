# scripts
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src import utils

# libraries
from rich.progress import Progress
from rich.console import Console

import random

console = Console()

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
matplotlib.use('Qt5Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ccxt
import time
import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

#import tensorflow as tf
from xgboost import XGBClassifier
from scipy.signal import argrelextrema
import joblib

# hello world
console.print(f"[purple][bold]BUILDING STRATEGY:[/bold]")

#################### DATASET ####################
console.print("[purple][bold]► [/bold][white] loading dataset")
df = pd.read_csv('../../data/tmp_1.csv')

#################### FEATURES ####################
console.print("[purple][bold]► [/bold][white] processing dataset")
df['SMA_20'] = df['4'].rolling(window=20).mean()
df['SMA_50'] = df['4'].rolling(window=50).mean()
#df['RSI'] = 100 - (100 / (1 + df[4].pct_change().rolling(window=14).apply(lambda x: (x[x > 0].sum() / -x[x < 0].sum()) if x[x < 0].sum() != 0 else 0)))

df['RSI'] = ta.rsi(df['4'], length=14)
df['CMO'] = ta.cmo(df['4'], length=14)
df['ROC'] = ta.roc(df['4'], length=14)
df["CCI"] = ta.cci(df['2'], df['3'], df['4'], length=14)
df['ATR'] = ta.atr(df['2'], df['3'], df['4'])
supertrend = ta.supertrend(df['2'], df['3'], df['4'], length=10, multiplier=3)
df['SuperTrend'] = supertrend[f'SUPERT_10_3.0']  # SuperTrend column name format is SUPERT_length_multiplier
df['SuperTrend_Direction'] = supertrend[f'SUPERTd_10_3.0']  # Trend direction: 1 (bullish), -1 (bearish)

bollinger = ta.bbands(df['4'], length=15, std=2)  # Default length is 20, std is 2
df['BB_Middle'] = bollinger['BBM_15_2.0']  # Bollinger Middle Band
df['BB_Upper'] = bollinger['BBU_15_2.0']   # Bollinger Upper Band
df['BB_Lower'] = bollinger['BBL_15_2.0']   # Bollinger Lower Band

df['smoothed_close_small'] = df['4'].rolling(window=25).mean()
#df['smoothed_close_large'] = df[4].rolling(window=60).mean()
df['KAMA'] = ta.kama(df['4'], length=10, fast=10, slow=50)
df['EMA'] = ta.sma(df['4'], length=25, adjust=True)

df['DIST_MAX'] = 1000000000
df['DIST_MIN'] = 1000000000

df['max'] = df['4'].iloc[argrelextrema(df['EMA'].values, np.greater_equal, order=8)[0]]
df['min'] = df['4'].iloc[argrelextrema(df['EMA'].values, np.less_equal, order=8)[0]]

df['mean'] = df['4'].rolling(window=15).mean()
df['std'] = df['4'].rolling(window=15).std()

df['zscore'] = (df['4'] - df['mean']) / df['std']

df['target'] = 2  # Default is no action

for i in range(6, len(df['DIST_MIN'])):
    truncated_series_max = df['max'].iloc[:i-4]
    last_valid_index_max = truncated_series_max.last_valid_index()

    if last_valid_index_max is not None:
        df.loc[i, 'DIST_MAX'] = i - last_valid_index_max

    truncated_series_min = df['min'].iloc[:i-4]
    last_valid_index_min = truncated_series_min.last_valid_index()

    if last_valid_index_min is not None:
        df.loc[i, 'DIST_MIN'] = i - last_valid_index_min

period = 10
period_2 = 20
df['returns'] = df['EMA'].pct_change(periods=-period)
df['returns_lagged'] = df['EMA'].pct_change(periods=2)
df['lagged'] = df['EMA'].shift(1)
df['lagged_2'] = df['EMA'].shift(5)

df['target'] = np.where((df['returns'] > 0.005), 1,
                        np.where(df['returns'] < -0.005, 0, 2))

features = ['1', '2', '3', '4', '5', 'lagged', 'lagged_2', 'returns_lagged', 'zscore', 'RSI', 'ATR', 'EMA', 'BB_Middle', 'BB_Upper', 'BB_Lower']
X = df[features].values
y = df['target'].values

#################### XGBOOST MODEL ####################
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, shuffle=False)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)  # Transform test set without fitting again

console.print("[purple][bold]► [/bold][white] creating model")
console.print("[purple][bold]► [/bold][white] using model settings 'model.conf'")

# define and train model
model = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=4)
model.fit(X_train, y_train)

# obtain the predictions
predictions = model.predict(X_test)

# output accuracy of the model
accuracy = accuracy_score(y_test, predictions)
console.print(f"[purple][bold]► [/bold][white] model accuracy: {accuracy * 100:.2f}%")

# save model
joblib.dump(model, '../../models/xgboost_model.joblib')

# load model
loaded = True
console.print("[purple][bold]► [/bold][white] model loaded")

#################### BACKTEST ####################
signals = utils.generate_signals(predictions)
long_trades = []
short_trades = []
long_price = None
short_price = None
isLong = False
isShort = False
sltp = 0.10

for i in range(len(signals)):
    if signals[i] == 1:
        long_price = X_test[i, 3]
        isLong = True

        if isShort:
            short_trades.append(-(X_test[i, 3] - short_price) / short_price)
            isShort = False

    elif signals[i] == -1:
        short_price = X_test[i, 3]
        isShort = True

        if isLong:
            long_trades.append((X_test[i, 3] - long_price) / long_price)
            isLong = False

    elif isLong and abs((X_test[i, 3] - long_price) / long_price) > sltp and long_price != None:
        long_trades.append((X_test[i, 3] - long_price) / long_price)
        isLong = False
        isShort = False

    elif isShort and abs(-(X_test[i, 3] - short_price) / short_price) > sltp and short_price != None:
        short_trades.append(-(X_test[i, 3] - short_price) / short_price)
        isLong = False
        isShort = False

long_pnl = utils.get_pnl(long_trades)
short_pnl = utils.get_pnl(short_trades)
total_pnl = utils.get_pnl(np.concatenate((np.array(long_trades), np.array(short_trades))))

total_trades = np.concatenate((np.array(long_trades), np.array(short_trades)))
np_total_trades = np.array(total_trades)
np_long_trades = np.array(long_trades)
np_short_trades = np.array(short_trades)
total_trade_accuracy = percentage_positive = np.sum(np_total_trades > 0) / len(np_total_trades)
long_trade_accuracy = percentage_positive = np.sum(np_long_trades > 0) / len(np_long_trades)
short_trade_accuracy = percentage_positive = np.sum(np_short_trades > 0) / len(np_short_trades)
sharpe_ratio = np.mean(total_trades) / np.std(total_trades) * np.sqrt(200)
profit_factor = np.sum(np_total_trades[np_total_trades > 0]) / -np.sum(np_total_trades[np_total_trades < 0])
max_drawdown = np.min((np.cumprod(1 + np_total_trades) - np.maximum.accumulate(np.cumprod(1 + np_total_trades))) / np.maximum.accumulate(np.cumprod(1 + np_total_trades)))

# backtest retults
console.print(f"\n[purple][bold]BACKTEST RESULTS:[/bold]")
console.print(f"[purple][bold]► [/bold][white] total trades:        {len(total_trades): .0f}")
console.print(f"[purple][bold]► [/bold][white] trade accuracy:      {long_trade_accuracy * 100: .2f}% {short_trade_accuracy * 100: .2f}% {total_trade_accuracy * 100: .2f}%")
console.print(f"[purple][bold]► [/bold][white] pnl:                 {long_pnl * 100: .2f}% {short_pnl * 100: .2f}% {total_pnl * 100: .2f}%")
console.print(f"[purple][bold]► [/bold][white] max drawdown:        {max_drawdown * 100: .2f}%")
console.print(f"[purple][bold]► [/bold][white] profit factor:       {profit_factor: .2f}")
console.print(f"[purple][bold]► [/bold][white] interval:            {len(signals) / 24: .1f} days")

#################### PLOT ####################
# BTC close long and short positions
df['date'] = pd.to_datetime(df['0'], unit='ms')
plt.style.use('dark_background')
plt.figure(figsize=(14, 6))
plt.plot(df.iloc[len(X_train):len(X_train) + len(X_test)]['date'], df.iloc[len(X_train):len(X_train) + len(X_test)]['4'], color='white')

signals = np.array(signals)

buy_indices = np.where(signals == 1)[0]
plt.scatter(df.iloc[len(X_train) + buy_indices]['date'], df.iloc[len(X_train) + buy_indices]['4'], label='Long Position', marker='^', color='green', s=100, zorder=2)

sell_indices = np.where(signals == -1)[0]
plt.scatter(df.iloc[len(X_train) + sell_indices]['date'], df.iloc[len(X_train) + sell_indices]['4'], label='Short Position', marker='v', color='red', s=100, zorder=2)

plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

# cummulative profits
cum_returns = np.cumprod(1 + np_total_trades)

df['date'] = pd.to_datetime(df['0'], unit='ms')
plt.style.use('dark_background')
plt.figure(figsize=(14, 6))
plt.plot(cum_returns, color='white')

plt.xlabel('Trades')
plt.ylabel('Cummulative Profit')
plt.grid(True)
plt.tight_layout()
plt.show()




'''
# back-test variables
start = len(X_train)
stop = len(X_train) + len(X_test)

initial_balance = 100  # Starting capital in USD
balance = initial_balance
position = 0  # 1 = long, -1 = short, 0 = no position
entry_price = 0 # Price at which the position is entered
trading_fee = 0.001 # 0.075% trading fee per transaction
good_trades = []
bad_trades = []
total_trades = 0

plt.figure(figsize=(14, 7))
actual_times = df['0'].iloc[start:stop].values
actual_prices = df['4'].iloc[start:stop].values
rsi_indicator = df['RSI'].iloc[start:stop].values
st_indicator = df['SuperTrend_Direction'].iloc[start:stop].values

rsi_up = False
rsi_down = False

console.print("[purple][bold]► [/bold][white] trades:")

# Backtest loop
for i in range(len(predictions)):
    current_price = df['4'].iloc[start + i]  # Current price of the asset
    signal = signals[i]  # Predicted signal (1 = Buy, -1 = Sell, 0 = Hold)

    if rsi_indicator[i] > 60.00:
        rsi_up = True
        rsi_down = False
    elif rsi_indicator[i] < 40.00:
        rsi_up = False
        rsi_down = True

    if position == 0 and signal == 1:  # Buy signal
        position = 1
        entry_price = current_price
        balance -= balance * trading_fee  # Deduct trading fee for entering the position

        plt.scatter(actual_times[i], actual_prices[i], color='green', s=150, label='Buy Signal')

    elif position == 1 and (signal == -1 or abs((current_price - entry_price) / entry_price) > 1): # Sell signal
        balance += ((current_price - entry_price) / entry_price) * balance  # Calculate profit/loss
        balance -= balance * trading_fee  # Deduct trading fee for exiting the position
        position = 0

        if (current_price - entry_price) > 0:
            good_trades.append((current_price - entry_price) / entry_price)
            console.print(f"[white]trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
            total_trades += 1
        elif (current_price - entry_price) <= 0:
            bad_trades.append((current_price - entry_price) / entry_price)
            console.print(f"[white]trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
            total_trades += 1

        plt.scatter(actual_times[i], actual_prices[i], color='red', s=150, label='Sell Signal')

    

#if position == 1:
    #balance += (current_price - entry_price) / entry_price * balance
    #balance -= current_price * trading_fee  # Deduct trading fee for exiting the position
    #print(f"Final Sell: Closing position at {current_price:.2f}, Balance: {balance:.2f}")

# Backtest summary
console.print(f"[purple][bold]► [/bold][white] net profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%")
console.print(f"[purple][bold]► [/bold][white] total trades: {total_trades}")
if total_trades > 0:
    console.print(f"[purple][bold]► [/bold][white] trade accuracy: {(len(good_trades) / total_trades) * 100: .2f}%")
console.print(f"[purple][bold]► [/bold][white] average good trades: {np.mean(good_trades) * 100: .2f}%")
console.print(f"[purple][bold]► [/bold][white] average bad trades: {np.mean(bad_trades) * 100: .2f}%")
console.print(f"[purple][bold]► [/bold][white] timeframe: {len(predictions) / 24:.2f} days")

plt.plot(df.iloc[start:stop]['0'], df.iloc[start:stop]['4'], color='black', label='Data Points')
#plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_small'], color='blue', label='Data Points')
###plt.plot(df.iloc[start:stop]['0'], df.iloc[start:stop]['EMA'], color='orange', label='Data Points')
#plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_large'], color='red', label='Data Points')
#plt.scatter(df.iloc[start:stop]['0'], df.iloc[start:stop]['max'], color='orange', marker='^', label='Sell Signal')
#plt.scatter(df.iloc[start:stop]['0'], df.iloc[start:stop]['min'], color='purple', marker='^', label='Sell Signal')
#plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_large'], color='purple', marker='^', label='Sell Signal')
#plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_small'], color='green', marker='^', label='Sell Signal')
#axs[1].plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['SuperTrend_Direction'], color='purple', label='Data Points')
plt.tight_layout()
plt.show()
'''
