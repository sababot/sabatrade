# donchian breakout + dual MA filter
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import matplot
import matplotlib.pyplot as plt

# import ohlcv data
df = pd.read_csv('data/BTC_1h.csv')

# functions
def donchian_channels(data, n):
    upper = data.shift(1).rolling(window=n).max()
    lower = data.shift(1).rolling(window=n).min()
    middle = (upper + lower) / 2

    return upper, middle, lower

# indicators
df['EMA_20'] = df['4'].ewm(span=20, adjust=True).mean()
df['EMA_50'] = df['4'].ewm(span=50, adjust=True).mean()
df['EMA_200'] = df['4'].ewm(span=200, adjust=True).mean()
df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = donchian_channels(df['4'], 20)

# backtest
trades = []
balance = 100.0
position = 0
entry_price = 0
entries = []
exits = []

for i in range(len(df['Donchian_Upper'])):
    if (df['4'].iloc[i] > df['Donchian_Upper'].iloc[i] and df['EMA_50'].iloc[i] < df['EMA_200'].iloc[i]) and position == 0:
        entry_price = df['4'].iloc[i]
        position = 1
        entries.append([df['0'].iloc[i], df['4'].iloc[i]])

    elif (df['4'].iloc[i] < df['Donchian_Lower'].iloc[i] or df['EMA_50'].iloc[i] > df['EMA_200'].iloc[i]) and position == 1:
        balance *= 1 + ((df['4'].iloc[i]  - entry_price) / entry_price)
        position = 0
        exits.append([df['0'].iloc[i], df['4'].iloc[i]])
        trades.append(((df['4'].iloc[i] - entry_price) / entry_price))

pnl = (((balance - 100) / 100)) * 100
trade_accuracy = (np.array(trades) > 0).sum() / len(np.array(trades)) * 100

# basic backtest results
print(f'pnl: {pnl: .1f}%')
print(f'number of trades: {len(trades)}')
print(f'trade accuracy: {trade_accuracy: .1f}%')
print(f'time period: {(len(df['Donchian_Upper']) * 15) / 60 / 24: .1f} days')

# price with entry and exit points
plt.figure
plt.plot(df['0'], df['4'], color='black', label="price")
plt.scatter(np.array(entries)[:, 0], np.array(entries)[:, 1], color="green", label="entries")
plt.scatter(np.array(exits)[:, 0], np.array(exits)[:, 1], color="red", label="exits")
plt.legend()
plt.show()

# cummulative profits graph
cum_returns = np.cumprod(np.array(1 + np.array(trades)))
plt.plot(cum_returns)
plt.show()
