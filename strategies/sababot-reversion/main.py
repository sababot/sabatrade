import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import time
import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# functions
def generate_signals(predictions):
    position = 0
    signals = []

    for i in range(len(predictions)):
        if position == 0 and predictions[i] == 1:
            signals.append(1)
            position = 1

        elif position == 1 and predictions[i] == 0:
            signals.append(-1)
            position = 0
        
        else:
            signals.append(0)

    return signals

def get_pnl(trades):
    multipliers = 1 + np.array(trades)
    initial = 100.0
    final = initial * np.prod(multipliers)
    return (final - initial) / initial

# import dataset
df = pd.read_csv('data/1,000,000_15.csv')

# add features and targets
df['EMA_25'] = df['4'].ewm(span=25, adjust=True).mean()
df['EMA_50'] = df['4'].ewm(span=50, adjust=True).mean()
df['EMA_200'] = df['4'].ewm(span=200, adjust=True).mean()

df['mean'] = df['4'].rolling(window=15).mean()
df['std'] = df['4'].rolling(window=15).std()
df['zscore'] = (df['4'] - df['mean']) / df['std']

df['RSI'] = ta.rsi(df['4'], length=14)
df['ATR'] = ta.atr(df['2'], df['3'], df['4'])

bollinger = ta.bbands(df['4'], length=15, std=2)
df['BB_Lower'] = bollinger['BBL_15_2.0']
df['BB_Middle'] = bollinger['BBM_15_2.0']
df['BB_Upper'] = bollinger['BBU_15_2.0']

df['date'] = pd.to_datetime(df['0'], unit='ms')
df['lagged_1'] = df['EMA_25'].shift(1)
df['lagged_2'] = df['EMA_25'].shift(5)
df['lagged_returns'] = df['EMA_25'].pct_change(periods=2)

df['returns'] = df['EMA_25'].pct_change(periods=-10)
df['target'] = np.where((df['returns'] > 0.005), 1,
                        np.where(df['returns'] < -0.005, 0, 2))

X = df[['1', '2', '3', '4', '5',
	'lagged_1', 'lagged_2', 'lagged_returns',
	'zscore', 'RSI', 'ATR', 'EMA_25',
	'BB_Lower', 'BB_Middle', 'BB_Upper']].values
y = df['target'].values

# train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

model = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=4)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

model_accuracy = accuracy_score(y_test, predictions)

# backtest
signals = generate_signals(predictions)
long_trades, short_trades = [], []
realized_signals = np.zeros(len(signals))
long_price, short_price = None, None
isLong, isShort = False, False
sltp = 0.10

for i in range(len(signals)):
    if signals[i] == 1 and df['EMA_50'].iloc[len(X_train) + i] < df['EMA_200'].iloc[len(X_train) + i] and isLong == False:
        long_price = X_test[i, 3]
        realized_signals[i] = 1
        isLong = True

        if isShort:
            short_trades.append(-(X_test[i, 3] - short_price) / short_price)
            isShort = False

    elif signals[i] == -1 and df['EMA_50'].iloc[len(X_train) + i] > df['EMA_200'].iloc[len(X_train) + i] and isShort == False:
        short_price = X_test[i, 3]
        realized_signals[i] = 1
        isShort = True

        if isLong:
            long_trades.append((X_test[i, 3] - long_price) / long_price)
            realized_signals[i] = -1
            isLong = False

    elif isLong and abs((X_test[i, 3] - long_price) / long_price) > sltp and long_price != None and isLong:
        long_trades.append((X_test[i, 3] - long_price) / long_price)
        realized_signals[i] = -1
        isLong = False
        isShort = False

    elif isShort and abs(-(X_test[i, 3] - short_price) / short_price) > sltp and short_price != None and isLong:
        short_trades.append(-(X_test[i, 3] - short_price) / short_price)
        realized_signals[i] = -1
        isLong = False
        isShort = False

# backtest results
long_pnl = get_pnl(long_trades)
short_pnl = get_pnl(short_trades)
total_pnl = get_pnl(np.concatenate((np.array(long_trades), np.array(short_trades))))

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
cum_returns = np.cumprod(1 + np_total_trades)

print(f"model accuracy:      {model_accuracy * 100: .1f}%")
print(f"total trades:        {len(total_trades): .0f}")
print(f"trade accuracy:      {long_trade_accuracy * 100: .2f}% {short_trade_accuracy * 100: .2f}% {total_trade_accuracy * 100: .2f}%")
print(f"pnl:                 {long_pnl * 100: .2f}% {short_pnl * 100: .2f}% {total_pnl * 100: .2f}%")
print(f"max drawdown:        {max_drawdown * 100: .2f}%")
print(f"profit factor:       {profit_factor: .2f}")
print(f"interval:            {(len(signals) * 15) / (60 * 24): .1f} days")

# plot
fig, axs = plt.subplots(2, 1, figsize=(14, 6))

axs[0].plot(df.iloc[len(X_train):len(X_train) + len(X_test)]['date'], df.iloc[len(X_train):len(X_train) + len(X_test)]['4'], color='black')

buy_indices = np.where(realized_signals == 1)[0]
axs[0].scatter(df.iloc[len(X_train) + buy_indices]['date'], df.iloc[len(X_train) + buy_indices]['4'], label='Long Position', marker='^', color='green', s=80, zorder=2)

sell_indices = np.where(realized_signals == -1)[0]
axs[0].scatter(df.iloc[len(X_train) + sell_indices]['date'], df.iloc[len(X_train) + sell_indices]['4'], label='Short Position', marker='v', color='red', s=80, zorder=2)

axs[0].set_xlabel('Date')
axs[0].set_ylabel('Price (USDT)')
axs[0].legend()

axs[1].plot(cum_returns, color='black')
axs[1].set_xlabel('Trades')
axs[1].set_ylabel('Cummulative Profit')

plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
plt.style.use('dark_background')
plt.tight_layout()
plt.grid(True)
plt.show()