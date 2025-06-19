from rich.progress import Progress
from rich.console import Console

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import argrelextrema

import time
import datetime

import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

console = Console()

def welcome_text():
    console.print("[bold][white]┌──────────────────────────[purple]──────────────────────────┐")
    console.print("[bold][white]│[purple]             _           [white]_                 _        [purple]│")
    console.print("[bold][white]│[purple]   ___  __ _| |__   __ _[white]| |_ _ __ __ _  __| | ___   [purple]│")
    console.print("[bold][white]│[purple]  / __|/ _` | '_ \\ / _` [white]| __| '__/ _` |/ _` |/ _ \\  [purple]│")
    console.print("[bold][purple]│[purple]  \\__ \\ (_| | |_) | (_| [white]| |_| | | (_| | (_| |  __/  [white]│")
    console.print("[bold][purple]│[purple]  |___/\\__,_|_.__/ \\__,_|[white]\\__|_|  \\__,_|\\__,_|\\___|  [white]│")
    console.print("[bold][purple]│                                                    [white]│")
    console.print("[bold][purple]└──────────────────────────[white]──────────────────────────┘\n")

def connect_to_exchange():
    console.print("[purple][bold]st[/bold] [white]► connecting to server")
    return ccxt.binance()

def fetch_data(symbol, timeframe, exchange, n):
    limit = 1000
    since = exchange.fetch_ohlcv(symbol, timeframe, limit=1)[0][0] - (5 * 60 * 1000 * limit * (n))

    ohlcv = []

    console.print("[purple][bold]st[/bold] [white]► fetching historical data")
    # Iterate to fetch historical price data
    with Progress() as progress:
        # Add a progress task
        task = progress.add_task("  progress:", total=n)

        for i in range(n):
            ohlcv_chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            # Append the new data to the all_data list
            ohlcv += ohlcv_chunk
            
            # Update 'since' for the next request (use the timestamp of the last candle)
            since = ohlcv_chunk[-1][0]  # The timestamp of the last candle

            # Print results
            #if ((i + 1) % 5 == 0):
            #    print(f"[INFO] Fetching Historical Data: {(i + 1) * 1000}/{n * 1000}")

            # Sleep to avoid hitting rate limits
            progress.update(task, advance=1)
            time.sleep(1)

    return ohlcv

def process_data(df, ):
    # Feature engineering: Create additional features (e.g., moving averages, RSI)
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

    bollinger = ta.bbands(df['4'], length=20, std=2)  # Default length is 20, std is 2
    df['BB_Middle'] = bollinger['BBM_20_2.0']  # Bollinger Middle Band
    df['BB_Upper'] = bollinger['BBU_20_2.0']   # Bollinger Upper Band
    df['BB_Lower'] = bollinger['BBL_20_2.0']   # Bollinger Lower Band

    df['smoothed_close_small'] = df['4'].rolling(window=25).mean()
    #df['smoothed_close_large'] = df[4].rolling(window=60).mean()
    df['KAMA'] = ta.kama(df['4'], length=10, fast=10, slow=50)
    df['EMA'] = ta.sma(df['4'], length=40, adjust=True)


    df['max'] = df['4'].iloc[argrelextrema(df['EMA'].values, np.greater_equal, order=10)[0]]
    df['min'] = df['4'].iloc[argrelextrema(df['EMA'].values, np.less_equal, order=10)[0]]

    df['target'] = 2  # Default is no action
    df.loc[df['min'].notna(), 'target'] = 1  # Buy at lows
    df.loc[df['max'].notna(), 'target'] = 0  # Sell at highs

    last = 2
    for i in range(len(df['target'])):
        if (df.loc[i, 'target']) == 1:
            last = 1
        elif (df.loc[i, 'target']) == 0:
            last = 0
        elif (df.loc[i, 'target']) == 2:
            (df.loc[i, 'target']) = last

    #print(df['target'])

    period = 1
    period_2 = 2
    df['returns'] = df['4'].pct_change(periods=-period)
    df['lagged'] = df['4'].shift(period)
    df['lagged_2'] = df['4'].shift(period_2)
    df['returns_lagged'] = df['4'].pct_change(periods=period)
    #df['lagged_forward'] = df[4].shift(-period)

    # Select features for the model
    features = ['1', '2', '3', '4', '5', 'lagged', 'lagged_2', 'RSI', 'ATR', 'CMO', 'CCI', 'ROC', 'SuperTrend_Direction', 'EMA', 'BB_Upper', 'BB_Lower']
    X = df[features].values
    y = df['target'].values

    return df, X, y

def load_data(prompt):
    skip = False
    ymt = False

    console.print("[purple][bold]"+ prompt +"[/bold] [white]► import data options:\n  1) load    2) fetch    3)    1 year - 1 month intervals 4) quit")
    try:
        data_todo = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
    except:
        data_todo = 3

    if data_todo == 1:
        df = pd.read_csv('data/100,000_1d.csv')
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► data loaded")
    elif data_todo == 2:
        n = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► kilocandles to regress: "))
        exchange = connect_to_exchange()
        ohlcv = fetch_data('DOGE/USDT', '1h', exchange, n)
        df = pd.DataFrame(ohlcv)
        df.to_csv(f'data/DOGE_1h.csv', index=False)
        df = pd.read_csv('data/tmp_1.csv')
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► data loaded")
    elif data_todo == 3:
        n = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► kilocandles to regress: "))
        exchange = connect_to_exchange()
        ohlcv = fetch_data('ETH/USDT', '5m', exchange, 108)
        df = pd.DataFrame(ohlcv)
        df.to_csv(f'data/1-year-5-min.csv', index=False)
        df = pd.read_csv('data/1-year-5-min.csv')
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► data loaded")
        ymt = True
    elif data_todo == 4:
        df = []
        skip = True

    return df, skip

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


def back_test(df, predictions, prompt, start, stop):
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

    # Backtest loop
    for i in range(len(predictions)):
        current_price = df['4'].iloc[start + i]  # Current price of the asset
        signal = predictions[i]  # Predicted signal (1 = Buy, -1 = Sell, 0 = Hold)

        if rsi_indicator[i] > 60.00:
            rsi_up = True
            rsi_down = False
        elif rsi_indicator[i] < 40.00:
            rsi_up = False
            rsi_down = True

        if signal == 1 and position == 0 and rsi_down == True and st_indicator[i] == 1:  # Buy signal
            position = 1
            entry_price = current_price
            balance -= balance * trading_fee  # Deduct trading fee for entering the position
            plt.scatter(actual_times[i], actual_prices[i], color='green', s=150, label='Buy Signal')

        elif signal == 0 and position == 1:  # Sell signal
            balance += ((current_price - entry_price) / entry_price) * balance  # Calculate profit/loss
            balance -= balance * trading_fee  # Deduct trading fee for exiting the position
            position = 0

            if (current_price - entry_price) > 0:
                good_trades.append((current_price - entry_price) / entry_price)
                console.print(f"[purple][bold]st[/bold] [white]► trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
                total_trades += 1
            elif (current_price - entry_price) <= 0:
                bad_trades.append((current_price - entry_price) / entry_price)
                console.print(f"[purple][bold]st[/bold] [white]► trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
                total_trades += 1

            plt.scatter(actual_times[i], actual_prices[i], color='red', s=150, label='Sell Signal')

        if position == 1 and (current_price - entry_price) / entry_price < -0.05:
            balance += ((current_price - entry_price) / entry_price) * balance  # Calculate profit/loss
            balance -= balance * trading_fee  # Deduct trading fee for exiting the position
            position = 0

            if (current_price - entry_price) > 0:
                good_trades.append((current_price - entry_price) / entry_price)
                console.print(f"[purple][bold]st[/bold] [white]► trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
                total_trades += 1
            elif (current_price - entry_price) <= 0:
                bad_trades.append((current_price - entry_price) / entry_price)
                console.print(f"[purple][bold]st[/bold] [white]► trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
                total_trades += 1

            plt.scatter(actual_times[i], actual_prices[i], color='red', s=150, label='Sell Signal')

    #if position == 1:
        #balance += (current_price - entry_price) / entry_price * balance
        #balance -= current_price * trading_fee  # Deduct trading fee for exiting the position
        #print(f"Final Sell: Closing position at {current_price:.2f}, Balance: {balance:.2f}")

    # Backtest summary
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► net profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► total trades: {total_trades}")
    if total_trades > 0:
        console.print("[purple][bold]"+ prompt +f"[/bold] [white]► accuracy: {(len(good_trades) / total_trades) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► average good trades: {np.mean(good_trades) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► average bad trades: {np.mean(bad_trades) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► timeframe: {(len(predictions) * 5) / 60 / 24:.2f} days")

    plt.plot(df.iloc[start:stop]['0'], df.iloc[start:stop]['4'], color='black', label='Data Points')
    #plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_small'], color='blue', label='Data Points')
    plt.plot(df.iloc[start:stop]['0'], df.iloc[start:stop]['EMA'], color='orange', label='Data Points')
    #plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_large'], color='red', label='Data Points')
    plt.scatter(df.iloc[start:stop]['0'], df.iloc[start:stop]['max'], color='orange', marker='^', label='Sell Signal')
    plt.scatter(df.iloc[start:stop]['0'], df.iloc[start:stop]['min'], color='purple', marker='^', label='Sell Signal')
    #plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_large'], color='purple', marker='^', label='Sell Signal')
    #plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_small'], color='green', marker='^', label='Sell Signal')
    #axs[1].plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['SuperTrend_Direction'], color='purple', label='Data Points')
    plt.tight_layout()
    plt.show()
