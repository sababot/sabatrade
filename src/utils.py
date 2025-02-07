from rich.progress import Progress
from rich.console import Console

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import argrelextrema

import time
import datetime

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
    df['EMA'] = ta.sma(df['4'], length=25, adjust=True)


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
    period_2 = 5
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

def load_data():
    skip = False

    console.print("[purple][bold]"+ prompt +"[/bold] [white]► import data options:\n  1) load    2) fetch    3) quit")
    try:
        data_todo = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
    except:
        data_todo = 3

    if data_todo == 1:
        df = pd.read_csv('data/100,000_5.csv')
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► data loaded")
    elif data_todo == 2:
        n = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► kilocandles to regress: "))
        exchange = utils.connect_to_exchange()
        ohlcv = utils.fetch_data('ETH/USDT', '5m', exchange, n)
        df = pd.DataFrame(ohlcv)
        df.to_csv(f'data/{n * 1000}_5.csv', index=False)
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► data loaded")
    elif data_todo == 3:
        skip = True

    return df, skip