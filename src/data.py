import time

import numpy as np
import pandas as pd
import pandas_ta as ta
from rich.progress import Progress
from scipy.signal import argrelextrema

from common import console
from exchange import connect_to_exchange


def fetch_data(symbol, timeframe, exchange, n):
    limit = 1000
    since = exchange.fetch_ohlcv(symbol, timeframe, limit=1)[0][0] - (
        5 * 60 * 1000 * limit * (n)
    )

    ohlcv = []

    console.print("[purple][bold]st[/bold] [white]► fetching historical data")
    with Progress() as progress:
        task = progress.add_task("  progress:", total=n)

        for _ in range(n):
            ohlcv_chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            ohlcv += ohlcv_chunk
            since = ohlcv_chunk[-1][0]
            progress.update(task, advance=1)
            time.sleep(1)

    return ohlcv


def process_data(df):
    # Feature engineering
    df["SMA_20"] = df["4"].rolling(window=20).mean()
    df["SMA_50"] = df["4"].rolling(window=50).mean()

    df["RSI"] = ta.rsi(df["4"], length=14)
    df["CMO"] = ta.cmo(df["4"], length=14)
    df["ROC"] = ta.roc(df["4"], length=14)
    df["CCI"] = ta.cci(df["2"], df["3"], df["4"], length=14)
    df["ATR"] = ta.atr(df["2"], df["3"], df["4"])

    supertrend = ta.supertrend(df["2"], df["3"], df["4"], length=10, multiplier=3)
    df["SuperTrend"] = supertrend["SUPERT_10_3"]
    df["SuperTrend_Direction"] = supertrend["SUPERTd_10_3"]

    bollinger = ta.bbands(df["4"], length=20, std=2)
    df["BB_Middle"] = bollinger["BBM_20_2.0_2.0"]
    df["BB_Upper"] = bollinger["BBU_20_2.0_2.0"]
    df["BB_Lower"] = bollinger["BBL_20_2.0_2.0"]

    df["smoothed_close_small"] = df["4"].rolling(window=25).mean()
    df["KAMA"] = ta.kama(df["4"], length=10, fast=10, slow=50)
    df["EMA"] = ta.sma(df["4"], length=40, adjust=True)

    df["max"] = df["4"].iloc[argrelextrema(df["EMA"].values, np.greater_equal, order=10)[0]]
    df["min"] = df["4"].iloc[argrelextrema(df["EMA"].values, np.less_equal, order=10)[0]]

    df["target"] = 2
    df.loc[df["min"].notna(), "target"] = 1
    df.loc[df["max"].notna(), "target"] = 0

    last = 2
    for i in range(len(df["target"])):
        if df.loc[i, "target"] == 1:
            last = 1
        elif df.loc[i, "target"] == 0:
            last = 0
        elif df.loc[i, "target"] == 2:
            df.loc[i, "target"] = last

    period = 1
    period_2 = 2
    df["returns"] = df["4"].pct_change(periods=-period)
    df["lagged"] = df["4"].shift(period)
    df["lagged_2"] = df["4"].shift(period_2)
    df["returns_lagged"] = df["4"].pct_change(periods=period)

    features = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "lagged",
        "lagged_2",
        "RSI",
        "ATR",
        "CMO",
        "CCI",
        "ROC",
        "SuperTrend_Direction",
        "EMA",
        "BB_Upper",
        "BB_Lower",
    ]
    X = df[features].values
    y = df["target"].values

    return df, X, y


def load_data(prompt):
    skip = False
    ymt = False

    console.print(
        "[purple][bold]"
        + prompt
        + "[/bold] [white]► import data options:\n  1) load    2) fetch    3)    1 year - 1 month intervals 4) quit"
    )
    try:
        data_todo = int(console.input("[purple][bold]" + prompt + "[/bold] [white]► "))
    except Exception:
        data_todo = 3

    if data_todo == 1:
        df = pd.read_csv("data/100,000_1d.csv")
        console.print("[purple][bold]" + prompt + "[/bold] [white]► data loaded")
    elif data_todo == 2:
        n = int(console.input("[purple][bold]" + prompt + "[/bold] [white]► kilocandles to regress: "))
        exchange = connect_to_exchange()
        ohlcv = fetch_data("BTC/USDT", "1h", exchange, n)
        df = pd.DataFrame(ohlcv)
        df.to_csv("data/BTC_1h.csv", index=False)
        df = pd.read_csv("data/tmp_1.csv")
        console.print("[purple][bold]" + prompt + "[/bold] [white]► data loaded")
    elif data_todo == 3:
        _ = int(console.input("[purple][bold]" + prompt + "[/bold] [white]► kilocandles to regress: "))
        exchange = connect_to_exchange()
        ohlcv = fetch_data("ETH/USDT", "5m", exchange, 108)
        df = pd.DataFrame(ohlcv)
        df.to_csv("data/1-year-5-min.csv", index=False)
        df = pd.read_csv("data/1-year-5-min.csv")
        console.print("[purple][bold]" + prompt + "[/bold] [white]► data loaded")
        ymt = True
    elif data_todo == 4:
        df = []
        skip = True
    else:
        df = []
        skip = True

    return df, skip, ymt


