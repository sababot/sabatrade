import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import console
from data import fetch_data, process_data
from exchange import connect_to_exchange


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


def generate_volume_news_signals(df, start_index=100, lookback=60):
    if len(df) < start_index + lookback + 12:
        return []

    df["VOL_MA"] = df["5"].rolling(window=lookback).mean()
    df["RET_12"] = df["4"].pct_change(periods=12)

    df["News_Sentiment"] = 0
    df.loc[df["RET_12"] > 0.01, "News_Sentiment"] = 1
    df.loc[df["RET_12"] < -0.01, "News_Sentiment"] = -1

    predictions = []

    for i in range(start_index, len(df)):
        vol = df["5"].iloc[i]
        vol_ma = df["VOL_MA"].iloc[i]

        if "News_Sentiment_Real" in df.columns:
            sentiment = df["News_Sentiment_Real"].iloc[i]
        else:
            sentiment = df["News_Sentiment"].iloc[i]

        trend = (
            df["SMA_20"].iloc[i]
            if "SMA_20" in df.columns
            else (df["EMA"].iloc[i] if "EMA" in df.columns else df["4"].iloc[i])
        )
        rsi = df["RSI"].iloc[i] if "RSI" in df.columns else 50

        if pd.isna(vol_ma):
            predictions.append(2)
            continue

        bullish = (vol > 1.5 * vol_ma) and (sentiment == 1) and (df["4"].iloc[i] > trend)
        bearish = (sentiment == -1) or (df["4"].iloc[i] < trend) or (rsi > 75)

        if bullish:
            predictions.append(1)
        elif bearish:
            predictions.append(0)
        else:
            predictions.append(2)

    return predictions


def generate_altcoin_signals(df, start_index=100, lookback=60):
    if len(df) < start_index + lookback + 12:
        return []

    predictions = []
    for i in range(start_index, len(df)):
        rsi = df["RSI"].iloc[i] if "RSI" in df.columns else 50
        price = df["4"].iloc[i]
        bb_lower = df["BB_Lower"].iloc[i] if "BB_Lower" in df.columns else 0
        bb_upper = df["BB_Upper"].iloc[i] if "BB_Upper" in df.columns else float("inf")

        bullish = (rsi < 30) or (price < bb_lower)
        bearish = (rsi > 70) or (price > bb_upper)

        if bullish:
            predictions.append(1)
        elif bearish:
            predictions.append(0)
        else:
            predictions.append(2)

    return predictions


def build_volume_news_ai_features(df, lookback=60, horizon=12, future_threshold=0.0):
    df_feat = df.copy()
    df_feat["VOL_MA"] = df_feat["5"].rolling(window=lookback).mean()
    df_feat["VOL_RATIO"] = df_feat["5"] / df_feat["VOL_MA"]

    df_feat["RET_12"] = df_feat["4"].pct_change(periods=horizon)
    df_feat["News_Sentiment_Proxy"] = 0
    df_feat.loc[df_feat["RET_12"] > 0.01, "News_Sentiment_Proxy"] = 1
    df_feat.loc[df_feat["RET_12"] < -0.01, "News_Sentiment_Proxy"] = -1

    if "News_Sentiment_Real" in df_feat.columns:
        df_feat["News_Sentiment_Feature"] = df_feat["News_Sentiment_Real"]
    else:
        df_feat["News_Sentiment_Feature"] = df_feat["News_Sentiment_Proxy"]

    df_feat["FUT_RET"] = df_feat["4"].pct_change(periods=horizon).shift(-horizon)
    df_feat["label"] = (df_feat["FUT_RET"] > future_threshold).astype(int)

    feature_cols = [
        "VOL_RATIO",
        "News_Sentiment_Feature",
        "RSI",
        "SuperTrend_Direction",
        "ATR",
        "ROC",
        "CCI",
        "BB_Upper",
        "BB_Lower",
    ]

    data = df_feat[feature_cols + ["label"]].dropna()
    X = data[feature_cols].values
    y = data["label"].values
    idx = data.index
    return X, y, idx


def attach_real_news_sentiment(df, symbol, tolerance_ms=60 * 60 * 1000):
    path = f"data/news_{symbol}.csv"
    if not os.path.exists(path):
        return df

    try:
        news = pd.read_csv(path)
    except Exception:
        return df

    if "timestamp" not in news.columns or "sentiment" not in news.columns:
        return df

    news = news.sort_values("timestamp")
    df_local = df.copy()
    df_local = df_local.sort_values("0")

    try:
        merged = pd.merge_asof(
            df_local,
            news[["timestamp", "sentiment"]].sort_values("timestamp"),
            left_on="0",
            right_on="timestamp",
            direction="backward",
            tolerance=tolerance_ms,
        )
    except Exception:
        return df

    df_local["News_Sentiment_Real"] = merged["sentiment"].fillna(0)
    df_local = df_local.sort_index()
    return df_local


def train_ai_filter(df, lookback=60, horizon=12, future_threshold=0.0):
    X, y, idx = build_volume_news_ai_features(
        df, lookback=lookback, horizon=horizon, future_threshold=future_threshold
    )
    if len(X) < 200:
        return None, None

    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(n_estimators=200, learning_rate=0.02, max_depth=4)
    model.fit(X_train, y_train)

    proba = model.predict_proba(scaler.transform(X))[:, 1]
    return proba, idx


def apply_ai_filter(base_signals, start_index, proba_map, buy_thresh=0.55, sell_thresh=0.45):
    if proba_map is None:
        return base_signals

    final = []
    for offset, base in enumerate(base_signals):
        idx = start_index + offset
        p = proba_map.get(idx)
        if p is None:
            final.append(base)
            continue

        if base == 1 and p > buy_thresh:
            final.append(1)
        elif base == 0 and p < sell_thresh:
            final.append(0)
        else:
            final.append(2)
    return final


def back_test(df, predictions, prompt, start, stop, block=True):
    initial_balance = 100
    balance = initial_balance
    position = 0
    entry_price = 0
    trading_fee = 0.001
    slippage = 0.0005
    good_trades = []
    bad_trades = []
    total_trades = 0

    equity_curve = [balance]

    plt.figure(figsize=(14, 7))
    actual_times = df["0"].iloc[start:stop].values
    actual_prices = df["4"].iloc[start:stop].values
    rsi_indicator = df["RSI"].iloc[start:stop].values
    st_indicator = df["SuperTrend_Direction"].iloc[start:stop].values

    rsi_up = False
    rsi_down = False

    for i in range(len(predictions)):
        current_price = df["4"].iloc[start + i]
        signal = predictions[i]

        if rsi_indicator[i] > 60.00:
            rsi_up = True
            rsi_down = False
        elif rsi_indicator[i] < 40.00:
            rsi_up = False
            rsi_down = True

        if signal == 1 and position == 0 and rsi_down is True and st_indicator[i] == 1:
            position = 1
            entry_price = current_price * (1 + slippage)
            balance -= balance * trading_fee
            plt.scatter(actual_times[i], actual_prices[i], color="green", s=150, label="Buy Signal")

        elif signal == 0 and position == 1:
            exit_price = current_price * (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price
            balance += trade_return * balance
            balance -= balance * trading_fee
            position = 0

            if trade_return > 0:
                good_trades.append(trade_return)
            else:
                bad_trades.append(trade_return)
            total_trades += 1

            console.print(f"[purple][bold]st[/bold] [white]► trade {trade_return * 100: .2f}%")
            plt.scatter(actual_times[i], actual_prices[i], color="red", s=150, label="Sell Signal")

        if position == 1 and (current_price - entry_price) / entry_price < -0.05:
            exit_price = current_price * (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price
            balance += trade_return * balance
            balance -= balance * trading_fee
            position = 0
            total_trades += 1
            (good_trades if trade_return > 0 else bad_trades).append(trade_return)
            console.print(f"[purple][bold]st[/bold] [white]► trade {trade_return * 100: .2f}%")
            plt.scatter(actual_times[i], actual_prices[i], color="red", s=150, label="Sell Signal")

        equity_curve.append(balance)

    equity_arr = np.array(equity_curve)
    if len(equity_arr) > 0:
        peaks = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - peaks) / peaks
        max_drawdown = drawdowns.min()
    else:
        max_drawdown = 0.0

    best_trade = max(good_trades) if len(good_trades) > 0 else 0.0
    worst_trade = min(bad_trades) if len(bad_trades) > 0 else 0.0

    console.print(
        "[purple][bold]"
        + prompt
        + f"[/bold] [white]► net profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%"
    )
    console.print("[purple][bold]" + prompt + f"[/bold] [white]► total trades: {total_trades}")
    if total_trades > 0:
        console.print(
            "[purple][bold]"
            + prompt
            + f"[/bold] [white]► accuracy: {(len(good_trades) / total_trades) * 100: .2f}%"
        )
    console.print(
        "[purple][bold]" + prompt + f"[/bold] [white]► average good trades: {np.mean(good_trades) * 100: .2f}%"
    )
    console.print(
        "[purple][bold]" + prompt + f"[/bold] [white]► average bad trades: {np.mean(bad_trades) * 100: .2f}%"
    )
    console.print("[purple][bold]" + prompt + f"[/bold] [white]► best trade: {best_trade * 100: .2f}%")
    console.print("[purple][bold]" + prompt + f"[/bold] [white]► worst trade: {worst_trade * 100: .2f}%")
    console.print("[purple][bold]" + prompt + f"[/bold] [white]► max drawdown: {max_drawdown * 100: .2f}%")
    console.print(
        "[purple][bold]" + prompt + f"[/bold] [white]► timeframe: {(len(predictions) * 5) / 60 / 24:.2f} days"
    )

    plt.plot(df.iloc[start:stop]["0"], df.iloc[start:stop]["4"], color="black", label="Data Points")
    plt.plot(df.iloc[start:stop]["0"], df.iloc[start:stop]["EMA"], color="orange", label="Data Points")
    plt.scatter(df.iloc[start:stop]["0"], df.iloc[start:stop]["max"], color="orange", marker="^", label="Sell Signal")
    plt.scatter(df.iloc[start:stop]["0"], df.iloc[start:stop]["min"], color="purple", marker="^", label="Sell Signal")
    plt.title(prompt)
    plt.tight_layout()
    plt.show(block=block)


def back_test_ai(df, predictions, prompt, start, stop, block=True):
    initial_balance = 100
    balance = initial_balance
    position = 0
    entry_price = 0
    trading_fee = 0.001
    slippage = 0.0005
    good_trades = []
    bad_trades = []
    total_trades = 0

    equity_curve = [balance]

    plt.figure(figsize=(14, 7))
    actual_times = df["0"].iloc[start:stop].values
    actual_prices = df["4"].iloc[start:stop].values

    for i in range(len(predictions)):
        current_price = df["4"].iloc[start + i]
        signal = predictions[i]

        if signal == 1 and position == 0:
            position = 1
            entry_price = current_price * (1 + slippage)
            balance -= balance * trading_fee
            plt.scatter(actual_times[i], actual_prices[i], color="green", s=150, label="Buy Signal")

        elif signal == 0 and position == 1:
            exit_price = current_price * (1 - slippage)
            pnl = (exit_price - entry_price) / entry_price
            balance += pnl * balance
            balance -= balance * trading_fee
            position = 0
            (good_trades if pnl > 0 else bad_trades).append(pnl)
            total_trades += 1

            console.print(f"[purple][bold]st[/bold] [white]► trade {pnl * 100: .2f}%")
            plt.scatter(actual_times[i], actual_prices[i], color="red", s=150, label="Sell Signal")

        if position == 1 and (current_price - entry_price) / entry_price < -0.05:
            exit_price = current_price * (1 - slippage)
            pnl = (exit_price - entry_price) / entry_price
            balance += pnl * balance
            balance -= balance * trading_fee
            position = 0
            (good_trades if pnl > 0 else bad_trades).append(pnl)
            total_trades += 1

            console.print(f"[purple][bold]st[/bold] [white]► trade {pnl * 100: .2f}%")
            plt.scatter(actual_times[i], actual_prices[i], color="red", s=150, label="Sell Signal")

        equity_curve.append(balance)

    equity_arr = np.array(equity_curve)
    if len(equity_arr) > 0:
        peaks = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - peaks) / peaks
        max_drawdown = drawdowns.min()
    else:
        max_drawdown = 0.0

    best_trade = max(good_trades) if len(good_trades) > 0 else 0.0
    worst_trade = min(bad_trades) if len(bad_trades) > 0 else 0.0

    console.print(
        "[purple][bold]"
        + prompt
        + f"[/bold] [white]► net profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%"
    )
    console.print("[purple][bold]" + prompt + f"[/bold] [white]► total trades: {total_trades}")
    if total_trades > 0:
        console.print(
            "[purple][bold]" + prompt + f"[/bold] [white]► average good trades: {np.mean(good_trades) * 100: .2f}%"
        )
        console.print(
            "[purple][bold]" + prompt + f"[/bold] [white]► average bad trades: {np.mean(bad_trades) * 100: .2f}%"
        )
        console.print("[purple][bold]" + prompt + f"[/bold] [white]► best trade: {best_trade * 100: .2f}%")
        console.print("[purple][bold]" + prompt + f"[/bold] [white]► worst trade: {worst_trade * 100: .2f}%")
        console.print("[purple][bold]" + prompt + f"[/bold] [white]► max drawdown: {max_drawdown * 100: .2f}%")
    console.print(
        "[purple][bold]" + prompt + f"[/bold] [white]► timeframe: {(len(predictions) * 5) / 60 / 24:.2f} days"
    )

    plt.plot(df.iloc[start:stop]["0"], df.iloc[start:stop]["4"], color="black", label="Data Points")
    plt.plot(df.iloc[start:stop]["0"], df.iloc[start:stop]["EMA"], color="orange", label="EMA")
    plt.tight_layout()
    plt.show(block=block)

    return (balance - initial_balance) / initial_balance


def back_test_return_only(df, predictions, start, stop):
    initial_balance = 100.0
    balance = initial_balance
    position = 0
    entry_price = 0.0
    trading_fee = 0.001
    slippage = 0.0005

    rsi_indicator = df["RSI"].iloc[start:stop].values
    st_indicator = df["SuperTrend_Direction"].iloc[start:stop].values

    rsi_up = False
    rsi_down = False

    for i in range(len(predictions)):
        current_price = df["4"].iloc[start + i]
        signal = predictions[i]

        if rsi_indicator[i] > 60.0:
            rsi_up = True
            rsi_down = False
        elif rsi_indicator[i] < 40.0:
            rsi_up = False
            rsi_down = True

        if signal == 1 and position == 0 and rsi_down and st_indicator[i] == 1:
            position = 1
            entry_price = current_price * (1 + slippage)
            balance -= balance * trading_fee

        elif signal == 0 and position == 1:
            exit_price = current_price * (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price
            balance += trade_return * balance
            balance -= balance * trading_fee
            position = 0

        if position == 1 and (current_price - entry_price) / entry_price < -0.05:
            exit_price = current_price * (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price
            balance += trade_return * balance
            balance -= balance * trading_fee
            position = 0

    return (balance - initial_balance) / initial_balance


def run_simple_portfolio():
    assets = {
        "ETH": "data/1-year-5-min.csv",
        "BTC": "data/BTC_1h.csv",
        "DOGE": "data/DOGE_1h.csv",
        "SOL": "data/SOL_5m_1y.csv",
        "LINK": "data/LINK_5m_1y.csv",
        "ADA": "data/ADA_5m_1y.csv",
    }

    returns = {}

    for symbol, path in assets.items():
        if not os.path.exists(path):
            console.print(f"[purple][bold]st[/bold] [white]► no data for {symbol}, fetching via ccxt")
            exchange = connect_to_exchange()

            timeframe = "1h" if symbol == "BTC" else "5m"
            ohlcv = fetch_data(f"{symbol}/USDT", timeframe, exchange, 108)
            df_tmp = pd.DataFrame(ohlcv)
            df_tmp.to_csv(path, index=False)
            console.print(f"[purple][bold]st[/bold] [white]► saved {symbol} data to {path}")

        console.print(f"[purple][bold]st[/bold] [white]► running strategy on {symbol} ({path})")
        df = pd.read_csv(path)

        df = attach_real_news_sentiment(df, symbol)
        df, X, y = process_data(df)

        proba, idx = train_ai_filter(df)
        proba_map = None
        if proba is not None:
            proba_map = {int(idx[i]): float(proba[i]) for i in range(len(idx))}

        start = 100

        if symbol == "ETH":
            base_preds = generate_volume_news_signals(df, start_index=start)
            if len(base_preds) == 0:
                console.print(f"[purple][bold]st[/bold] [white]► no predictions for {symbol}, skipping")
                continue

            preds = base_preds
            stop = start + len(preds)
            ret = back_test_return_only(df, preds, start, stop)
            returns[symbol] = ret
            console.print(f"[purple][bold]st[/bold] [white]► {symbol} return: {ret * 100: .2f}%")
            back_test(df, preds, symbol, start, stop, block=False)

        elif symbol == "BTC":
            base_preds = generate_volume_news_signals(df, start_index=start)
            if len(base_preds) == 0:
                console.print(f"[purple][bold]st[/bold] [white]► no predictions for {symbol}, skipping")
                continue

            preds = apply_ai_filter(base_preds, start, proba_map, buy_thresh=0.55, sell_thresh=0.45)
            stop = start + len(preds)
            ret = back_test_return_only(df, preds, start, stop)
            returns[symbol] = ret
            console.print(f"[purple][bold]st[/bold] [white]► {symbol} return: {ret * 100: .2f}%")
            back_test(df, preds, symbol, start, stop, block=False)

        else:
            base_preds = generate_altcoin_signals(df, start_index=start)
            if len(base_preds) == 0:
                console.print(f"[purple][bold]st[/bold] [white]► no altcoin signals for {symbol}, skipping")
                continue

            preds = apply_ai_filter(base_preds, start, proba_map, buy_thresh=0.6, sell_thresh=0.4)
            stop = start + len(preds)
            ret = back_test_ai(df, preds, symbol, start, stop, block=False)
            returns[symbol] = ret
            console.print(f"[purple][bold]st[/bold] [white]► {symbol} return (alt + AI): {ret * 100: .2f}%")

    if not returns:
        console.print("[purple][bold]st[/bold] [white]► no assets could be evaluated")
        return

    portfolio_return = np.mean(list(returns.values()))
    console.print(
        "[purple][bold]st[/bold] [white]► equal-weight portfolio return: "
        f"{portfolio_return * 100: .2f}%"
    )


