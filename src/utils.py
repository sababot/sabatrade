from rich.progress import Progress
from rich.console import Console

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import time
import datetime
import logging

import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

console = Console()

# Simple file logger for live trading so we can inspect testnet behaviour later
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("live_trading")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "live_trades.log"))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
    console.print("[purple][bold]st[/bold] [white]► connecting to Coinbase Advanced")
    exchange = ccxt.coinbaseadvanced({
        "apiKey": os.environ.get("COINBASE_API_KEY"),
        "secret": os.environ.get("COINBASE_API_SECRET"),
        "enableRateLimit": True,
    })

    # Coinbase keys are often just 'apiKey' and 'secret', but sometimes require 'password' (passphrase)
    # for Coinbase Advanced Trade / Pro depending on migration status.
    # Assuming standard API keys for now.
    
    if not exchange.apiKey or not exchange.secret:
         console.print("[red][bold]st[/bold] [white]► Error: COINBASE_API_KEY and COINBASE_API_SECRET must be set.")

    return exchange


def place_order(exchange, symbol, side, amount, price=None):
    """
    Minimal spot order wrapper for live trading.

    Parameters
    ----------
    exchange : ccxt.binance instance (already authenticated)
    symbol   : str, e.g. "ETH/USDT"
    side     : "buy" or "sell"
    amount   : float, base-asset amount (e.g. 0.05 for 0.05 ETH)
    price    : float or None
        If None, sends a market order.
        If provided, sends a limit order at this price.
    """
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"Invalid side '{side}', expected 'buy' or 'sell'")

    try:
        # Coinbase Advanced requires a price for market buys to calculate cost (amount * price)
        if price is None and side == 'buy':
            price = exchange.fetch_ticker(symbol)['last']

        # Force sell max available if side is sell, to avoid insufficient funds due to fees
        if side == 'sell':
            try:
                base_currency = symbol.split('/')[0]
                bal = exchange.fetch_balance()
                available = bal.get(base_currency, {}).get('free', 0.0)
                if available > 0:
                    # Ensure precision is correct
                    amount = float(exchange.amount_to_precision(symbol, available))
                    console.print(f"[purple][bold]st[/bold] [white]► overriding sell amount to max available: {amount}")
            except Exception as e:
                console.print(f"[red]Error fetching balance for sell: {e}[/red]")

        # Try to respect Binance min notional / min amount filters where possible
        markets = getattr(exchange, "markets", None) or exchange.load_markets()
        market = markets.get(symbol) if isinstance(markets, dict) else None

        if price is not None and market is not None:
            limits = market.get("limits", {})
            cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}
            min_cost = cost_limits.get("min")

            if min_cost is not None:
                est_cost = float(amount) * float(price)
                if est_cost < float(min_cost):
                    # Scale amount up to satisfy min notional; this avoids Filter failure: NOTIONAL
                    min_amount = float(min_cost) / float(price)
                    amount = min_amount
                    console.print(
                        f"[purple][bold]st[/bold] [white]► adjusting amount for {symbol} to meet min notional: "
                        f"requested notional {est_cost:.4f} < min {min_cost}, using amount={amount:.8f}"
                    )
                    logger.info(
                        f"{symbol} adjust amount for min notional: old_cost={est_cost:.4f}, "
                        f"min_cost={min_cost}, new_amount={amount:.8f}"
                    )

        if price is None:
            order = exchange.create_order(symbol, "market", side, amount)
        else:
            # Pass price to create_order so ccxt/coinbase can calculate 'cost' for market buys
            # or place a limit order if 'limit' was intended (but logic here uses 'market' type)
            order = exchange.create_order(symbol, "market", side, amount, price)

        msg = f"order placed: {symbol} {side} {amount} @ {price or 'market'}"
        console.print(f"[purple][bold]st[/bold] [white]► {msg}")
        logger.info(msg)
        return order
    except Exception as e:
        msg = f"order error for {symbol} {side} {amount} @ {price or 'market'}: {e}"
        console.print(f"[purple][bold]st[/bold] [white]► {msg}")
        logger.error(msg)
        return None


def live_trade_loop(symbol, timeframe, base_amount, window=500, poll_seconds=60):
    """
    Very simple live trading loop using the existing strategy.

    - Fetches recent OHLCV data for `symbol` and `timeframe`.
    - Recomputes indicators via process_data.
    - Generates volume+news signals for the whole window and uses ONLY the last one.
    - Tracks a single position (0 = flat, 1 = long).
    - Places market orders with place_order when signals change.

    Parameters
    ----------
    symbol       : str, e.g. "ETH/USDT"
    timeframe    : str, e.g. "5m", "1h"
    base_amount  : float, size in base asset (e.g. 0.05 for 0.05 ETH)
    window       : int, number of candles to keep in the rolling dataframe
    poll_seconds : int, how often to refresh (in seconds)
    """
    exchange = connect_to_exchange()
    position = 0  # 0 = flat, 1 = long

    # Decide which rule set to use, based on the base asset
    base_symbol = symbol.split("/")[0]
    use_alt_strategy = base_symbol in ("DOGE", "ADA", "SOL", "LINK")

    console.print(f"[purple][bold]st[/bold] [white]► starting live loop for {symbol} ({timeframe}), amount {base_amount}")
    logger.info(f"starting live loop for {symbol} {timeframe}, amount={base_amount}, window={window}, poll={poll_seconds}")

    # Startup Check: Buy then Sell - COMMENTED OUT TO SAVE FEES
    # console.print("[purple][bold]st[/bold] [white]► performing startup check (buy + sell)")
    # try:
    #     # Buy
    #     console.print(f"[purple][bold]st[/bold] [white]► startup: buying {base_amount} {symbol}")
    #     buy_order = place_order(exchange, symbol, "buy", base_amount)
    #     if buy_order:
    #         console.print("[purple][bold]st[/bold] [white]► startup buy successful, waiting 5s before selling...")
    #         time.sleep(5)
    #         # Sell
    #         console.print(f"[purple][bold]st[/bold] [white]► startup: selling {base_amount} {symbol}")
    #         sell_order = place_order(exchange, symbol, "sell", base_amount)
    #         if sell_order:
    #             console.print("[purple][bold]st[/bold] [white]► startup check complete: buy and sell successful")
    #         else:
    #             console.print("[red][bold]st[/bold] [white]► startup sell failed!")
    #     else:
    #         console.print("[red][bold]st[/bold] [white]► startup buy failed!")
    # except Exception as e:
    #     console.print(f"[red][bold]st[/bold] [white]► startup check error: {e}")


    while True:
        try:
            # Fetch latest window of candles
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=window)
            df = pd.DataFrame(ohlcv)

            # Ensure column names match the rest of the pipeline ('0'..'5')
            df.columns = [str(i) for i in range(df.shape[1])]

            # Build indicators and signals
            df, _, _ = process_data(df)
            start = 100

            # Use altcoin strategy for alts, volume+news for majors
            if use_alt_strategy:
                preds = generate_altcoin_signals(df, start_index=start)
            else:
                preds = generate_volume_news_signals(df, start_index=start)

            if len(preds) == 0:
                logger.info(f"{symbol} {timeframe} no signals (len(preds)==0), position={position}")
                time.sleep(poll_seconds)
                continue

            last_signal = preds[-1]
            last_price = df['4'].iloc[-1]
            logger.info(f"{symbol} {timeframe} tick price={last_price} signal={last_signal} position={position}")

            entry_price = 0.0 if 'entry_price' not in locals() else entry_price

            # Decide and place orders
            
            # --- Safety & Fee Checks ---
            
            # 1. STOP LOSS (Always active if in position)
            if position == 1 and entry_price > 0:
                pnl_pct = (last_price - entry_price) / entry_price
                if pnl_pct < -0.05: # 5% Stop Loss
                    console.print(f"[red][bold]STOP LOSS[/bold] [white]► PnL: {pnl_pct*100:.2f}% triggered.")
                    logger.info(f"{symbol} {timeframe} action=STOP_LOSS price={last_price} pnl={pnl_pct:.4f}")
                    order = place_order(exchange, symbol, "sell", base_amount, price=last_price)
                    if order is not None:
                        position = 0
                        entry_price = 0.0
                    # Skip normal signal processing if SL hit
                    continue

            # 2. BUY Logic
            if position == 0 and last_signal == 1:
                # Filter: Check spread for Altcoin strategy to ensure room for profit > fees
                can_buy = True
                if use_alt_strategy:
                    bb_upper = df['BB_Upper'].iloc[-1]
                    bb_lower = df['BB_Lower'].iloc[-1]
                    if pd.notna(bb_upper) and pd.notna(bb_lower) and last_price > 0:
                        spread = (bb_upper - bb_lower) / last_price
                        # We need > 3% to cover ~2.4% fees + slip + profit
                        if spread < 0.03:
                            console.print(f"[yellow][bold]st[/bold] [white]► Skipping Buy: BB Spread {spread*100:.2f}% < 3% threshold")
                            logger.info(f"{symbol} SKIP BUY: Spread too small ({spread:.4f})")
                            can_buy = False

                if can_buy:
                    console.print(f"[purple][bold]st[/bold] [white]► live signal BUY at {last_price}")
                    logger.info(f"{symbol} {timeframe} action=BUY price={last_price} amount={base_amount}")
                    # Pass last_price so place_order can enforce min notional
                    order = place_order(exchange, symbol, "buy", base_amount, price=last_price)
                    if order is not None:
                        position = 1
                        entry_price = last_price

            # 3. SELL Logic
            elif position == 1 and last_signal == 0:
                # Filter: Minimum Profit to cover fees
                pnl_pct = (last_price - entry_price) / entry_price
                min_profit = 0.025 # 2.5% to cover 2.4% fees approx
                
                if pnl_pct > min_profit:
                    console.print(f"[purple][bold]st[/bold] [white]► live signal SELL at {last_price} (PnL: {pnl_pct*100:.2f}%)")
                    logger.info(f"{symbol} {timeframe} action=SELL price={last_price} amount={base_amount}")
                    # Pass last_price so place_order can enforce min notional
                    order = place_order(exchange, symbol, "sell", base_amount, price=last_price)
                    if order is not None:
                        position = 0
                        entry_price = 0.0
                else:
                    console.print(f"[yellow][bold]st[/bold] [white]► Holding: Signal Sell but PnL {pnl_pct*100:.2f}% < {min_profit*100}%")
                    logger.info(f"{symbol} IGNORE SELL: PnL {pnl_pct:.4f} too low")

        except Exception as e:
            console.print(f"[purple][bold]st[/bold] [white]► live loop error: {e}")
            logger.error(f"{symbol} {timeframe} live loop error: {e}")

        time.sleep(poll_seconds)

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
    df['SuperTrend'] = supertrend[f'SUPERT_10_3']  # SuperTrend column name format is SUPERT_length_multiplier
    df['SuperTrend_Direction'] = supertrend[f'SUPERTd_10_3']  # Trend direction: 1 (bullish), -1 (bearish)

    bollinger = ta.bbands(df['4'], length=20, std=2)  # Default length is 20, std is 2
    df['BB_Middle'] = bollinger['BBM_20_2.0_2.0']  # Bollinger Middle Band
    df['BB_Upper'] = bollinger['BBU_20_2.0_2.0']   # Bollinger Upper Band
    df['BB_Lower'] = bollinger['BBL_20_2.0_2.0']   # Bollinger Lower Band

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
        ohlcv = fetch_data('BTC/USDT', '1h', exchange, n)
        df = pd.DataFrame(ohlcv)
        df.to_csv(f'data/BTC_1h.csv', index=False)
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


def generate_volume_news_signals(df, start_index=100, lookback=60):
    """
    Simple volume + news-proxy strategy.

    - Volume signal: current volume > 1.5 * rolling mean volume.
    - News signal (proxy): use recent return as a proxy for news impact:
        * sentiment =  1 if return over the last 12 bars > +1%
        * sentiment = -1 if return over the last 12 bars < -1%
        * sentiment =  0 otherwise
    - Trade logic (aligned with back_test expectations):
        * prediction = 1 → candidate buy (back_test will still gate with RSI / SuperTrend)
        * prediction = 0 → candidate exit
        * prediction = 2 → no action

    Returns a list of integer predictions for indices [start_index, len(df)).
    """
    # Defensive: ensure we have enough data
    if len(df) < start_index + lookback + 12:
        return []

    # Rolling volume mean
    df['VOL_MA'] = df['5'].rolling(window=lookback).mean()

    # Return over last 12 bars (~1h for 5m data)
    df['RET_12'] = df['4'].pct_change(periods=12)

    # Simple news sentiment proxy (price-based) as a fallback
    df['News_Sentiment'] = 0
    df.loc[df['RET_12'] > 0.01, 'News_Sentiment'] = 1
    df.loc[df['RET_12'] < -0.01, 'News_Sentiment'] = -1

    predictions = []

    for i in range(start_index, len(df)):
        vol = df['5'].iloc[i]
        vol_ma = df['VOL_MA'].iloc[i]
        # Prefer real news sentiment if available; otherwise use price-based proxy
        if 'News_Sentiment_Real' in df.columns:
            sentiment = df['News_Sentiment_Real'].iloc[i]
        else:
            sentiment = df['News_Sentiment'].iloc[i]
        
        # Use SMA_20 for faster trend reaction (was EMA/SMA_40)
        trend = df['SMA_20'].iloc[i] if 'SMA_20' in df.columns else (df['EMA'].iloc[i] if 'EMA' in df.columns else df['4'].iloc[i])
        rsi = df['RSI'].iloc[i] if 'RSI' in df.columns else 50

        # Skip until volume MA is defined
        if pd.isna(vol_ma):
            predictions.append(2)
            continue

        # Bullish condition: volume spike + positive sentiment + price above trend
        bullish = (vol > 1.5 * vol_ma) and (sentiment == 1) and (df['4'].iloc[i] > trend)

        # Bearish/exit condition: negative sentiment OR price drops below trend OR RSI overbought
        bearish = (sentiment == -1) or (df['4'].iloc[i] < trend) or (rsi > 75)

        if bullish:
            predictions.append(1)
        elif bearish:
            predictions.append(0)
        else:
            predictions.append(2)

    return predictions


def generate_altcoin_signals(df, start_index=100, lookback=60):
    """
    Mean Reversion strategy for Altcoins (Buying the dip).
    Replaces the previous momentum/breakout logic which was buying tops.

    - Buy: RSI < 30 (Oversold) OR Price < Lower Bollinger Band (Dip).
    - Sell: RSI > 70 (Overbought) OR Price > Upper Bollinger Band (Pump).

    Returns predictions for indices [start_index, len(df)):
        1 → enter/hold long bias
        0 → exit / flat
        2 → no action
    """
    if len(df) < start_index + lookback + 12:
        return []

    predictions = []

    for i in range(start_index, len(df)):
        rsi = df['RSI'].iloc[i] if 'RSI' in df.columns else 50
        price = df['4'].iloc[i]
        bb_lower = df['BB_Lower'].iloc[i] if 'BB_Lower' in df.columns else 0
        bb_upper = df['BB_Upper'].iloc[i] if 'BB_Upper' in df.columns else float('inf')

        # Mean Reversion Entry: Buy when oversold or below lower band
        bullish = (rsi < 30) or (price < bb_lower)

        # Mean Reversion Exit: Sell when overbought or above upper band
        bearish = (rsi > 70) or (price > bb_upper)

        if bullish:
            predictions.append(1)
        elif bearish:
            predictions.append(0)
        else:
            predictions.append(2)

    return predictions


def build_volume_news_ai_features(df, lookback=60, horizon=12, future_threshold=0.0):
    """
    Build a supervised learning dataset focused on volume + news-proxy features.

    - Volume: use volume / rolling mean volume as a feature.
    - News proxy: 12-bar return and its discretised sentiment.
    - Target: 1 if future return over 'horizon' bars > future_threshold (default 0.0), else 0.

    Returns:
        X: feature matrix (np.ndarray)
        y: labels (np.ndarray)
        idx: index (pd.Index) mapping rows back to the original df
    """
    df_feat = df.copy()

    # Rolling volume mean and ratio
    df_feat['VOL_MA'] = df_feat['5'].rolling(window=lookback).mean()
    df_feat['VOL_RATIO'] = df_feat['5'] / df_feat['VOL_MA']

    # News sentiment feature: prefer real news if available, otherwise price-based proxy
    df_feat['RET_12'] = df_feat['4'].pct_change(periods=horizon)
    df_feat['News_Sentiment_Proxy'] = 0
    df_feat.loc[df_feat['RET_12'] > 0.01, 'News_Sentiment_Proxy'] = 1
    df_feat.loc[df_feat['RET_12'] < -0.01, 'News_Sentiment_Proxy'] = -1

    if 'News_Sentiment_Real' in df_feat.columns:
        df_feat['News_Sentiment_Feature'] = df_feat['News_Sentiment_Real']
    else:
        df_feat['News_Sentiment_Feature'] = df_feat['News_Sentiment_Proxy']

    # Future return over the same horizon as the news window
    df_feat['FUT_RET'] = df_feat['4'].pct_change(periods=horizon).shift(-horizon)
    df_feat['label'] = (df_feat['FUT_RET'] > future_threshold).astype(int)

    feature_cols = [
        'VOL_RATIO',
        'News_Sentiment_Feature',
        'RSI',
        'SuperTrend_Direction',
        'ATR',
        'ROC',
        'CCI',
        'BB_Upper',
        'BB_Lower',
    ]

    # Drop rows where any feature or label is NaN
    data = df_feat[feature_cols + ['label']].dropna()

    X = data[feature_cols].values
    y = data['label'].values
    idx = data.index

    return X, y, idx


def attach_real_news_sentiment(df, symbol, tolerance_ms=60 * 60 * 1000):
    """
    Attach real news sentiment to a price dataframe if a news CSV is available.

    Expected file per symbol:
        data/news_{symbol}.csv

    CSV columns:
        - timestamp: Unix ms (same scale as df['0'])
        - sentiment: numeric in [-1, 1] (e.g. -1 bearish, 0 neutral, 1 bullish)

    For each bar, uses the most recent news item within `tolerance_ms` and
    writes it to df['News_Sentiment_Real']. Bars with no news in the window
    get 0. If no file is found or format is wrong, df is returned unchanged.
    """
    path = f"data/news_{symbol}.csv"
    if not os.path.exists(path):
        return df

    try:
        news = pd.read_csv(path)
    except Exception:
        return df

    if 'timestamp' not in news.columns or 'sentiment' not in news.columns:
        return df

    # Ensure sorted by time
    news = news.sort_values('timestamp')

    # Work on a copy to avoid side effects
    df_local = df.copy()
    df_local = df_local.sort_values('0')

    try:
        merged = pd.merge_asof(
            df_local,
            news[['timestamp', 'sentiment']].sort_values('timestamp'),
            left_on='0',
            right_on='timestamp',
            direction='backward',
            tolerance=tolerance_ms,
        )
    except Exception:
        return df

    df_local['News_Sentiment_Real'] = merged['sentiment'].fillna(0)

    # Restore original index order
    df_local = df_local.sort_index()
    return df_local


def train_ai_filter(df, lookback=60, horizon=12, future_threshold=0.0):
    """
    Train a simple XGBoost classifier to act as a filter on top of the
    existing volume+news / altcoin rule-based strategies.

    - Uses build_volume_news_ai_features for inputs and labels.
    - Returns:
        proba: np.ndarray of P(y=1 | features) for each index in idx
        idx:   pd.Index of df rows corresponding to proba
      or (None, None) if there is not enough data.
    """
    X, y, idx = build_volume_news_ai_features(df, lookback=lookback, horizon=horizon, future_threshold=future_threshold)
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

    # Probabilities on full feature set (for alignment with df indices)
    proba = model.predict_proba(scaler.transform(X))[:, 1]
    return proba, idx


def apply_ai_filter(base_signals, start_index, proba_map, buy_thresh=0.55, sell_thresh=0.45):
    """
    Combine rule-based signals with AI probabilities:

    - If no proba_map is provided, returns base_signals unchanged.
    - Otherwise:
        * base == 1 and p > buy_thresh  -> 1
        * base == 0 and p < sell_thresh -> 0
        * else                          -> 2 (no action)
    """
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
    initial_balance = 100  # Starting capital in USD
    balance = initial_balance
    position = 0  # 1 = long, -1 = short, 0 = no position
    entry_price = 0 # Price at which the position is entered
    trading_fee = 0.001  # 0.1% trading fee per transaction
    slippage = 0.0005    # 0.05% slippage per trade side
    good_trades = []
    bad_trades = []
    total_trades = 0

    # Equity curve for risk metrics
    equity_curve = [balance]

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
            # Assume we pay slightly worse than mid due to slippage
            entry_price = current_price * (1 + slippage)
            balance -= balance * trading_fee  # Deduct trading fee for entering the position
            plt.scatter(actual_times[i], actual_prices[i], color='green', s=150, label='Buy Signal')

        elif signal == 0 and position == 1:  # Sell signal
            # Effective exit price with slippage against us
            exit_price = current_price * (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price
            balance += trade_return * balance  # Calculate profit/loss
            balance -= balance * trading_fee  # Deduct trading fee for exiting the position
            position = 0

            if trade_return > 0:
                good_trades.append(trade_return)
                console.print(f"[purple][bold]st[/bold] [white]► trade {trade_return * 100: .2f}%")
                total_trades += 1
            elif trade_return <= 0:
                bad_trades.append(trade_return)
                console.print(f"[purple][bold]st[/bold] [white]► trade {trade_return * 100: .2f}%")
                total_trades += 1

            plt.scatter(actual_times[i], actual_prices[i], color='red', s=150, label='Sell Signal')

        if position == 1 and (current_price - entry_price) / entry_price < -0.05:
            # Stop-loss exit with slippage
            exit_price = current_price * (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price
            balance += trade_return * balance  # Calculate profit/loss
            balance -= balance * trading_fee  # Deduct trading fee for exiting the position
            position = 0

            if trade_return > 0:
                good_trades.append(trade_return)
                console.print(f"[purple][bold]st[/bold] [white]► trade {trade_return * 100: .2f}%")
                total_trades += 1
            elif trade_return <= 0:
                bad_trades.append(trade_return)
                console.print(f"[purple][bold]st[/bold] [white]► trade {trade_return * 100: .2f}%")
                total_trades += 1

            plt.scatter(actual_times[i], actual_prices[i], color='red', s=150, label='Sell Signal')

        # Record balance after this bar for drawdown calculation
        equity_curve.append(balance)

    #if position == 1:
        #balance += (current_price - entry_price) / entry_price * balance
        #balance -= current_price * trading_fee  # Deduct trading fee for exiting the position
        #print(f"Final Sell: Closing position at {current_price:.2f}, Balance: {balance:.2f}")

    # Risk metrics
    equity_arr = np.array(equity_curve)
    if len(equity_arr) > 0:
        peaks = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - peaks) / peaks
        max_drawdown = drawdowns.min()
    else:
        max_drawdown = 0.0

    best_trade = max(good_trades) if len(good_trades) > 0 else 0.0
    worst_trade = min(bad_trades) if len(bad_trades) > 0 else 0.0

    # Backtest summary
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► net profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► total trades: {total_trades}")
    if total_trades > 0:
        console.print("[purple][bold]"+ prompt +f"[/bold] [white]► accuracy: {(len(good_trades) / total_trades) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► average good trades: {np.mean(good_trades) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► average bad trades: {np.mean(bad_trades) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► best trade: {best_trade * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► worst trade: {worst_trade * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► max drawdown: {max_drawdown * 100: .2f}%")
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
    plt.title(prompt)
    plt.tight_layout()
    plt.show(block=block)


def back_test_ai(df, predictions, prompt, start, stop, block=True):
    """
    Simpler backtest for AI strategies:

    - signal == 1 → enter long if flat
    - signal == 0 → exit long if in position
    - signal == 2 → hold

    Keeps the same fee and stop-loss logic as back_test, but does not gate on RSI/SuperTrend.
    """
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
    actual_times = df['0'].iloc[start:stop].values
    actual_prices = df['4'].iloc[start:stop].values

    for i in range(len(predictions)):
        current_price = df['4'].iloc[start + i]
        signal = predictions[i]

        # Enter long
        if signal == 1 and position == 0:
            position = 1
            entry_price = current_price * (1 + slippage)
            balance -= balance * trading_fee
            plt.scatter(actual_times[i], actual_prices[i], color='green', s=150, label='Buy Signal')

        # Exit long on model signal
        elif signal == 0 and position == 1:
            exit_price = current_price * (1 - slippage)
            pnl = (exit_price - entry_price) / entry_price
            balance += pnl * balance
            balance -= balance * trading_fee
            position = 0
            if pnl > 0:
                good_trades.append(pnl)
            else:
                bad_trades.append(pnl)
            total_trades += 1

            console.print(f"[purple][bold]st[/bold] [white]► trade {pnl * 100: .2f}%")
            plt.scatter(actual_times[i], actual_prices[i], color='red', s=150, label='Sell Signal')

        # Stop-loss at -5%
        if position == 1 and (current_price - entry_price) / entry_price < -0.05:
            exit_price = current_price * (1 - slippage)
            pnl = (exit_price - entry_price) / entry_price
            balance += pnl * balance
            balance -= balance * trading_fee
            position = 0
            if pnl > 0:
                good_trades.append(pnl)
            else:
                bad_trades.append(pnl)
            total_trades += 1

            console.print(f"[purple][bold]st[/bold] [white]► trade {pnl * 100: .2f}%")
            plt.scatter(actual_times[i], actual_prices[i], color='red', s=150, label='Sell Signal')

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

    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► net profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► total trades: {total_trades}")
    if total_trades > 0:
        console.print("[purple][bold]"+ prompt +f"[/bold] [white]► average good trades: {np.mean(good_trades) * 100: .2f}%")
        console.print("[purple][bold]"+ prompt +f"[/bold] [white]► average bad trades: {np.mean(bad_trades) * 100: .2f}%")
        console.print("[purple][bold]"+ prompt +f"[/bold] [white]► best trade: {best_trade * 100: .2f}%")
        console.print("[purple][bold]"+ prompt +f"[/bold] [white]► worst trade: {worst_trade * 100: .2f}%")
        console.print("[purple][bold]"+ prompt +f"[/bold] [white]► max drawdown: {max_drawdown * 100: .2f}%")
    console.print("[purple][bold]"+ prompt +f"[/bold] [white]► timeframe: {(len(predictions) * 5) / 60 / 24:.2f} days")

    plt.plot(df.iloc[start:stop]['0'], df.iloc[start:stop]['4'], color='black', label='Data Points')
    plt.plot(df.iloc[start:stop]['0'], df.iloc[start:stop]['EMA'], color='orange', label='EMA')
    plt.tight_layout()
    plt.show(block=block)

    return (balance - initial_balance) / initial_balance


def back_test_return_only(df, predictions, start, stop):
    """
    Backtest the existing rule-based strategy but return only total return.

    This mirrors the logic in back_test (RSI + SuperTrend gating, fee, stop loss)
    without plotting or console printing, so it can be used inside a portfolio runner.
    """
    initial_balance = 100.0
    balance = initial_balance
    position = 0
    entry_price = 0.0
    trading_fee = 0.001
    slippage = 0.0005

    rsi_indicator = df['RSI'].iloc[start:stop].values
    st_indicator = df['SuperTrend_Direction'].iloc[start:stop].values

    rsi_up = False
    rsi_down = False

    for i in range(len(predictions)):
        current_price = df['4'].iloc[start + i]
        signal = predictions[i]

        if rsi_indicator[i] > 60.0:
            rsi_up = True
            rsi_down = False
        elif rsi_indicator[i] < 40.0:
            rsi_up = False
            rsi_down = True

        # Buy condition (same as back_test)
        if signal == 1 and position == 0 and rsi_down and st_indicator[i] == 1:
            position = 1
            entry_price = current_price * (1 + slippage)
            balance -= balance * trading_fee

        # Sell condition
        elif signal == 0 and position == 1:
            exit_price = current_price * (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price
            balance += trade_return * balance
            balance -= balance * trading_fee
            position = 0

        # Stop-loss at -5%
        if position == 1 and (current_price - entry_price) / entry_price < -0.05:
            exit_price = current_price * (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price
            balance += trade_return * balance
            balance -= balance * trading_fee
            position = 0

    return (balance - initial_balance) / initial_balance


def run_simple_portfolio():
    """
    Run the volume + news rule-based strategy on a small portfolio
    of assets and report per-asset and equal-weight portfolio returns.

    Expected data files (adjust paths as needed):
        ETH : data/1-year-5-min.csv
        BTC : data/BTC_1h.csv
        DOGE: data/DOGE_1h.csv
    """
    assets = {
        "ETH": "data/1-year-5-min.csv",   # existing ETH 5m data
        "BTC": "data/BTC_1h.csv",         # existing BTC 1h data
        "DOGE": "data/DOGE_1h.csv",       # existing DOGE data
        "SOL": "data/SOL_5m_1y.csv",      # new alt: Solana
        "LINK": "data/LINK_5m_1y.csv",    # new alt: Chainlink
        "ADA": "data/ADA_5m_1y.csv",      # new alt: Cardano
    }

    returns = {}

    for symbol, path in assets.items():
        if not os.path.exists(path):
            console.print(f"[purple][bold]st[/bold] [white]► no data for {symbol}, fetching via ccxt")
            exchange = connect_to_exchange()

            # Use 5m data for alts (ETH, SOL, LINK, ADA, DOGE if needed), keep BTC at 1h
            if symbol == "BTC":
                timeframe = "1h"
            else:
                timeframe = "5m"

            # Roughly one year of data: 108 * 1000 candles at 5m, reuse for 1h for simplicity
            ohlcv = fetch_data(f"{symbol}/USDT", timeframe, exchange, 108)
            df_tmp = pd.DataFrame(ohlcv)
            df_tmp.to_csv(path, index=False)
            console.print(f"[purple][bold]st[/bold] [white]► saved {symbol} data to {path}")

        console.print(f"[purple][bold]st[/bold] [white]► running strategy on {symbol} ({path})")
        df = pd.read_csv(path)

        # Attach real news sentiment if available for this symbol
        df = attach_real_news_sentiment(df, symbol)

        df, X, y = process_data(df)

        # Train AI filter once per asset on the processed dataframe
        proba, idx = train_ai_filter(df)
        proba_map = None
        if proba is not None:
            proba_map = {int(idx[i]): float(proba[i]) for i in range(len(idx))}

        start = 100

        # Use more conservative rule-based strategy for BTC, slightly less restrictive for ETH,
        # and higher-risk alt strategy for others, all optionally filtered by the AI model.
        if symbol == "ETH":
            base_preds = generate_volume_news_signals(df, start_index=start)
            if len(base_preds) == 0:
                console.print(f"[purple][bold]st[/bold] [white]► no predictions for {symbol}, skipping")
                continue

            # For ETH, do NOT apply the AI filter to keep the strategy less restrictive.
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
    console.print("[purple][bold]st[/bold] [white]► equal-weight portfolio return: "
                  f"{portfolio_return * 100: .2f}%")
