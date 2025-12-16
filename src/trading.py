import time

import pandas as pd

from common import console, logger
from data import process_data
from exchange import connect_to_exchange
from strategy import generate_altcoin_signals, generate_volume_news_signals


def place_order(exchange, symbol, side, amount, price=None):
    """
    Minimal spot order wrapper for live trading.
    """
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"Invalid side '{side}', expected 'buy' or 'sell'")

    try:
        # Coinbase Advanced requires a price for market buys to calculate cost (amount * price)
        if price is None and side == "buy":
            price = exchange.fetch_ticker(symbol)["last"]

        # Force sell max available if side is sell, to avoid insufficient funds due to fees
        if side == "sell":
            try:
                base_currency = symbol.split("/")[0]
                bal = exchange.fetch_balance()
                available = bal.get(base_currency, {}).get("free", 0.0)
                if available > 0:
                    amount = float(exchange.amount_to_precision(symbol, available))
                    console.print(
                        f"[purple][bold]st[/bold] [white]► overriding sell amount to max available: {amount}"
                    )
            except Exception as e:
                console.print(f"[red]Error fetching balance for sell: {e}[/red]")

        markets = getattr(exchange, "markets", None) or exchange.load_markets()
        market = markets.get(symbol) if isinstance(markets, dict) else None

        if price is not None and market is not None:
            limits = market.get("limits", {})
            cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}
            min_cost = cost_limits.get("min")

            if min_cost is not None:
                est_cost = float(amount) * float(price)
                if est_cost < float(min_cost):
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
    - Generates signals for the whole window and uses ONLY the last one.
    - Tracks a single position (0 = flat, 1 = long).
    - Places market orders with place_order when signals change.
    """
    exchange = connect_to_exchange()
    position = 0

    base_symbol = symbol.split("/")[0]
    use_alt_strategy = base_symbol in ("DOGE", "ADA", "SOL", "LINK")

    console.print(
        f"[purple][bold]st[/bold] [white]► starting live loop for {symbol} ({timeframe}), amount {base_amount}"
    )
    logger.info(
        f"starting live loop for {symbol} {timeframe}, amount={base_amount}, window={window}, poll={poll_seconds}"
    )

    entry_price = 0.0

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=window)
            df = pd.DataFrame(ohlcv)
            df.columns = [str(i) for i in range(df.shape[1])]

            df, _, _ = process_data(df)
            start = 100

            if use_alt_strategy:
                preds = generate_altcoin_signals(df, start_index=start)
            else:
                preds = generate_volume_news_signals(df, start_index=start)

            if len(preds) == 0:
                logger.info(f"{symbol} {timeframe} no signals (len(preds)==0), position={position}")
                time.sleep(poll_seconds)
                continue

            last_signal = preds[-1]
            last_price = df["4"].iloc[-1]
            logger.info(
                f"{symbol} {timeframe} tick price={last_price} signal={last_signal} position={position}"
            )

            # 1) STOP LOSS
            if position == 1 and entry_price > 0:
                pnl_pct = (last_price - entry_price) / entry_price
                if pnl_pct < -0.05:
                    console.print(
                        f"[red][bold]STOP LOSS[/bold] [white]► PnL: {pnl_pct*100:.2f}% triggered."
                    )
                    logger.info(
                        f"{symbol} {timeframe} action=STOP_LOSS price={last_price} pnl={pnl_pct:.4f}"
                    )
                    order = place_order(exchange, symbol, "sell", base_amount, price=last_price)
                    if order is not None:
                        position = 0
                        entry_price = 0.0
                    time.sleep(poll_seconds)
                    continue

            # 2) BUY
            if position == 0 and last_signal == 1:
                can_buy = True
                if use_alt_strategy:
                    bb_upper = df["BB_Upper"].iloc[-1]
                    bb_lower = df["BB_Lower"].iloc[-1]
                    if pd.notna(bb_upper) and pd.notna(bb_lower) and last_price > 0:
                        spread = (bb_upper - bb_lower) / last_price
                        if spread < 0.03:
                            console.print(
                                f"[yellow][bold]st[/bold] [white]► Skipping Buy: BB Spread {spread*100:.2f}% < 3% threshold"
                            )
                            logger.info(f"{symbol} SKIP BUY: Spread too small ({spread:.4f})")
                            can_buy = False

                if can_buy:
                    console.print(f"[purple][bold]st[/bold] [white]► live signal BUY at {last_price}")
                    logger.info(
                        f"{symbol} {timeframe} action=BUY price={last_price} amount={base_amount}"
                    )
                    order = place_order(exchange, symbol, "buy", base_amount, price=last_price)
                    if order is not None:
                        position = 1
                        entry_price = float(last_price)

            # 3) SELL
            elif position == 1 and last_signal == 0:
                pnl_pct = (last_price - entry_price) / entry_price if entry_price else 0.0
                min_profit = 0.025

                if pnl_pct > min_profit:
                    console.print(
                        f"[purple][bold]st[/bold] [white]► live signal SELL at {last_price} (PnL: {pnl_pct*100:.2f}%)"
                    )
                    logger.info(
                        f"{symbol} {timeframe} action=SELL price={last_price} amount={base_amount}"
                    )
                    order = place_order(exchange, symbol, "sell", base_amount, price=last_price)
                    if order is not None:
                        position = 0
                        entry_price = 0.0
                else:
                    console.print(
                        f"[yellow][bold]st[/bold] [white]► Holding: Signal Sell but PnL {pnl_pct*100:.2f}% < {min_profit*100}%"
                    )
                    logger.info(f"{symbol} IGNORE SELL: PnL {pnl_pct:.4f} too low")

        except Exception as e:
            console.print(f"[purple][bold]st[/bold] [white]► live loop error: {e}")
            logger.error(f"{symbol} {timeframe} live loop error: {e}")

        time.sleep(poll_seconds)


