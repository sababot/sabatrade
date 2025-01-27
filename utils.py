from rich.progress import Progress
from rich.console import Console

import ccxt
import numpy as np
import pandas as pd

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
    since = exchange.fetch_ohlcv(symbol, timeframe, limit=1)[0][0] - (60 * 60 * 1000 * limit * (n))

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