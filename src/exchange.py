import os

import ccxt

from common import console


def connect_to_exchange():
    console.print("[purple][bold]st[/bold] [white]► connecting to Coinbase Advanced")
    exchange = ccxt.coinbaseadvanced(
        {
            "apiKey": os.environ.get("COINBASE_API_KEY"),
            "secret": os.environ.get("COINBASE_API_SECRET"),
            "enableRateLimit": True,
        }
    )

    if not exchange.apiKey or not exchange.secret:
        console.print(
            "[red][bold]st[/bold] [white]► Error: COINBASE_API_KEY and COINBASE_API_SECRET must be set."
        )

    return exchange


