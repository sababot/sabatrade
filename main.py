from rich.progress import Progress
from rich.console import Console

console = Console()

console.print("[bold][white]┌──────────────────────────[purple]──────────────────────────┐")
console.print("[bold][white]│[purple]             _           [white]_                 _        [purple]│")
console.print("[bold][white]│[purple]   ___  __ _| |__   __ _[white]| |_ _ __ __ _  __| | ___   [purple]│")
console.print("[bold][white]│[purple]  / __|/ _` | '_ \\ / _` [white]| __| '__/ _` |/ _` |/ _ \\  [purple]│")
console.print("[bold][purple]│[purple]  \\__ \\ (_| | |_) | (_| [white]| |_| | | (_| | (_| |  __/  [white]│")
console.print("[bold][purple]│[purple]  |___/\\__,_|_.__/ \\__,_|[white]\\__|_|  \\__,_|\\__,_|\\___|  [white]│")
console.print("[bold][purple]│                                                    [white]│")
console.print("[bold][purple]└──────────────────────────[white]──────────────────────────┘\n")

console.print("[purple][bold]st[/bold] [white]initializing libraries")

import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ccxt
import time
import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

#print("[st] Fetching Historical Data: Contacting Server")

def connect_to_exchange():
    #console.print("[purple][bold]st[/bold] [white]connecting to server")
    return ccxt.binance()

console.print("[purple][bold]st[/bold] [white]connecting to server")
exchange = connect_to_exchange()

def fetch_data(symbol, timeframe, exchange, n):
    limit = 1000
    since = exchange.fetch_ohlcv(symbol, timeframe, limit=1)[0][0] - (15 * 60 * 1000 * limit * (n + 55))

    ohlcv = []

    console.print("[purple][bold]st[/bold] [white]fetching historical data")
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

# KNN Algorithm using Tensorflow
class KNN(tf.keras.Model):
    def __init__(self, k=9):
        super(KNN, self).__init__()
        self.k = k

    def fit(self, X_train, y_train, X_test, y_test):
        self.X_train = tf.constant(X_train, dtype=tf.double)
        self.y_train = tf.constant(y_train, dtype=tf.int32)
        self.X_test = tf.constant(X_test, dtype=tf.double)
        self.y_test = tf.constant(y_test, dtype=tf.int32)

    def predict(self, show):
        predictions = []
        if show == True:
            with Progress() as progress:
                # Add a progress task
                task = progress.add_task("  progress:", total=len(self.X_test))

                for test_point in self.X_test:
                    # Compute Euclidean distance from the test point to all training points
                    distances = tf.norm(self.X_train - test_point, axis=1)
                    # Find the indices of the k nearest neighbors
                    k_indices = tf.argsort(distances)[:self.k]
                    # Get the labels of the k nearest neighbors
                    k_labels = tf.gather(self.y_train, k_indices)
                    # Predict the majority class (mode)
                    unique_labels, _, counts = tf.unique_with_counts(k_labels)
                    majority_class = unique_labels[tf.argmax(counts)]
                    predictions.append(int(majority_class))
                    
                    #if (len(predictions) % 10 == 0):
                    #    print(f"[INFO] {len(predictions)}/{len(X_test)}")

                    progress.update(task, advance=1)

        elif show == False:
            for test_point in self.X_test:
                # Compute Euclidean distance from the test point to all training points
                distances = tf.norm(self.X_train - test_point, axis=1)
                # Find the indices of the k nearest neighbors
                k_indices = tf.argsort(distances)[:self.k]
                # Get the labels of the k nearest neighbors
                k_labels = tf.gather(self.y_train, k_indices)
                # Predict the majority class (mode)
                unique_labels, _, counts = tf.unique_with_counts(k_labels)
                majority_class = unique_labels[tf.argmax(counts)]
                predictions.append(int(majority_class))
                if (len(predictions) % 10 == 0):
                    print(f"[INFO] {len(predictions)}/{len(self.X_test)}")
        return predictions

    def save_model(self, filepath):
        np.savez(filepath, X_train=self.X_train.numpy(), y_train=self.y_train.numpy())

    def load_model(self, filepath):
        data = np.load(filepath)
        self.X_train = tf.constant(data['X_train'], dtype=tf.double)
        self.y_train = tf.constant(data['y_train'], dtype=tf.int32)

knn = KNN(k=9)
predictions = []
loaded = False

console.print("[purple][bold]st[/bold] [white]select option:\n  1) new\n  2) load\n  3) back-test\n  4) quit")
choice = int(console.input("[purple][bold]st[/bold] [white]: "))
while choice != 4:
    if choice == 1:
        ohlcv = fetch_data('ETH/USDT', '15m', exchange, 10)

        # Load data (use your own dataset or fetch it via an API)
        # Example: Load a CSV file with OHLCV data
        df = pd.DataFrame(ohlcv)

        # Feature engineering: Create additional features (e.g., moving averages, RSI)
        df['SMA_20'] = df[4].rolling(window=20).mean()
        df['SMA_50'] = df[4].rolling(window=50).mean()
        df['RSI'] = 100 - (100 / (1 + df[4].pct_change().rolling(window=14).apply(lambda x: (x[x > 0].sum() / -x[x < 0].sum()) if x[x < 0].sum() != 0 else 0)))

        # Drop NaN values after feature calculation
        df = df.dropna()

        # Define the target variable (e.g., price increase or decrease over the next 5 timesteps)
        future_steps_buy = 8
        future_steps_sell = 3
        cutoff_buy = 0.01
        cutoff_sell = 0.005
        df['next_buy'] = df[4].shift(-future_steps_buy)
        df['next_sell'] = df[4].shift(-future_steps_sell)
        df['target'] = df.apply(
            lambda row: 1 if (row['next_buy'] - row[4]) / row[4] > cutoff_buy else
                       -1 if (row['next_sell'] - row[4]) / row[4] < -cutoff_sell else
                       0,
            axis=1
        )
        #df['target'] = ((df[4].shift(+future_steps) > df[4])).astype(int)

        # Drop rows without target
        df = df.dropna()

        # Select features for the model
        features = [4, 5, 'SMA_20', 'SMA_50', 'RSI']
        X = df[features].values
        y = df['target'].values

        # Normalize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        console.print("[purple][bold]st[/bold] [white]creating model")
        knn.fit(X_train, y_train, X_test, y_test)
        knn.save_model('models/model.npz')

        predictions = knn.predict(True)

        # Evaluate accuracy
        accuracy = np.mean(np.array(predictions) == y_test)
        console.print(f"[purple][bold]st[/bold] [white]model accuracy: {accuracy * 100:.2f}%")

        loaded == True

    elif choice == 2:
        knn.load_model('models/model.npz')
        predictions = knn.predict(True)

        # Evaluate accuracy
        accuracy = np.mean(np.array(predictions) == y_test)
        console.print(f"[purple][bold]st[/bold] [white]model accuracy: {accuracy * 100:.2f}%")

        loaded == True

    elif choice == 3:
        if loaded == True:
            initial_balance = 100  # Starting capital in USD
            balance = initial_balance
            position = 0  # 1 = long, -1 = short, 0 = no position
            entry_price = 0  # Price at which the position is entered
            trading_fee = 0.001  # 0.075% trading fee per transaction

            # Backtest loop
            for i in range(len(predictions)):
                current_price = df.iloc[len(X_train) + i][4]  # Current price of the asset
                signal = predictions[i]  # Predicted signal (1 = Buy, -1 = Sell, 0 = Hold)

                if signal == 1:  # Buy signal
                    if position == 0:  # Open a long position if no position is open
                        pause = future_steps_buy
                        position = 1
                        entry_price = current_price
                        balance -= balance * trading_fee  # Deduct trading fee for entering the position
                        #print(f"[RESULT] Buy: Entering at {entry_price:.2f}, Balance: {balance:.2f}")
                        #print(f"[RESULT] Buy")

                elif signal == -1:  # Sell signal
                    if position == 1:  # Close the long position if it's open
                        balance += ((current_price - entry_price) / entry_price) * balance  # Calculate profit/loss
                        balance -= balance * trading_fee  # Deduct trading fee for exiting the position
                        position = 0
                        #print(f"Sell: Exiting at {current_price:.2f}, Balance: {balance:.2f}")
                        console.print(f"[purple][bold]st[/bold] [white]trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
                        #print(f"[st] Trade {((current_price - entry_price) / entry_price) * 100: .2f}%")

                elif signal == 0:  # Hold signal
                    if (position == 1 and (current_price - entry_price) / entry_price < -0.0025):
                        balance += 0.01 * balance  # Calculate profit/loss
                        balance -= balance * trading_fee  # Deduct trading fee for exiting the position
                        position = 0
                        #print(f"Sell: Exiting at {current_price:.2f}, Balance: {balance:.2f}, -0.01%")
                        #print(f"[RESULT] Sell -1%")
                        console.print(f"[purple][bold]st[/bold] [white]liquidation -0.25%")
                        #print(f"[ST] Liquidation -1.00%")
                    continue  # Do nothing and move to the next step

            #if position == 1:
                #balance += (current_price - entry_price) / entry_price * balance
                #balance -= current_price * trading_fee  # Deduct trading fee for exiting the position
                #print(f"Final Sell: Closing position at {current_price:.2f}, Balance: {balance:.2f}")

            # Backtest summary
            #print(f"[RESULT] Initial Balance: ${initial_balance:.2f}")
            #print(f"[RESULT] Final Balance: ${balance:.2f}")
            #print(f"[RESULT] Net Profit: ${balance - initial_balance:.2f} {((balance - initial_balance) / initial_balance) * 100: .2f}%")
            console.print(f"[purple][bold]st[/bold] [white]net profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%")
            #print(f"[st] Net Profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%")
            #print(f"[RESULT] $100.00 --> ${((1 + ((balance - initial_balance) / initial_balance)) * 100):.2f}")
        else:
            continue

    console.print("[purple][bold]st[/bold] [white]select option:\n  1) new\n  2) load\n  3) back-test\n  4) quit")
    choice = int(console.input("[purple][bold]st[/bold] [white]: "))

quit()
#**********************************************************************************************#