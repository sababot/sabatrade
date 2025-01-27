# local files
from models import KNN
from utils import welcome_text, connect_to_exchange, fetch_data

# libraries
from rich.progress import Progress
from rich.console import Console

console = Console()
welcome_text()
console.print("[purple][bold]st[/bold] [white]► initializing libraries")

import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ccxt
import time
import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from xgboost import XGBClassifier
from scipy.signal import argrelextrema
import joblib

# global variables
prompt = "st"
choice = 0
knn = KNN(k=12)
df = []
predictions = []
loaded = False
choice_model = 0
X_train, X_test, y_train, y_test = [], [], [], []

# initial choice
'''
console.print("[purple][bold]"+ prompt +"[/bold] [white]► select option:\n  1) new    2) load    3) back-test    4) options    5) quit")
try:
    choice = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
except:
    choice = 0
'''

# cli-program
while choice != 5:
    console.print("[purple][bold]"+ prompt +"[/bold] [white]► options:\n  1) new    2) load    3) back-test    4) options    5) quit")
    try:
        choice = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
    except:
        choice = 0

    if choice == 1:
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► models:\n  1) knn    2) gradient boosting    3) random forest    4) quit")
        try:
            choice_model = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
        except:
            choice_model = 4

        if choice_model == 1:
                n = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► kilocandles to regress: "))
                exchange = connect_to_exchange()
                ohlcv = fetch_data('ETH/USDT', '1m', exchange, n)

                console.print("[purple][bold]"+ prompt +"[/bold] [white]► creating model")

                # Load data (use your own dataset or fetch it via an API)
                # Example: Load a CSV file with OHLCV data
                df = pd.DataFrame(ohlcv)

                # Feature engineering: Create additional features (e.g., moving averages, RSI)
                df['SMA_20'] = df[4].rolling(window=20).mean()
                df['SMA_50'] = df[4].rolling(window=50).mean()
                #df['RSI'] = 100 - (100 / (1 + df[4].pct_change().rolling(window=14).apply(lambda x: (x[x > 0].sum() / -x[x < 0].sum()) if x[x < 0].sum() != 0 else 0)))

                df['RSI'] = ta.rsi(df[4], length=14)
                df['CMO'] = ta.cmo(df[4], length=14)
                df['ROC'] = ta.roc(df[4], length=14)
                df["CCI"] = ta.cci(df[2], df[3], df[4], length=14)
                df['ATR'] = ta.atr(df[2], df[3], df[4])
                supertrend = ta.supertrend(df[2], df[3], df[4], length=10, multiplier=3)
                df['SuperTrend'] = supertrend[f'SUPERT_10_3.0']  # SuperTrend column name format is SUPERT_length_multiplier
                df['SuperTrend_Direction'] = supertrend[f'SUPERTd_10_3.0']  # Trend direction: 1 (bullish), -1 (bearish)

                bollinger = ta.bbands(df[4], length=20, std=2)  # Default length is 20, std is 2
                df['BB_Middle'] = bollinger['BBM_20_2.0']  # Bollinger Middle Band
                df['BB_Upper'] = bollinger['BBU_20_2.0']   # Bollinger Upper Band
                df['BB_Lower'] = bollinger['BBL_20_2.0']   # Bollinger Lower Band

                # Drop NaN values after feature calculation
                df = df.dropna()

                # Define the target variable (e.g., price increase or decrease over the next 5 timesteps)
                future_steps_buy = 18
                future_steps_sell = 8
                cutoff_buy = 0
                cutoff_sell = 0
                df['next_buy'] = df[4].shift(-future_steps_buy)
                df['next_sell'] = df[4].shift(-future_steps_sell)
                df['target'] = df.apply(
                    lambda row: 1 if (row['next_buy'] - row[4]) / row[4] > cutoff_buy else
                                0 if (row['next_sell'] - row[4]) / row[4] < -cutoff_sell else
                                2,
                    axis=1
                )

                # Drop rows without target
                df = df.dropna()

                # Select features for the model
                features = [5, 'RSI', 'ROC', 'ATR', 'BB_Upper', 'BB_Lower']
                X = df[features].values
                y = df['target'].values

                # Normalize features
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)

                # Split the dataset
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                knn.fit(X_train, y_train, X_test, y_test)
                knn.save_model('models/model.npz')

                console.print("[purple][bold]"+ prompt +"[/bold] [white]► making predictions")
                predictions = knn.predict(True)

                # Evaluate accuracy
                accuracy = np.mean(np.array(predictions) == y_test)
                console.print("[purple][bold]"+ prompt +f"[/bold] [white]► model accuracy: {accuracy * 100:.2f}%")
                
                loaded = True
                prompt = prompt + " (model.npz)"
                console.print("[purple][bold]"+ prompt +"[/bold] [white]► model loaded")

        elif choice_model == 2:
            n = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► kilocandles to regress: "))
            exchange = connect_to_exchange()
            ohlcv = fetch_data('ETH/USDT', '5m', exchange, n)

            console.print("[purple][bold]"+ prompt +"[/bold] [white]► creating model")

            # Load data (use your own dataset or fetch it via an API)
            # Example: Load a CSV file with OHLCV data
            df = pd.DataFrame(ohlcv)

            # Feature engineering: Create additional features (e.g., moving averages, RSI)
            df['SMA_20'] = df[4].rolling(window=20).mean()
            df['SMA_50'] = df[4].rolling(window=50).mean()
            #df['RSI'] = 100 - (100 / (1 + df[4].pct_change().rolling(window=14).apply(lambda x: (x[x > 0].sum() / -x[x < 0].sum()) if x[x < 0].sum() != 0 else 0)))

            df['RSI'] = ta.rsi(df[4], length=14)
            df['CMO'] = ta.cmo(df[4], length=14)
            df['ROC'] = ta.roc(df[4], length=14)
            df["CCI"] = ta.cci(df[2], df[3], df[4], length=14)
            df['ATR'] = ta.atr(df[2], df[3], df[4])
            supertrend = ta.supertrend(df[2], df[3], df[4], length=10, multiplier=3)
            df['SuperTrend'] = supertrend[f'SUPERT_10_3.0']  # SuperTrend column name format is SUPERT_length_multiplier
            df['SuperTrend_Direction'] = supertrend[f'SUPERTd_10_3.0']  # Trend direction: 1 (bullish), -1 (bearish)

            bollinger = ta.bbands(df[4], length=20, std=2)  # Default length is 20, std is 2
            df['BB_Middle'] = bollinger['BBM_20_2.0']  # Bollinger Middle Band
            df['BB_Upper'] = bollinger['BBU_20_2.0']   # Bollinger Upper Band
            df['BB_Lower'] = bollinger['BBL_20_2.0']   # Bollinger Lower Band

            df['smoothed_close_small'] = df[4].rolling(window=250).mean()
            #df['smoothed_close_large'] = df[4].rolling(window=60).mean()

            #df['min_small'] = df[4][(df['smoothed_close_small'].shift(100) > df['smoothed_close_small']) & (df['smoothed_close_small'].shift(-100) > df['smoothed_close_small'])]
            #df['max_small'] = df[4][(df['smoothed_close_small'].shift(100) < df['smoothed_close_small']) & (df['smoothed_close_small'].shift(-100) < df['smoothed_close_small'])]
            #df['min_large'] = df[4][(df['smoothed_close_large'].shift(15) > df['smoothed_close_large']) & (df['smoothed_close_large'].shift(-15) > df['smoothed_close_large'])]
            #df['max_large'] = df[4][(df['smoothed_close_large'].shift(15) < df['smoothed_close_large']) & (df['smoothed_close_large'].shift(-15) < df['smoothed_close_large'])]

            df['max'] = df[4].iloc[argrelextrema(df['smoothed_close_small'].values, np.greater_equal, order=5)[0]]
            df['min'] = df[4].iloc[argrelextrema(df['smoothed_close_small'].values, np.less_equal, order=5)[0]]

            df['target'] = np.nan  # Default is no action
            df.loc[df['min'].notna(), 'target'] = 1  # Buy at lows
            df.loc[df['max'].notna(), 'target'] = 0  # Sell at highs

            df['target'] = df['target'].fillna(method='ffill').fillna(method='bfill')

            print(df['target'])

            period = 1
            df['returns'] = df[4].pct_change(periods=-period)
            df['lagged'] = df[4].shift(period)
            #df['lagged_forward'] = df[4].shift(-period)

            '''
            cutoff = 0.02
            df['target'] = df.apply(
                lambda row: 1 if row['returns'] > cutoff else
                            0 if row['returns'] < cutoff else
                            2,
                axis=1
            )
            '''

            # Drop rows without target
            #df = df.dropna()

            # Select features for the model
            features = [1, 2, 3, 4, 5, 'returns', 'lagged', 'RSI', 'ATR', 'SuperTrend_Direction', 'BB_Upper', 'BB_Lower']
            X = df[features].values
            y = df['target'].values

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

            model = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=8)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            print(y_test)
            print(predictions)

            accuracy = accuracy_score(y_test, predictions)
            console.print("[purple][bold]"+ prompt +f"[/bold] [white]► model accuracy: {accuracy * 100:.2f}%")

            joblib.dump(model, 'xgboost_model.pkl')

            loaded = True
            prompt = prompt + " (model.npz)"
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► model loaded")

        elif choice_model == 3:
            print("asdf")

        elif choice_model == 4:
            continue

    elif choice == 2:
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► loading model")
        knn.load_model('models/model.npz')

        console.print("[purple][bold]"+ prompt +"[/bold] [white]► making predictions")
        predictions = knn.predict(True)

        # Evaluate accuracy
        accuracy = np.mean(np.array(predictions) == knn.y_test)
        console.print(f"[purple][bold]"+ prompt +"[/bold] [white]► model accuracy: {accuracy * 100:.2f}%")

        loaded = True
        prompt = prompt + " (model.npz)"
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► model loaded")

    elif choice == 3:
        if loaded == True:
            initial_balance = 100  # Starting capital in USD
            balance = initial_balance
            position = 0  # 1 = long, -1 = short, 0 = no position
            entry_price = 0  # Price at which the position is entered
            trading_fee = 0 #0.001  # 0.075% trading fee per transaction

            plt.figure(figsize=(14, 7))
            actual_times = df[0].iloc[len(X_train):len(X_train) + len(y_test)].values
            actual_prices = df[4].iloc[len(X_train):len(X_train) + len(y_test)].values

            # Backtest loop
            for i in range(len(predictions)):
                current_price = df[4].iloc[len(X_train) + i]  # Current price of the asset
                signal = predictions[i]  # Predicted signal (1 = Buy, -1 = Sell, 0 = Hold)

                if signal == 1:  # Buy signal
                    if position == 0:  # Open a long position if no position is open
                        position = 1
                        entry_price = current_price
                        balance -= balance * trading_fee  # Deduct trading fee for entering the position
                        print(f"[RESULT] Buy: Entering at {entry_price:.2f}, Balance: {balance:.2f}")
                        #print(f"[RESULT] Buy")
                        plt.scatter(actual_times[i], actual_prices[i], color='green', s=100, label='Buy Signal')

                elif signal == 0:  # Sell signal
                    if position == 1:  # Close the long position if it's open
                        balance += ((current_price - entry_price) / entry_price) * balance  # Calculate profit/loss
                        balance -= balance * trading_fee  # Deduct trading fee for exiting the position
                        position = 0
                        print(f"Sell: Exiting at {current_price:.2f}, Balance: {balance:.2f}")
                        console.print(f"[purple][bold]st[/bold] [white]► trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
                        #print(f"[st] Trade {((current_price - entry_price) / entry_price) * 100: .2f}%")
                        plt.scatter(actual_times[i], actual_prices[i], color='red', s=100, label='Sell Signal')

                elif signal == 2:  # Hold signal
                    '''
                    if (position == 1 and (current_price - entry_price) / entry_price < -0.025):
                        balance += ((current_price - entry_price) / entry_price) * balance  # Calculate profit/loss
                        balance -= balance * trading_fee  # Deduct trading fee for exiting the position
                        position = 0
                        #print(f"Sell: Exiting at {current_price:.2f}, Balance: {balance:.2f}, -0.01%")
                        #print(f"[RESULT] Sell -1%")
                        console.print(f"[purple][bold]st[/bold] [white]► liquidation -0.5%")
                        #print(f"[ST] Liquidation -1.00%")
                        plt.scatter(actual_times[i], actual_prices[i], color='red', s=100, label='Sell Signal')

                    if (position == 1 and (current_price - entry_price) / entry_price > 0.05):
                        balance += ((current_price - entry_price) / entry_price) * balance  # Calculate profit/loss
                        balance -= balance * trading_fee  # Deduct trading fee for exiting the position
                        position = 0
                        #print(f"Sell: Exiting at {current_price:.2f}, Balance: {balance:.2f}, -0.01%")
                        #print(f"[RESULT] Sell -1%")
                        console.print(f"[purple][bold]st[/bold] [white]► top 5%")
                        #print(f"[ST] Liquidation -1.00%")
                        plt.scatter(actual_times[i], actual_prices[i], color='red', s=100, label='Sell Signal')
                    '''

                    continue  # Do nothing and move to the next step

            #if position == 1:
                #balance += (current_price - entry_price) / entry_price * balance
                #balance -= current_price * trading_fee  # Deduct trading fee for exiting the position
                #print(f"Final Sell: Closing position at {current_price:.2f}, Balance: {balance:.2f}")

            # Backtest summary
            #print(f"[RESULT] Initial Balance: ${initial_balance:.2f}")
            #print(f"[RESULT] Final Balance: ${balance:.2f}")
            #print(f"[RESULT] Net Profit: ${balance - initial_balance:.2f} {((balance - initial_balance) / initial_balance) * 100: .2f}%")
            console.print("[purple][bold]"+ prompt +f"[/bold] [white]► net profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%")
            #print(f"[st] Net Profit: {((balance - initial_balance) / initial_balance) * 100: .2f}%")
            #print(f"[RESULT] $100.00 --> ${((1 + ((balance - initial_balance) / initial_balance)) * 100):.2f}")

            '''
            for i, pred in enumerate(predictions):
                if pred == 1:  # Buy signal
                    plt.scatter(actual_times[i], actual_prices[i], color='green', label='Buy Signal' if i == 0 else "")
                elif pred == -1:  # Sell signal
                    plt.scatter(actual_times[i], actual_prices[i], color='red', label='Sell Signal' if i == 0 else "")
            '''

            plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))][4], color='black', label='Data Points')
            plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_small'], color='blue', label='Data Points')
            #plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_large'], color='red', label='Data Points')
            plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['max'], color='orange', marker='^', label='Sell Signal')
            plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min'], color='purple', marker='^', label='Sell Signal')
            #plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_large'], color='purple', marker='^', label='Sell Signal')
            #plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_small'], color='green', marker='^', label='Sell Signal')
            plt.show()

    elif choice == 4:
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► select option:\n  1) new    2) load    3) back-test    4) options    5) quit")

quit()