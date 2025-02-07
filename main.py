# local files
from src import models
from src import utils

# libraries
from rich.progress import Progress
from rich.console import Console

console = Console()
#welcome_text()
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
from sklearn.datasets import make_classification
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler

#import tensorflow as tf
from xgboost import XGBClassifier
from scipy.signal import argrelextrema
import joblib

# global variables
prompt = "st"
choice = 0
#knn = KNN(k=12)
df = []
predictions = []
loaded = False
choice_model = 0
X_train, X_test, y_train, y_test = [], [], [], []

# cli-program
while choice != 5:
    console.print("[purple][bold]"+ prompt +"[/bold] [white]► options:\n  1) new    2) load    3) back-test    4) options    5) quit")
    try:
        choice = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
    except:
        choice = 0

    # CREATE NEW MODEL #
    if choice == 1:
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► models:\n  1) knn    2) gradient boosting    3) random forest    4) quit")
        try:
            choice_model = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
        except:
            choice_model = 4

        if choice_model == 1:
                console.print("[purple][bold]"+ prompt +"[/bold] [white]► import data options:\n  1) load    2) fetch    3) quit")
                try:
                    data_todo = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
                except:
                    data_todo = 3

                if data_todo == 1:
                    df = pd.read_csv('data/100,000_15.csv')
                    console.print("[purple][bold]"+ prompt +"[/bold] [white]► data loaded")
                elif data_todo == 2:
                    n = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► kilocandles to regress: "))
                    exchange = utils.connect_to_exchange()
                    ohlcv = utils.fetch_data('ETH/USDT', '15m', exchange, n)
                    df = pd.DataFrame(ohlcv)
                    df.to_csv(f'data/{n * 1000}_15.csv', index=False)
                    console.print("[purple][bold]"+ prompt +"[/bold] [white]► data loaded")
                elif data_todo == 3:
                    continue

                console.print("[purple][bold]"+ prompt +"[/bold] [white]► creating model")
                console.print("[purple][bold]"+ prompt +"[/bold] [white]► using model settings 'model.conf'")

                # Feature engineering: Create additional features (e.g., moving averages, RSI)
                df['SMA_20'] = df['4'].rolling(window=20).mean()
                df['SMA_50'] = df['4'].rolling(window=50).mean()
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
                continue

            console.print("[purple][bold]"+ prompt +"[/bold] [white]► creating model")
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► using model settings 'model.conf'")

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

            '''
            cutoff = 0
            df['target'] = df.apply(
                lambda row: 1 if row['returns'] > cutoff else
                            0 if row['returns'] < -cutoff else
                            2,
                axis=1
            )

            # Drop rows without target
            #df = df.dropna()

            features = ['4', '5', 'lagged', 'RSI', 'ATR', 'BB_Upper', 'BB_Lower', 'SMA_20', 'SMA_50']
            X = df[features].values
            y = df['target'].values

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)
            #X_train, y_train = make_classification(n_samples=99900, n_classes=3, weights=[0.1, 0.1, 0.8], n_informative=10, n_features=13)

            model = XGBClassifier(n_estimators=200, learning_rate=0.005, max_depth=4)
            model.fit(X_train, y_train)

            # obtain the predictions
            predictions = model.predict(X_test)
            print(y_test)
            print(predictions)

            # output accuracy of the model
            accuracy = accuracy_score(y_test, predictions)
            console.print("[purple][bold]"+ prompt +f"[/bold] [white]► model accuracy: {accuracy * 100:.2f}%")

            # save model
            joblib.dump(model, 'xgboost_model.pkl')

            # load model
            loaded = True
            prompt = prompt + " (model.npz)"
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► model loaded")
            '''

            # Select features for the model
            features = ['1', '2', '3', '4', '5', 'lagged', 'lagged_2', 'RSI', 'ATR', 'CMO', 'CCI', 'ROC', 'SuperTrend_Direction', 'EMA', 'BB_Upper', 'BB_Lower']
            X = df[features].values
            y = df['target'].values

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)  # Transform test set without fitting again

            model = XGBClassifier(n_estimators=250, learning_rate=0.01, max_depth=6)
            model.fit(X_train, y_train)

            # obtain the predictions
            predictions = model.predict(X_test)
            print(y_test)
            print(predictions)

            # output accuracy of the model
            accuracy = accuracy_score(y_test, predictions)
            console.print("[purple][bold]"+ prompt +f"[/bold] [white]► model accuracy: {accuracy * 100:.2f}%")

            # save model
            joblib.dump(model, 'models/xgboost_model.joblib')

            # load model
            loaded = True
            prompt = prompt + " (xgboost_model.json)"
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► model loaded")

        elif choice_model == 3:
            print("asdf")

        elif choice_model == 4:
            continue

    # LOAD SAVED MODEL #
    elif choice == 2:
        '''
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► loaX, y = make_classification(n_samples=99900, n_classes=3, weights=[0.1, 0.1, 0.8], n_informative=5)ding model")
        knn.load_model('models/model.npz')

        console.print("[purple][bold]"+ prompt +"[/bold] [white]► making predictions")
        predictions = knn.predict(True)

        # Evaluate accuracy
        accuracy = np.mean(np.array(predictions) == knn.y_test)
        console.print(f"[purple][bold]"+ prompt +"[/bold] [white]► model accuracy: {accuracy * 100:.2f}%")

        loaded = True
        prompt = prompt + " (model.npz)"
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► model loaded")
        '''

        n = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► kilocandles to regress: "))
        exchange = utils.connect_to_exchange()
        ohlcv = utils.fetch_data('ETH/USDT', '5m', exchange, n)
        df = pd.DataFrame(ohlcv)
        df.to_csv(f'data/tmp_5.csv', index=False)

        df = pd.read_csv('data/tmp_5.csv')

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

        # Split the dataset            
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Transform test set without fitting again

        console.print("[purple][bold]"+ prompt +"[/bold] [white]► loading model")
        model = joblib.load("models/xgboost_model.joblib")

        console.print("[purple][bold]"+ prompt +"[/bold] [white]► making predictions")
        predictions = model.predict(X)
        print(predictions)

        # Evaluate accuracy
        accuracy = accuracy_score(y, predictions)
        console.print("[purple][bold]"+ prompt +f"[/bold] [white]► model accuracy: {accuracy * 100:.2f}%")

        loaded = True
        prompt = prompt + " (xgboost_model.json)"
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► model loaded")

    # BACK-TEST #
    elif choice == 3:
        if loaded == True:
            initial_balance = 100  # Starting capital in USD
            balance = initial_balance
            position = 0  # 1 = long, -1 = short, 0 = no position
            entry_price = 0 # Price at which the position is entered
            trading_fee = 0.001 # 0.075% trading fee per transaction
            good_trades = []
            bad_trades = []
            total_trades = 0

            plt.figure(figsize=(14, 7))
            actual_times = df['0'].iloc[len(X_train):len(X_train) + len(y_test)].values
            actual_prices = df['4'].iloc[len(X_train):len(X_train) + len(y_test)].values
            rsi_indicator = df['RSI'].iloc[len(X_train):len(X_train) + len(y_test)].values
            st_indicator = df['SuperTrend_Direction'].iloc[len(X_train):len(X_train) + len(y_test)].values

            rsi_up = False
            rsi_down = False

            # Backtest loop
            for i in range(len(predictions)):
                current_price = df['4'].iloc[len(X_train) + i]  # Current price of the asset
                signal = predictions[i]  # Predicted signal (1 = Buy, -1 = Sell, 0 = Hold)

                if rsi_indicator[i] > 80.00:
                    rsi_up = True
                    rsi_down = False
                elif rsi_indicator[i] < 20.00:
                    rsi_up = False
                    rsi_down = True

                if signal == 1 and position == 0 and rsi_down == True:  # Buy signal
                    position = 1
                    entry_price = current_price
                    balance -= balance * trading_fee  # Deduct trading fee for entering the position
                    plt.scatter(actual_times[i], actual_prices[i], color='green', s=150, label='Buy Signal')

                elif signal == 0 and position == 1 and rsi_up == True:  # Sell signal
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

            plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['4'], color='black', label='Data Points')
            #plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_small'], color='blue', label='Data Points')
            #axs[0].plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['EMA'], color='orange', label='Data Points')
            #plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_large'], color='red', label='Data Points')
            plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['max'], color='orange', marker='^', label='Sell Signal')
            plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min'], color='purple', marker='^', label='Sell Signal')
            #plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_large'], color='purple', marker='^', label='Sell Signal')
            #plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_small'], color='green', marker='^', label='Sell Signal')
            #axs[1].plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['SuperTrend_Direction'], color='purple', label='Data Points')
            plt.tight_layout()
            plt.show()

    # TMP LOAD BACK-TEST
    elif choice == 4:
        if loaded == True:
            initial_balance = 100  # Starting capital in USD
            balance = initial_balance
            position = 0  # 1 = long, -1 = short, 0 = no position
            entry_price = 0 # Price at which the position is entered
            trading_fee = 0.001 # 0.075% trading fee per transaction
            good_trades = []
            bad_trades = []
            total_trades = 0

            plt.figure(figsize=(14, 7))
            actual_times = df['0'].values
            actual_prices = df['4'].values
            rsi_indicator = df['RSI'].values
            st_indicator = df['SuperTrend_Direction'].values
            print("asdfasdf")
            print(rsi_indicator)

            rsi_up = False
            rsi_down = False

            # Backtest loop
            for i in range(len(predictions) - 1):
                current_price = df['4'].iloc[i]  # Current price of the asset
                signal = predictions[i]  # Predicted signal (1 = Buy, -1 = Sell, 0 = Hold)

                if rsi_indicator[i] > 80.00:
                    rsi_up = True
                    rsi_down = False
                elif rsi_indicator[i] < 20.00:
                    rsi_up = False
                    rsi_down = True

                if signal == 1 and position == 0 and rsi_down == True:  # Buy signal
                    position = 1
                    entry_price = current_price
                    balance -= balance * trading_fee  # Deduct trading fee for entering the position
                    plt.scatter(actual_times[i], actual_prices[i], color='green', s=150, label='Buy Signal')

                elif signal == 0 and position == 1 and rsi_up == True:  # Sell signal
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

                if position == 1 and (current_price - entry_price) / entry_price < -0.1:
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

            plt.plot(df.iloc[0:len(X)]['0'], df.iloc[0:len(X)]['4'], color='black', label='Data Points')
            #plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_small'], color='blue', label='Data Points')
            #axs[0].plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['EMA'], color='orange', label='Data Points')
            #plt.plot(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['smoothed_close_large'], color='red', label='Data Points')
            plt.scatter(df.iloc[0:len(X)]['0'], df.iloc[0:len(X)]['max'], color='orange', marker='^', label='Sell Signal')
            plt.scatter(df.iloc[0:len(X)]['0'], df.iloc[0:len(X)]['min'], color='purple', marker='^', label='Sell Signal')
            #plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_large'], color='purple', marker='^', label='Sell Signal')
            #plt.scatter(df.iloc[len(X_train):(len(X_train) + len(X_test))][0], df.iloc[len(X_train):(len(X_train) + len(X_test))]['min_small'], color='green', marker='^', label='Sell Signal')
            #axs[1].plot(df.iloc[len(X_train):(len(X_train) + len(X_test))]['0'], df.iloc[len(X_train):(len(X_train) + len(X_test))]['SuperTrend_Direction'], color='purple', label='Data Points')
            plt.tight_layout()
            plt.show()

    # PRINT HELP #
    #elif choice == 4:
    #    console.print("[purple][bold]"+ prompt +"[/bold] [white]► select option:\n  1) new    2) load    3) back-test    4) options    5) quit")

quit()