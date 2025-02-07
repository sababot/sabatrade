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
            # load data
            df, skip = load_data()
            if skip == true:
                continue

            console.print("[purple][bold]"+ prompt +"[/bold] [white]► creating model")
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► using model settings 'model.conf'")

            # process data
            df, X, y = process_data(df)

            # normalize features
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

            # split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # train and define model
            #knn = KNN(k=12)
            knn.fit(X_train, y_train, X_test, y_test)

            # save model
            knn.save_model('models/model.npz')

            # make predictions
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► making predictions")
            predictions = knn.predict(True)

            # Evaluate accuracy
            accuracy = np.mean(np.array(predictions) == y_test)
            console.print("[purple][bold]"+ prompt +f"[/bold] [white]► model accuracy: {accuracy * 100:.2f}%")
                
            loaded = True
            prompt = prompt + " (model.npz)"
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► model loaded")

        elif choice_model == 2:
            # load data
            df, skip = load_data()
            if skip == true:
                continue

            console.print("[purple][bold]"+ prompt +"[/bold] [white]► creating model")
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► using model settings 'model.conf'")

            # process data
            df, X, y = process_data(df)

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)  # Transform test set without fitting again

            # define and train model
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
        n = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► kilocandles to regress: "))
        exchange = utils.connect_to_exchange()
        ohlcv = utils.fetch_data('ETH/USDT', '5m', exchange, n)
        df = pd.DataFrame(ohlcv)
        df.to_csv(f'data/tmp_5.csv', index=False)

        df = pd.read_csv('data/tmp_5.csv')

        df, X, y = utils.process_data(df)

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

                if rsi_indicator[i] > 60.00:
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