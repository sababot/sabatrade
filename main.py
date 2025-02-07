# local files
from src import models
from src import utils

# libraries
from rich.progress import Progress
from rich.console import Console

console = Console()
utils.welcome_text()
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
start, stop = 0, 0

# cli-program
while choice != 5:
    console.print("[purple][bold]"+ prompt +"[/bold] [white]► options:\n  1) new    2) load    3) back-test    4) options    5) quit")
    try:
        choice = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
    except:
        choice = 0

    # CREATE NEW MODEL #
    if choice == 1:
        console.print("[purple][bold]"+ prompt +"[/bold] [white]► models:\n  1) knn    2) gradient boosting    3) random forest    4) back")
        try:
            choice_model = int(console.input("[purple][bold]"+ prompt +"[/bold] [white]► "))
        except:
            choice_model = 4

        # KNN
        if choice_model == 1:
            # load data
            df, skip = utils.load_data(prompt)
            if skip == True:
                print("\n")
                continue

            # process data
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► processing data")
            df, X, y = utils.process_data(df)

            # normalize features
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

            # split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            console.print("[purple][bold]"+ prompt +"[/bold] [white]► creating model")
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► using model settings 'model.conf'")

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

        # XGBOOST
        elif choice_model == 2:
            # load data
            df, skip = utils.load_data(prompt)
            if skip == True:
                print("\n")
                continue

            # process data
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► processing data")
            df, X, y = utils.process_data(df)

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)  # Transform test set without fitting again

            console.print("[purple][bold]"+ prompt +"[/bold] [white]► creating model")
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► using model settings 'model.conf'")

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

            # back-test variables
            start = len(X_train)
            stop = len(X_train) + len(X_test)

        # TBD
        elif choice_model == 3:
            print("asdf")

        # BACK
        elif choice_model == 4:
            continue

    # LOAD SAVED MODEL #
    elif choice == 2:
        df, skip = utils.load_data(prompt)

        if skip == False:
            console.print("[purple][bold]"+ prompt +"[/bold] [white]► processing data")
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

            # back-test variables
            start = 0
            stop = len(X)

    # BACK-TEST #
    elif choice == 3:
        if loaded == True:
            utils.back_test(df, predictions, prompt, start, stop)

    # SEPERATE ITERATIONS
    print("\n")

quit()