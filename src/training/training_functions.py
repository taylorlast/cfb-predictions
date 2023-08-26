import numpy as np
import pandas as pd

import cfbd

import requests
import json
import os
import sys

from datetime import datetime as dt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from resources.helper import load_configs

def load_training_data(features_configs):
    df = pd.read_csv("./data/training.csv")
    X = df[features_configs["FEATURES"]]
    y = df[features_configs["TARGET"]]
    return X, y

def load_data_train_test_split(features_configs):
    df = df = pd.read_csv("./data/training.csv")
    train = df[df["season"] <= 2022]
    test = df[df["season"] > 2022]

    X_train = train[features_configs["FEATURES"]]
    y_train = train[features_configs["TARGET"]]

    X_test = test[features_configs["FEATURES"]]
    y_test = test[features_configs["TARGET"]]

    return X_train, X_test, y_train, y_test
