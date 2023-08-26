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

from training.training_functions import load_data_train_test_split, load_training_data
from resources.helper import load_configs

def train(model):
    feature_configs = load_configs("./configs/features.yml")
    X_train, X_test, y_train, y_test = load_data_train_test_split(feature_configs)

    model.fit(X_train, y_train)
    return model