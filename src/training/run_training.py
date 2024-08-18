import xgboost as xgb

import pandas as pd

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from training.training_functions import validate


def train(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor()
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model trained")

    metrics = validate(model, X_train, X_test, y_train, y_test)

    return model, metrics
