import pandas as pd
import numpy as np

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from resources.helper import load_configs


def load_training_data(id, id_time):
    features_configs = load_configs("configs/features.yml")

    df = pd.read_csv(f"data/training_sets/{id}_{id_time}.csv")
    X = df[features_configs["FEATURES"]]
    y = df[features_configs["TARGET"]]
    return X, y


def load_data_train_test_split(id, id_time):
    features_configs = load_configs("configs/features.yml")
    model_configs = load_configs("configs/training_config.yml")

    df = pd.read_csv(f"data/training_sets/{id}_{id_time}.csv")
    df = df.sort_values("id").reset_index(drop=True)
    df = df[features_configs["FEATURES"] + [features_configs["TARGET"]]]
    df.dropna(inplace=True)

    test_size = model_configs.get("test_size")
    train = df.reset_index(drop=True).iloc[: int(len(df) * test_size)]
    test = df.reset_index(drop=True).iloc[int(len(df) * test_size) :]

    X_train = train[features_configs["FEATURES"]]
    y_train = train[features_configs["TARGET"]]

    X_test = test[features_configs["FEATURES"]]
    y_test = test[features_configs["TARGET"]]

    return X_train, X_test, y_train, y_test


def validate(model, X_train, X_test, y_train, y_test):

    metrics = {}

    # Training Scores
    print("*" * 50)
    print("*" * 50)
    y_pred = model.predict(X_train)
    train_df = pd.DataFrame({"y_pred": y_pred, "y_test": y_train})
    train_df["diff"] = train_df["y_pred"] - train_df["y_test"]
    train_rmse = np.sqrt(np.mean(train_df["diff"] ** 2))
    train_mae = np.mean(np.abs(train_df["diff"]))

    metrics["train_rmse"] = train_rmse
    print(f"Training RMSE: {train_rmse}")

    metrics["train_mae"] = train_mae
    print(f"Training MAE: {train_mae}")

    print("*" * 50)

    # Testing Scores
    y_pred = model.predict(X_test)
    test_df = pd.DataFrame({"y_pred": y_pred, "y_test": y_test})
    test_df["diff"] = test_df["y_pred"] - test_df["y_test"]
    test_rmse = np.sqrt(np.mean(test_df["diff"] ** 2))
    test_mae = np.mean(np.abs(test_df["diff"]))

    metrics["test_rmse"] = test_rmse
    print(f"Testing RMSE: {test_rmse}")

    metrics["test_mae"] = test_mae
    print(f"Testing MAE: {test_mae}")
    print("*" * 50)
    print("*" * 50)

    return metrics
