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

from preprocessing.preprocessing import get_latest_feature_values
from preprocessing.data_gathering import get_games, get_season_calendar
from resources.helper import load_configs, authenticate_api

def get_current_week(configuration):
    remaining_games = [
        (week.to_dict()['week'], week.to_dict()['season']) for week 
        in get_season_calendar(configuration, dt.now().year)
        if dt.now() <= dt.strptime(week.to_dict()["last_game_start"], "%Y-%m-%dT%H:%M:%S.%fZ")
    ]
    current_week, current_season = remaining_games[0]
    return current_season, current_week

def make_predictions(df, model, feature_configs):
    X = df[feature_configs["FEATURES"]]
    X.dropna(inplace=True)
    y_pred = model.predict(X)

    final_df = df.iloc[X.index,:]
    final_df.to_csv("test.csv")
    final_df["predicted_diff"] = y_pred
    final_df["spread_prediction"] = (final_df["consensus_spread(reversed)"] + final_df["predicted_diff"]) * -1
    final_df["spread"] = final_df["consensus_spread(reversed)"] * -1

    final_df.to_csv("./data/current_predictions.csv")
    _append_predictions_to_overall(final_df, model)

    submission_df = final_df[["id", "home_team", "away_team", "spread_prediction"]]
    submission_df.columns = ["id", "home", "away", "prediction"]
    submission_df = submission_df.set_index("id")
    submission_df.to_csv("./data/cfb_prediction_submission.csv")

def _append_predictions_to_overall(df, model):
    if os.path.exists("./data/running_predictions.csv"):
        history_df = pd.read_csv("./data/running_predictions.csv")
    else:
        history_df = pd.DataFrame()

    df["inference_date"] = dt.now()
    df["model_used"] = model.__class__.__name__

    pd.concat([history_df, df], axis=0).to_csv("./data/running_predictions.csv")
    