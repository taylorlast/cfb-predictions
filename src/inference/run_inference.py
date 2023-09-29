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

from preprocessing.preprocessing import get_latest_feature_values, join_features, update_primary_data
from preprocessing.data_gathering import get_games, get_season_calendar, get_betting_info
from resources.helper import load_configs, authenticate_api, load_model
from inference.inference_functions import get_current_week, make_predictions

def run_inference(configuration):
    season, week = get_current_week(configuration)
    update_primary_data(
        configuration=configuration,
        current_season=season,
        current_week=week
    )
    games = get_games(
        configuration=configuration, 
        year=season,
        week=week,
        only_fbs=True,
    )
    primary_stats = pd.read_csv("./data/stats_df.csv")
    stats = get_latest_feature_values(primary_stats)
    lines = get_betting_info(
        configuration=configuration, 
        year=season,
        week=week,
    )
    df = join_features(games, stats, lines)

    feature_configs = load_configs("./configs/features.yml")
    model = load_model("xgb") # Add model configs

    make_predictions(df, model, feature_configs)

    #TODO: Consider Dim Reduction
    #TODO: append predictions to running_predictions - consider adding a revision number to keep track of changes

# Load configs
api_configs_path = "./configs/api_configs.yml"
api_configs = load_configs(config_path=api_configs_path)

# Auth API
api_configuration = authenticate_api(api_key=api_configs["API_KEY"])

df = run_inference(api_configuration)