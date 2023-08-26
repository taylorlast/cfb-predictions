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

from preprocessing.preprocessing import get_latest_feature_values, join_features
from preprocessing.data_gathering import get_games, get_season_calendar, get_betting_info
from resources.helper import load_configs, authenticate_api
from inference.inference_functions import get_current_week

def run_inference(configuration):
    season, week = get_current_week(configuration)
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
    

    #TODO: Add filter to get only the features we want
    #TODO: Consider Dim Reduction
    #TODO: Read in model
    #TODO: Make predictions with model
    #TODO: Save predictions to current_predictions
    #TODO: append predictions to running_predictions
    return df
