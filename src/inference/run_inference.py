import numpy as np
import pandas as pd

import cfbd

import requests
import json
import os
import sys

from datetime import datetime as dt


from preprocessing.preprocessing import DataProcessor
from preprocessing.data_gathering import (
    get_games,
    get_season_calendar,
    get_betting_info,
)
from resources.helper import load_configs, authenticate_api, load_model
from inference.inference_functions import get_current_week, make_predictions


def run_inference():
    data_processor = DataProcessor()
    season, week = get_current_week(data_processor.api_configuration)
    games = get_games(
        configuration=data_processor.api_configuration,
        year=season,
        week=week,
        only_fbs=True,
    )
    stats_last_year = pd.read_csv(f"data/raw/stats/{season - 1}.csv")
    stats_this_year = None
    try:
        stats_this_year = pd.read_csv(f"data/raw/stats/{season}.csv")
    except Exception as e:
        print(e)
    if stats_this_year:
        stats = pd.concat([stats_last_year, stats_this_year])
    else:
        stats = stats_last_year
    stats = data_processor.get_latest_feature_values(stats)
    lines = data_processor.get_betting_info(
        year=season,
        week=week,
    )
    df = data_processor.join_features(games, stats, lines)

    model, metadata = load_model()
    make_predictions(df, model, metadata, data_processor.api_configuration)
