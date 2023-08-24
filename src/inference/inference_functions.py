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

def join_features(
        games,
        stats,
        lines
    ):
    pd.merge()

