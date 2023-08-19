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

# Local Imports
from resources.helper import load_configs, authenticate_api
from preprocessing.data_gathering import get_game_stats, get_games, get_betting_info

# Load configs
api_configs_path = "./configs/api_configs.yml"
api_configs = load_configs(config_path=api_configs_path)

# Auth API
api_configuration = authenticate_api(api_key=api_configs["API_KEY"])

if __name__ == "__main__":
    games = get_games(configuration=api_configuration, year=2022, only_fbs=True)
    stats = get_game_stats(configuration=api_configuration, year=2022)
    lines = get_betting_info(configuration=api_configuration, year=2022)