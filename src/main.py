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
from preprocessing.data_gathering import get_game_stats, get_games, get_betting_info, save_primary_data
from preprocessing.preprocessing import get_simple_rolling_stats, create_training_set

# Load configs
api_configs_path = "./configs/api_configs.yml"
api_configs = load_configs(config_path=api_configs_path)

# Auth API
api_configuration = authenticate_api(api_key=api_configs["API_KEY"])



def update_primary_sets(configuration, start_date, end_date):
    save_primary_data(configuration, start_date,end_date)

if __name__ == "__main__":
    # update_primary_sets(api_configuration, 2015,2022)
    create_training_set(output_path="./data/training.csv", only_fbs=True)