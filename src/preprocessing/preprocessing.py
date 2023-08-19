import numpy as np
import pandas as pd

import cfbd

import requests
import json
import os
import sys

from datetime import datetime as dt


def get_simple_rolling_stats(stats_df, period=7):
    return (
        stats_df
        .groupby("team")
        .rolling(period, min_periods=1)
        .mean()
        .groupby("team")
        .shift(1)
        .reset_index()
        .drop(["season", "week", "level_1"], axis=1)
    )