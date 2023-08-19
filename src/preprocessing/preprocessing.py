import numpy as np
import pandas as pd

import cfbd

import requests
import json
import os
import sys

from datetime import datetime as dt


def get_simple_rolling_stats(
        stats_df:pd.DataFrame, 
        period:int=7
    )->pd.DataFrame:
    """
    returns the rolling average features for each game.
    Uses only previous data by using the shift operator
    and doens't look into the future.

    parameters
    ----------
    stats_df: pd.DataFrame
        stats_df from cfbd api
    
    returns
    -------
    stats_df_rolling: pd.DataFrame
        rolling stats for all teams in cfb.
    """
    stats_df_rolling = (
        stats_df
        .groupby("team")
        .rolling(period, min_periods=1)
        .mean()
        .groupby("team")
        .shift(1)
        .reset_index()
        .drop(["season", "week", "level_1"], axis=1)
    )
    return stats_df_rolling