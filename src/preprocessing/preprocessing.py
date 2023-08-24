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

from preprocessing.data_gathering import get_betting_info, get_game_stats, get_games

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
        .set_index("id")
        .groupby("team")
        .rolling(period, min_periods=1)
        .mean()
        .groupby("team")
        .shift(1)
        .reset_index()
        .drop(["season", "week", "Unnamed: 0"], axis=1)
    )
    return stats_df_rolling

def create_training_set(output_path, only_fbs):
    """
    overwrites a training set with new features 
    """
    games_df = pd.read_csv("./data/games_df.csv")
    stats_df = pd.read_csv("./data/stats_df.csv")
    lines_df = pd.read_csv("./data/betting_df.csv")

    if only_fbs:
        games_df = games_df[
            (games_df["home_division"] == "fbs")
            & (games_df["away_division"] == "fbs")
        ]
    stats = get_simple_rolling_stats(stats_df)

    train_df = join_features(
        games=games_df,
        stats=stats,
        lines=lines_df,
        for_inference=False
    )

    train_df["diff"] = train_df["point_diff"] - train_df["consensus_spread(reversed)"]

    train_df.to_csv(output_path)

def add_suffix(df, suffix, cols_to_exclude):
    for col in df.columns:
        if col not in cols_to_exclude:
            df = df.rename(columns={col: f"{col}_{suffix}"})
    return df

def join_features(
        games:pd.DataFrame,
        stats:pd.DataFrame,
        lines:pd.DataFrame,
        for_inference:bool=True
    ):

    if for_inference:
        df = pd.merge(
            left=games, 
            right=add_suffix(stats.drop("id", axis=1), "home", ["team"]), 
            how="left", 
            left_on=["home_team"], 
            right_on=["team"]
        )

        df = pd.merge(
            left=df, 
            right=add_suffix(stats.drop("id", axis=1), "away", ["team"]), 
            how="left", 
            left_on=["away_team"], 
            right_on=["team"]
        )
    else:
        df = pd.merge(
            left=games, 
            right=add_suffix(stats, "home", ["id", "team"]), 
            how="left", 
            left_on=["id", "home_team"], 
            right_on=["id", "team"]
        )

        df = pd.merge(
            left=df, 
            right=add_suffix(stats, "away", ["id", "team"]), 
            how="left", 
            left_on=["id", "away_team"], 
            right_on=["id", "team"]
        )

    df = pd.merge(
        left=df,
        right=lines[["id", "consensus_spread(reversed)"]],
        on="id",
        how="left",
    )

    return df

def get_latest_feature_values(
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
        .set_index("id")
        .groupby("team")
        .rolling(period, min_periods=1)
        .mean()
        .drop(["season", "week", "Unnamed: 0"], axis=1)
        .groupby("team")
        .tail(1)
        .reset_index()
    )
    return stats_df_rolling

def _add_new_data(df, path):
    """
    grabs the most recent week of data and writes it to
    the primary games data.

    parameters
    ----------
    df: pd.DataFrame
        df with information for all completed games
    path: str
        specifies where to read and write the data to

    returns
    ------
    None
    """
    primary_df = pd.read_csv(path)
    ids_to_add = [i for i in df["id"].values if i not in primary_df["id"].values]
    temp_df = pd.DataFrame({"id": ids_to_add})
    df = pd.merge(df, temp_df, how="inner", on="id")
    final_df = pd.concat([primary_df, df], axis=0)
    final_df.to_csv(path)
    del temp_df

def update_primary_data(configuration, current_season, current_week):
    """
    updates the primary data sets (games, stats, betting info)

    parameters
    ----------
    configuration: cfbd.Configuration
        authenticated api session for cfbd
    current_season: int
        current season for the cfb year
    current_week: int
        current week for cfb year

    returns
    --------
    None 
    """

    games = get_games(
        configuration=configuration,
        year=current_season,
        week=current_week,
        only_fbs=True
    )
    stats = get_game_stats(
        configuration=configuration,
        year=current_season,
        week=current_week
    )
    lines = get_betting_info(
        configuration=configuration,
        year=current_season,
        week=current_week
    )

    _add_new_data(games, "./data/games_df.csv")
    _add_new_data(stats, "./data/stats_df.csv")
    _add_new_data(lines, "./data/betting_df.csv")