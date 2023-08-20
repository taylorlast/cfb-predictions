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

    train_df = pd.merge(
        left=games_df, 
        right=add_suffix(stats, "home", ["id", "team"]), 
        how="left", 
        left_on=["id", "home_team"], 
        right_on=["id", "team"]
    )

    train_df = pd.merge(
        left=train_df, 
        right=add_suffix(stats, "away", ["id", "team"]), 
        how="left", 
        left_on=["id", "away_team"], 
        right_on=["id", "team"]
    )

    train_df = pd.merge(
        left=train_df,
        right=lines_df[["id", "consensus_spread(reversed)"]],
        on="id",
        how="left",
    )

    train_df["diff"] = train_df["point_diff"] - train_df["consensus_spread(reversed)"]

    train_df.to_csv(output_path)

def add_suffix(df, suffix, cols_to_exclude):
    for col in df.columns:
        if col not in cols_to_exclude:
            df = df.rename(columns={col: f"{col}_{suffix}"})
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

