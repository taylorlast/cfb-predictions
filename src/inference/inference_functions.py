import numpy as np
import pandas as pd

import cfbd

import requests
import json
import os
import sys

from datetime import datetime as dt

from preprocessing.data_gathering import get_games, get_season_calendar
from resources.helper import load_configs, authenticate_api, get_latest_version_number


def get_current_week(configuration):
    remaining_games = [
        (week.to_dict()["week"], week.to_dict()["season"])
        for week in get_season_calendar(configuration, dt.now().year)
        if dt.now()
        <= dt.strptime(week.to_dict()["last_game_start"], "%Y-%m-%dT%H:%M:%S.%fZ")
    ]
    current_week, current_season = remaining_games[0]
    return current_season, current_week


def make_predictions(df, model, metadata, configuration):
    prediction_metadata = {}
    X = df[metadata["features"]]
    X.dropna(inplace=True)
    y_pred = model.predict(X)

    prediction_df = df.iloc[X.index, :]
    prediction_df[f"predicted_{metadata['target']}"] = y_pred

    season, week = get_current_week(configuration)
    if not os.path.isdir(f"data/predictions/{season}_{week}"):
        os.mkdir(f"data/predictions/{season}_{week}")

    v_num = get_latest_version_number(f"data/predictions/{season}_{week}") + 1
    prediction_df.to_csv(f"data/predictions/{season}_{week}/v{v_num}.csv", index=False)

    prediction_metadata["model_metadata"] = metadata
    prediction_metadata["prediction_version_number"] = v_num

    # If target is 'diff' (difference between actaul point difference and spread),
    # then add to get predicted spread
    if metadata.get("target") == "diff":
        submission_df = prediction_df[
            [
                "id",
                "home_team",
                "away_team",
                f"predicted_{metadata['target']}",
                "bovada_spread",
            ]
        ]
        submission_df["prediction"] = (
            submission_df[f"predicted_{metadata['target']}"]
            + submission_df["bovada_spread"]
        )
        submission_df = submission_df[["id", "home_team", "away_team", "prediction"]]
    else:
        submission_df = prediction_df[
            ["id", "home_team", "away_team", f"predicted_{metadata['target']}"]
        ]
        submission_df = submission_df.rename(
            columns={f"predicted_{metadata['target']}": "prediction"}
        )
    submission_df.columns = ["id", "home", "away", "prediction"]
    submission_df.to_csv(
        f"data/predictions/{season}_{week}/submission.csv", index=False
    )
