import numpy as np
import pandas as pd

import cfbd

import requests
import json
import os
import sys

from datetime import datetime as dt

def get_games(
        configuration:cfbd.Configuration, 
        year:str,
        only_fbs:str=False,
    ) -> pd.DataFrame:
    """
    gets basic game information from a given year and stores in a dataframe.

    parameters
    ----------
    configuration: cfbd.Configuration
        authenticated api session for cfbd
    year: str
        year for data
    only_fbs: bool
        whether to only include games between fbs teams
    
    returns
    -------
    year_games: pd.DataFrame
        df containing basic game info
    """

    api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
    games = api_instance.get_games(year=year)

    year_games = pd.DataFrame().from_records([
        g.to_dict()
        for g in games
    ]
    )
    year_games['winner']=np.where(
        year_games.home_points > year_games.away_points, 
        year_games.home_team, 
        year_games.away_team
    )

    # "Spread will be the difference between home and away points"
    year_games['point_diff'] = year_games.home_points - year_games.away_points

    game_cols = [
        'id',
        'season',
        'week',
        'home_team',
        'away_team',
        'home_points',
        'away_points',
        'home_division',
        'away_division',
        'home_pregame_elo',
        'away_pregame_elo',
        'neutral_site',
        'point_diff',
        'winner',
    ]

    year_games = year_games[game_cols]

    if only_fbs:
        year_games = year_games[
            (year_games["home_division"] == "fbs")
            & (year_games["away_division"] == "fbs")
        ]

    year_games = year_games.reset_index(drop=True)
    return year_games

def get_game_stats(
        configuration:cfbd.Configuration,
        year:str
    )->pd.DataFrame:

    """
    retrieves advanced stats from a year and stores in a df.

    parameters
    ----------
    configuration: cfbd.Configuration
        authenticated api session for cfbd
    year: str
        year for data
    
    returns
    -------
    year_games: pd.DataFrame
        df containing advanced stats
    """

    api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))
    stats = api_instance.get_advanced_team_game_stats(year=year)

    stat_df = pd.DataFrame().from_records(
        [s.to_dict() for s in stats]
    )

    #offense columns
    offense_columns = list()
    offense = pd.json_normalize(stat_df['offense'])

    for i in offense.columns:
         offense_columns.append(i + '_offense')

    offense = pd.json_normalize(stat_df['offense'])
    offense.columns = offense_columns

    #defense columns
    defense_columns = list()
    defense = pd.json_normalize(stat_df['defense'])

    for i in defense.columns:
         defense_columns.append(i + '_defense')

    defense.columns = defense_columns

    #combine
    stats = pd.concat(
        [
            stat_df.drop(['offense','defense'],axis=1),
            offense,
            defense,
        ],
        axis=1
    )

    stats.rename(columns={'game_id':'id'},inplace=True)

    return stats