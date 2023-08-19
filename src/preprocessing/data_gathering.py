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
        year:int,
        week:int=None,
        only_fbs:bool=False,
    ) -> pd.DataFrame:
    """
    gets basic game information from a given year and stores in a dataframe.

    parameters
    ----------
    configuration: cfbd.Configuration
        authenticated api session for cfbd
    year: int
        year for data
    week: int
        week for data
    only_fbs: bool
        whether to only include games between fbs teams
    
    returns
    -------
    year_games: pd.DataFrame
        df containing basic game info
    """

    api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
    if week is None:
        games = api_instance.get_games(year=year)
    else:
        games = api_instance.get_games(year=year, week=week)

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
        year:int,
        week:int=None
    )->pd.DataFrame:

    """
    retrieves advanced stats from a year and stores in a df.

    parameters
    ----------
    configuration: cfbd.Configuration
        authenticated api session for cfbd
    year: int
        year for data
    week: int
        week for data
    
    returns
    -------
    year_games: pd.DataFrame
        df containing advanced stats
    """

    api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))

    if week is None:
        stats = api_instance.get_advanced_team_game_stats(year=year)
    else:
        stats = api_instance.get_advanced_team_game_stats(year=year, week=week)

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
    
    # Add season -- started coming as null in 2022
    stats["season"] = year

    return stats

def get_betting_info(
        configuration: cfbd.Configuration,
        year:int,
        week:int=None,
    ) -> pd.DataFrame:
    """
    gets vegas spreads for games. the column will be a reversed spread.
    so if the home team is favored by 7, the spread will be 7 and not -7.

    reason: the point total difference needs to stay consistent with the spread.

    parameters
    ----------
    configuration: cfbd.Configuration
        authenticated api session for cfbd
    year: str
        year for data
    week: int
        week for data
    
    returns
    -------
    spreads_df: pd.DataFrame
        dataframe with spreads indexed by game id.
    """

    api_instance =cfbd.BettingApi(cfbd.ApiClient(configuration))
    if week is None:
        spreads = api_instance.get_lines(year=year)
    else:
        spreads = api_instance.get_lines(year=year, week=week)

    spreads_df =  pd.DataFrame().from_records(
        [
            s.to_dict()
            for s in spreads
        ]
    )
    spreads_df["lines"] = spreads_df["lines"]\
        .apply(
            lambda x: [book for book in x if book["provider"]=="consensus"]
        )
    
    spreads_df = spreads_df[spreads_df.lines.str.len() != 0].reset_index(drop=True)
    spreads_df["consensus_spread(reversed)"] = spreads_df["lines"]\
        .apply(
            lambda x: x[0]["spread"]
        ).astype(float) * -1
    
    return spreads_df