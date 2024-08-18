import pandas as pd
import numpy as np
from datetime import datetime as dt
import os
import gc
import tqdm
import uuid
import cfbd
from dotenv import load_dotenv

from resources.helper import authenticate_api, get_latest_version_number

load_dotenv()


class DataProcessor:

    def __init__(self):
        self.api_configuration = authenticate_api(api_key=os.environ.get("API_KEY"))

    def get_games(
        self,
        year: int,
        week: int = None,
        only_fbs: bool = False,
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

        api_instance = cfbd.GamesApi(cfbd.ApiClient(self.api_configuration))
        if week is None:
            games = api_instance.get_games(year=year)
        else:
            games = api_instance.get_games(year=year, week=week)

        year_games = pd.DataFrame().from_records([g.to_dict() for g in games])
        year_games["winner"] = np.where(
            year_games.home_points > year_games.away_points,
            year_games.home_team,
            year_games.away_team,
        )

        # Actual difference will be how much the home team won or lost by.
        # A negative number represents the team winning, while positive is losing
        year_games["point_diff"] = year_games.home_points - year_games.away_points
        year_games["point_diff"] = year_games["point_diff"] * -1

        game_cols = [
            "id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_points",
            "away_points",
            "home_division",
            "away_division",
            "home_pregame_elo",
            "away_pregame_elo",
            "neutral_site",
            "point_diff",
            "winner",
        ]

        year_games = year_games[game_cols]

        if only_fbs:
            year_games = year_games[
                (year_games["home_division"] == "fbs")
                & (year_games["away_division"] == "fbs")
            ]

        year_games = year_games.reset_index(drop=True)
        return year_games

    def get_game_stats(self, year: int, week: int = None) -> pd.DataFrame:
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

        api_instance = cfbd.StatsApi(cfbd.ApiClient(self.api_configuration))

        if week is None:
            stats = api_instance.get_advanced_team_game_stats(year=year)
        else:
            stats = api_instance.get_advanced_team_game_stats(year=year, week=week)

        stat_df = pd.DataFrame().from_records([s.to_dict() for s in stats])

        # offense columns
        offense_columns = list()
        offense = pd.json_normalize(stat_df["offense"])

        for i in offense.columns:
            offense_columns.append(i + "_offense")

        offense = pd.json_normalize(stat_df["offense"])
        offense.columns = offense_columns

        # defense columns
        defense_columns = list()
        defense = pd.json_normalize(stat_df["defense"])

        for i in defense.columns:
            defense_columns.append(i + "_defense")

        defense.columns = defense_columns

        # combine
        stats = pd.concat(
            [
                stat_df.drop(["offense", "defense"], axis=1),
                offense,
                defense,
            ],
            axis=1,
        )

        stats.rename(columns={"game_id": "id"}, inplace=True)

        # Add season -- started coming as null in 2022
        stats["season"] = year

        return stats

    def get_betting_info(
        self,
        year: int,
        week: int = None,
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

        api_instance = cfbd.BettingApi(cfbd.ApiClient(self.api_configuration))
        if week is None:
            spreads = api_instance.get_lines(year=year)
        else:
            spreads = api_instance.get_lines(year=year, week=week)

        spreads_df = pd.DataFrame().from_records([s.to_dict() for s in spreads])
        spreads_df["lines"] = spreads_df["lines"].apply(
            lambda x: [book for book in x if book["provider"] == "Bovada"]
        )

        spreads_df = spreads_df[spreads_df.lines.str.len() != 0].reset_index(drop=True)
        spreads_df["bovada_spread"] = (
            spreads_df["lines"].apply(lambda x: x[0]["spread"]).astype(float)
        )

        return spreads_df

    def get_season_calendar(
        self,
        year: int,
    ) -> pd.DataFrame:

        api_instance = cfbd.GamesApi(self.api_configuration)
        calendar = api_instance.get_calendar(year=year)
        return calendar

    def get_simple_rolling_stats(
        self, stats_df: pd.DataFrame, period: int = 7
    ) -> pd.DataFrame:
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
            stats_df.sort_values(by="id")
            .drop(["opponent", "week", "season"], axis=1)
            .set_index("id")
            .groupby("team")
            .rolling(period, min_periods=1)
            .mean()
            .groupby("team")
            .shift(1)
            .reset_index()
        )
        return stats_df_rolling

    def gather_initial_data(
        self, start_year, end_year=dt.now().year, update_raw_data=True
    ):

        if not os.path.isdir("data/raw"):
            os.mkdir("data/raw")
            os.mkdir("data/raw/stats")
            os.mkdir("data/raw/games")
            os.mkdir("data/raw/betting")

        if update_raw_data:
            for year in tqdm.tqdm(range(start_year, end_year + 1)):
                # Get rolling stats
                self.save_raw_data(year)
                gc.collect()

        stat_data = {}
        game_data = {}
        betting_data = {}

        for year in tqdm.tqdm(range(start_year, end_year + 1)):

            stat_data, game_data, betting_data = self.read_raw_data(
                year, stat_data, game_data, betting_data
            )

        combined_stat_df = pd.concat(stat_data.values())
        combined_stat_df = self.get_simple_rolling_stats(combined_stat_df)
        combined_games_df = pd.concat(game_data.values())
        combined_betting_df = pd.concat(betting_data.values())

        return combined_stat_df, combined_games_df, combined_betting_df

    def save_initial_data(
        self, start_year, end_year=dt.now().year, update_raw_data=True
    ):
        """
        grabs all inital data and versions it.

        parameters
        ----------
        start_year: int
            first year to grab data
        end_year: int
            last year to grab data
        update_raw_data: bool
            whether raw data needs to be updated. This is true when underlying data
            changes

        returns
        -------
        None
        """
        combined_stat_df, combined_games_df, combined_betting_df = (
            self.gather_initial_data(
                start_year=start_year,
                end_year=end_year,
                update_raw_data=update_raw_data,
            )
        )
        self.save_static_data(combined_stat_df, combined_games_df, combined_betting_df)

    def save_static_data(
        self, combined_stat_df, combined_games_df, combined_betting_df
    ):
        # Save games to static data directory
        game_version = get_latest_version_number("data/static/games")
        combined_games_df.to_csv(
            f"data/static/games/v{game_version + 1}.csv", index=False
        )
        # Save stats to static data directory
        stat_version = get_latest_version_number("data/static/rolling_stats")
        combined_stat_df.to_csv(
            f"data/static/rolling_stats/v{stat_version + 1}.csv", index=False
        )
        # Save betting info to static data directory
        betting_version = get_latest_version_number("data/static/betting_data")
        combined_betting_df.to_csv(
            f"data/static/betting_data/v{betting_version + 1}.csv", index=False
        )

    def save_raw_data(self, year):
        try:
            stats = self.get_game_stats(year=year)
            stats.to_csv(f"data/raw/stats/{year}.csv", index=False)
            del stats
        except Exception as e:
            print(e)
        # Get games
        try:
            games = self.get_games(year=year)
            games.to_csv(f"data/raw/games/{year}.csv", index=False)
            del games
        except Exception as e:
            print(e)
        # Get betting info
        try:
            betting = self.get_betting_info(year=year)
            betting.to_csv(f"data/raw/betting/{year}.csv", index=False)
            del betting
        except Exception as e:
            print(e)

    def read_raw_data(self, year, stat_data, game_data, betting_data):
        try:
            stats = pd.read_csv(f"data/raw/stats/{year}.csv")
            stat_data[year] = stats
            del stats
        except Exception as e:
            print(e)
        # Get games
        try:
            games = pd.read_csv(f"data/raw/games/{year}.csv")
            game_data[year] = games
            del games
        except Exception as e:
            print(e)
        # Get betting info
        try:
            betting = pd.read_csv(f"data/raw/betting/{year}.csv")
            betting_data[year] = betting
            del betting
        except Exception as e:
            print(e)

        return stat_data, game_data, betting_data

    def create_training_set(self, only_fbs):
        """
        creates a training set for static data and saves it to
        data/training_sets/{id}_{id_time}, which corresponds to
        a unique hash, and datetime that the training set was created.
        this combo will be used to identify models that used the same
        training set.

        parameters
        ---------
        only_fbs: bool
            whether or not to only include fbs games

        returns
        -------
        id: str
            unique hash for the training set
        id_time
            time the training set was created
        """

        game_version = get_latest_version_number("data/static/games")
        games_df = pd.read_csv(f"data/static/games/v{game_version}.csv")

        stat_version = get_latest_version_number("data/static/rolling_stats")
        stats_df = pd.read_csv(f"data/static/rolling_stats/v{stat_version}.csv")

        betting_version = get_latest_version_number("data/static/betting_data")
        lines_df = pd.read_csv(f"data/static/betting_data/v{betting_version}.csv")

        if only_fbs:
            games_df = games_df[
                (games_df["home_division"] == "fbs")
                & (games_df["away_division"] == "fbs")
            ]

        train_df = self.join_features(
            games=games_df, stats=stats_df, lines=lines_df, for_inference=False
        )

        """
        Creates diff column (target) -- represents the difference between
        actual point diff and the spread.

        Example:
        team A vs team B
        bovada_spread: -21
        point_diff: -38
        diff: -17

        This is saying that team A (home) should've been favored by 17 more
        than they were.
        """
        train_df["diff"] = train_df["point_diff"] - train_df["bovada_spread"]

        # generate unique hash
        id = str(uuid.uuid1().hex)

        # get time of training set
        id_time = dt.now().strftime(format="%Y%m%d_%H%M%S")

        output_path = f"data/training_sets/{id}_{id_time}.csv"
        train_df.to_csv(output_path, index=False)

        return id, id_time

    def add_suffix(self, df, suffix, cols_to_exclude):
        for col in df.columns:
            if col not in cols_to_exclude:
                df = df.rename(columns={col: f"{col}_{suffix}"})
        return df

    def join_features(
        self,
        games: pd.DataFrame,
        stats: pd.DataFrame,
        lines: pd.DataFrame,
        for_inference: bool = True,
    ):

        if for_inference:
            df = pd.merge(
                left=games,
                right=self.add_suffix(stats.drop("id", axis=1), "home", ["team"]),
                how="left",
                left_on=["home_team"],
                right_on=["team"],
            )

            df = pd.merge(
                left=df,
                right=self.add_suffix(stats.drop("id", axis=1), "away", ["team"]),
                how="left",
                left_on=["away_team"],
                right_on=["team"],
            )
        else:
            df = pd.merge(
                left=games,
                right=self.add_suffix(stats, "home", ["id", "team"]),
                how="left",
                left_on=["id", "home_team"],
                right_on=["id", "team"],
            )

            df = pd.merge(
                left=df,
                right=self.add_suffix(stats, "away", ["id", "team"]),
                how="left",
                left_on=["id", "away_team"],
                right_on=["id", "team"],
            )

        df = pd.merge(
            left=df,
            right=lines[["id", "bovada_spread"]],
            on="id",
            how="left",
        )

        return df

    def get_latest_feature_values(
        self, stats_df: pd.DataFrame, period: int = 7
    ) -> pd.DataFrame:
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
            stats_df.sort_values("id")
            .drop(["opponent", "week", "season"], axis=1)
            .set_index("id")
            .groupby("team")
            .rolling(period, min_periods=1)
            .mean()
            .groupby("team")
            .tail(1)
            .reset_index()
        )
        return stats_df_rolling

    def update_data(self):

        year = dt.now().year
        # Get rolling stats
        self.save_raw_data(year)

        stat_data = {}
        game_data = {}
        betting_data = {}

        stat_data, game_data, betting_data = self.read_raw_data(
            year, stat_data, game_data, betting_data
        )

        try:
            combined_stat_df = pd.concat(stat_data.values())
            combined_stat_df = self.get_simple_rolling_stats(combined_stat_df)
            combined_games_df = pd.concat(game_data.values())
            combined_betting_df = pd.concat(betting_data.values())

            self.save_static_data(
                combined_stat_df, combined_games_df, combined_betting_df
            )

        except Exception as e:
            print(e)

        gc.collect()
