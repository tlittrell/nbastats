import time
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from toolz.functoolz import pipe
from tqdm import tqdm


def scrape_all_schedule_tables(parameters: dict) -> pd.DataFrame:
    schedule_tables = parameters["schedule_tables_to_scrape"]
    schedules = []
    for season, month in schedule_tables:
        url = (
            f"https://www.basketball-reference.com/leagues/NBA_{season}_games-"
            f"{month.lower()}.html"
        )
        schedules.append(get_raw_schedule_table(url))
        time.sleep(0.5)

    result: pd.DataFrame = pd.concat(schedules).dropna()
    return result


def get_raw_schedule_table(url: str) -> pd.DataFrame:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "lxml")
    parsed_table = soup.find_all("table")[0]
    data = [
        [
            td.a["href"] if td.find("a") else "".join(td.stripped_strings)
            for td in row.find_all("td")
        ]
        for row in parsed_table.find_all("tr")
    ]
    cols = [
        "start_eastern_time",
        "visitor_team_url",
        "visitor_points",
        "home_team_url",
        "home_points",
        "boxscore_url",
        "overtime",
        "attendance",
        "notes",
    ]
    df = pd.DataFrame(data[1:], columns=cols)
    return df


def scrape_all_boxscores(schedule: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas()
    result = schedule.progress_apply(
        lambda row: scrape_single_boxscore(
            url=f"https://www.basketball-reference.com{row['boxscore_url']}",
            home_team=row["home_team"],
            visitor_team=row["visitor_team"],
            home_points=row["home_points"],
            visitor_points=row["visitor_points"],
            date=row["game_date"],
        ),
        axis=1,
    )
    return pd.concat(result.to_list())


def scrape_single_boxscore(
    url: str,
    home_team: str,
    visitor_team: str,
    home_points: int,
    visitor_points: int,
    date: np.datetime64,
) -> pd.DataFrame:
    """
    Get a dataframe with the player boxscores for a given game on the schedule.
    """

    def include_schedule_info(
        df: pd.DataFrame, team: str, opponent: str, points: int, date: np.datetime64
    ) -> pd.DataFrame:
        # Helper function to retain schedule level info in the boxscore data
        df["team"] = team
        df["opponent"] = opponent
        df["team_total_points"] = points
        df["game_date"] = date
        return df

    boxscores = pd.read_html(url)

    home_df = pipe(
        boxscores,
        partial(get_team_boxscore, points=home_points),
        partial(
            include_schedule_info,
            team=home_team,
            opponent=visitor_team,
            points=home_points,
            date=date,
        ),
    )
    visitor_df = pipe(
        boxscores,
        partial(get_team_boxscore, points=visitor_points),
        partial(
            include_schedule_info,
            team=visitor_team,
            opponent=home_team,
            points=visitor_points,
            date=date,
        ),
    )
    return pd.concat([home_df, visitor_df])


def get_team_boxscore(boxscores: List[pd.DataFrame], points: int) -> pd.DataFrame:
    """
    The basketball reference box score page has tons of different tables
    with basic and advanced stats by game, quarter, and half. We only want
    the basic stats for the entire game. To get the right table, we match
    up the total points scored by the players in a given table with the
    total points for the team in the schedule table. This strategy works
    because one team has to win every game so scores are unique across teams
    and the sum across players will only equal the sum for the team in the
    whole game summary table.
    """

    def basic_clean_boxscore_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function to get boxscore tables clean enough to check if it
        is the one we want
        """

        # Columns come in meaninglessly multilevel and with capitalized names
        df = df.droplevel(level=0, axis=1)
        df.columns = df.columns.str.lower()

        # Get rid of residual divider rows
        df = df.rename({"starters": "player"}, axis=1)
        df = df[~df["player"].isin(["Reserves", "Team Totals"])]

        df = df.replace("Did Not Play", np.nan)
        df = df.replace("Did Not Dress", np.nan)
        df = df.replace("Not With Team", np.nan)
        return df

    for boxscore in boxscores:
        boxscore = basic_clean_boxscore_table(boxscore)
        if "pts" in boxscore.columns:
            if boxscore["pts"].astype(float).sum() == points:
                return boxscore
    raise ValueError("No suitable boxscore found")


def scrape_injury_data() -> pd.DataFrame:
    tmp = pd.read_html("http://www.donbest.com/nba/injuries/")
    tmp = tmp[1].iloc[1:, :5].dropna()
    tmp["is_team_name"] = tmp[0] == tmp[1]
    tmp["is_header"] = tmp[0] == "Date"
    tmp = tmp.query("(is_team_name == False) & (is_header == False)").drop(
        ["is_team_name", "is_header"], axis=1
    )
    tmp.columns = ["date", "pos", "player", "injury", "status"]
    return tmp
