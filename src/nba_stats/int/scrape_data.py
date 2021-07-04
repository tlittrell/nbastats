import datetime
import logging
import time
import unicodedata
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from kedro.pipeline import Pipeline, node
from toolz.functoolz import pipe

log = logging.getLogger("rich")


def scrape_all_schedule_tables(parameters: dict) -> pd.DataFrame:
    schedule_tables = parameters["schedule_tables_to_scrape"]
    schedules = []
    for season, month in schedule_tables:
        log.info(f"scraping schedule for {season} {month}")
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
    result = schedule.apply(
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

    log.info(f"Scraping boxscore for {home_team} v. {visitor_team} on {date}")

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


def extract_team_from_url(url: str) -> str:
    assert url is not None, f"invalid url: {url}"
    assert url != "", f"invalid url: {url}"
    return url[7:10]


def clean_schedule_table(df: pd.DataFrame) -> pd.DataFrame:
    # remove games that haven't happened yet
    df = df[~(df["home_points"] == "")]

    df["visitor_team"] = df["visitor_team_url"].apply(extract_team_from_url)
    df["home_team"] = df["home_team_url"].apply(extract_team_from_url)
    df["game_date"] = df["boxscore_url"].apply(lambda x: x[11:19])
    df["is_overtime"] = df["overtime"].apply(lambda x: 1 if "OT" in x else 0)
    df["attendance"] = df["attendance"].str.replace(",", "")
    df = df.drop(["overtime", "notes"], axis=1, errors="ignore")
    df = df.astype(
        {
            "visitor_points": np.uint8,
            "home_points": np.uint8,
            "game_date": np.datetime64,
            "is_overtime": np.int8,
            "attendance": np.uint16,
        }
    )
    return df


def clean_raw_boxscores(df: pd.DataFrame) -> pd.DataFrame:
    df["mp"] = convert_minutes_played_to_decimal(df)

    df = (
        df.rename({"+/-": "plus_minus"}, axis=1)
        .astype(
            {
                "player": "category",
                "fg": float,
                "fga": float,
                "fg%": float,
                "3p": float,
                "3pa": float,
                "3p%": float,
                "ft": float,
                "fta": float,
                "ft%": float,
                "orb": float,
                "drb": float,
                "trb": float,
                "ast": float,
                "stl": float,
                "blk": float,
                "tov": float,
                "pf": float,
                "pts": float,
                "team": "category",
                "opponent": "category",
                "plus_minus": float,
            }
        )
        .rename({"3p": "three_p", "3pa": "three_pa", "3p%": "three_%"}, axis=1)
        .drop(["fg%", "three_%", "ft%"], axis=1)
    )

    cols = [
        "fg",
        "fga",
        "three_p",
        "three_pa",
        "ft",
        "fta",
        "orb",
        "drb",
        "trb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "pts",
        "plus_minus",
    ]
    df[cols] = df[cols].fillna(0)

    df["is_double_double"] = df.apply(
        lambda row: 1 if get_num_doubles(row) >= 2 else 0, axis=1
    )
    df["is_triple_double"] = df.apply(
        lambda row: 1 if get_num_doubles(row) >= 3 else 0, axis=1
    )
    df["dk_classic_pts"] = calc_draft_kings_classic_points(df)

    df["player"] = (
        df["player"]
        .apply(remove_accents)
        .apply(remove_name_suffixes)
        .apply(clean_abbreviated_names)
        .apply(map_bb_ref_to_dk_names)
    )

    # check output
    assert (df.eval("is_double_double - is_triple_double") >= 0).all(), (
        "Some rows are marked as triple doubles but aren't also marked as double "
        "doubles"
    )
    assert not df.isna().any().any()

    return df


def convert_minutes_played_to_decimal(df: pd.DataFrame) -> pd.Series:
    df["mp"] = df["mp"].replace(np.nan, "00:00")
    start_time = datetime.datetime.strptime("00:00:00.00000", "%H:%M:%S.%f")
    minutes = (
        df["mp"].apply(lambda x: datetime.datetime.strptime(x, "%M:%S")) - start_time
    )
    return minutes.apply(lambda x: x / np.timedelta64(1, "m"))


def get_num_doubles(row) -> int:
    """Helper function for detecting double-doubles and triple-doubles"""

    def is_double(col):
        return 1 if row[col] >= 10 else 0

    pts = is_double("pts")
    trb = is_double("trb")
    ast = is_double("ast")
    blk = is_double("blk")
    stl = is_double("stl")
    return pts + trb + ast + blk + stl


def calc_draft_kings_classic_points(df: pd.DataFrame) -> pd.Series:
    """Calculate DraftKings fantasy points based on
    https://www.draftkings.com/help/rules/nba"""
    base_points = df.eval("pts + 0.5*three_p + 1.25*trb + 1.5*ast + 2*stl - 0.5*tov")
    extra_pts = (
        3 * df["is_triple_double"]
        + 1.5 * (1 - df["is_triple_double"]) * df["is_double_double"]
    )
    result = base_points + extra_pts
    result = result.fillna(0)
    assert (result >= -10).all()
    assert not result.isna().any()
    return result


def create_pandas_profile(df: pd.DataFrame, title: str, out_file: str):
    profile = df.profile_report(title=title)
    profile.to_file(output_file=out_file)


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def remove_name_suffixes(name: str) -> str:
    return (
        name.replace(" Jr.", "")
        .replace(" Sr.", "")
        .replace(" IV", "")
        .replace(" III", "")
        .replace(" II", "")
        .replace(" I", "")
    )


def clean_abbreviated_names(name: str) -> str:
    return name.replace("TJ ", "T.J. ").replace("JJ ", "J.J. ")


def map_bb_ref_to_dk_names(name: str) -> str:
    bb_ref_to_dk_map = {
        "Jakob Poltl": "Jakob Poeltl",
        "Mo Bamba": "Mohamed Bamba",
        "JaKarr Sampson": "Jakarr Sampson",
        "Sviatoslav Mykhailiuk": "Svi Mykhailiuk",
        "Taurean Waller-Prince": "Taurean Prince",
        "Wesleywundu": "Weswundu",
    }
    if name in bb_ref_to_dk_map.keys():
        return bb_ref_to_dk_map[name]
    else:
        return name


def clean_injury_data(df: pd.DataFrame) -> pd.DataFrame:
    df["player"] = (
        df["player"]
        .apply(remove_accents)
        .apply(remove_name_suffixes)
        .apply(clean_abbreviated_names)
        .apply(map_bb_ref_to_dk_names)
    )
    return df


def scrape_data_pipeline() -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=scrape_all_schedule_tables,
                inputs="parameters",
                outputs="raw_schedule_table",
                name="scrape_all_schedule_tables",
                tags=["schedule_creation"],
            ),
            node(
                func=clean_schedule_table,
                inputs="raw_schedule_table",
                outputs="intermediate_schedule_table",
                name="clean_schedule_table",
                tags=["schedule_creation"],
            ),
            node(
                func=scrape_all_boxscores,
                inputs="intermediate_schedule_table",
                outputs="raw_boxscores",
                name="scrape_all_boxscores",
                tags=["boxscore_creation"],
            ),
            node(
                func=clean_raw_boxscores,
                inputs="raw_boxscores",
                outputs="intermediate_boxscores",
                name="clean_boxscores",
                tags=["boxscore_creation"],
            ),
            # node(
            #     func=scrape_injury_data,
            #     inputs=None,
            #     outputs="raw_injury_data",
            #     name="scrape_injury_data",
            #     tags=["injury_data_creation"],
            # ),
            #                 node(
            #     func=clean_injury_data,
            #     inputs="raw_injury_data",
            #     outputs="intermediate_injury_data",
            #     name="clean_injury_data",
            #     tags=["injury_data_creation"],
            # ),
        ],
        tags=["raw"],
    )
