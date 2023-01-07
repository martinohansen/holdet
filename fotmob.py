#!/usr/bin/env python3

import concurrent.futures
import logging
import re

import requests
from tqdm import tqdm  # type: ignore


class Fotmob:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.logger = logging.Logger("fotmob")

    class Error(Exception):
        pass

    def repr_string(self, obj, exclude=["session", "logger"]) -> str:
        attrs = []
        for attr, value in obj.__dict__.items():
            if attr not in exclude:
                attrs.append(f"{attr}={value!r}")
        return f"{obj.__class__.__name__}({', '.join(attrs)})"


class CareerStats(Fotmob):
    def __init__(self, stats: list) -> None:
        super().__init__()

        self.leagues: list[LeagueStats] = []
        if stats is not None:
            for stat in stats:
                self.leagues.append(LeagueStats(stat))

    def __repr__(self) -> str:
        return super().repr_string(self)


class LeagueStats(Fotmob):
    def __init__(self, stats: dict) -> None:
        super().__init__()

        self.id: int = stats["id"]
        self.name: str = stats["name"]

        self.matches: int = stats["totalMatches"]
        self.sub_in: int = stats["totalSubIn"]
        self.goals: int = stats["totalGoals"]
        self.assists: int = stats["totalAssists"]
        self.yellow_card: int = stats["totalYC"]
        self.red_card: int = stats["totalRC"]
        self.is_cup: bool = stats["isCup"]

        self.seasons: list[SeasonStats] = []
        for stat in stats["seasons"]:
            self.seasons.append(SeasonStats(stat))

    def __repr__(self) -> str:
        return super().repr_string(self)


class SeasonStats(Fotmob):
    def __init__(self, stats: dict) -> None:
        super().__init__()

        self.id = stats["seasonId"]
        self.name = stats["name"]

        self.matches: int = stats["matches"]
        self.sub_in: int = stats["subIn"]
        self.goals: int = stats["goals"]
        self.assists: int = stats["assists"]
        self.yellow_card: int = stats["yc"]
        self.red_card: int = stats["rc"]

        # Goalkeeper
        self.clean_sheets: float = 0
        self.saved_penalties: float = 0

        # Non-goalkeeper
        self.expected_goals: float = 0
        self.penalty_goals: float = 0
        self.shots_on_target: float = 0

        stats_arr = stats["stats"][0]["statsArr"]
        for stat in stats_arr:
            name = stat[0].lower().replace(" ", "_")
            name = re.sub(r"\(.*\)", "", name)
            value = stat[1]
            if (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
            ):
                setattr(self, name, float(value))

    def __repr__(self) -> str:
        return super().repr_string(self)


class Player(Fotmob):
    def __init__(self, id: int) -> None:
        super().__init__()

        resp = self.session.get(f"https://www.fotmob.com/api/playerData?id={id}")
        self.__player = resp.json()

        self.id = id
        self.name: str = self.__player["name"]

        self.stats: CareerStats = CareerStats(self.__player["careerStatistics"])

    def __repr__(self) -> str:
        id = self.id
        name = self.name
        stats = self.stats
        return f"Player({id=!r}, {name=!r}, {stats=!r}"

    @property
    def market_value(self) -> int:
        for prop in self.__player["playerProps"]:
            if prop["title"] == "Market value":
                return prop["value"]
        raise self.Error(f"failed to find market_value for: {self.name}")

    @property
    def injured(self) -> bool:
        if self.__player.get("injuryInformation"):
            return True
        else:
            return False


class Team(Fotmob):
    def __init__(self, id: int) -> None:
        super().__init__()

        resp = self.session.get(f"https://www.fotmob.com/api/teams?id={id}")
        team = resp.json()
        squad = team["squad"]

        self.id = id
        self.name: str = team["details"]["name"]
        self.players: list[Player] = []
        for position in squad[1:]:  # Index 0 is the coach
            for player in position[1]:  # Index 0 is position name
                self.players.append(Player(player["id"]))

    def __repr__(self) -> str:
        id = self.id
        name = self.name
        players = self.players
        return f"Team({id=!r}, {name=!r}, {players=!r}"


class League(Fotmob):
    def __init__(self, id: int = 47) -> None:
        super().__init__()

        resp = self.session.get(f"https://www.fotmob.com/api/leagues?id={id}")
        league = resp.json()
        teams = league["table"][0]["data"]["table"]["all"]

        self.id = id
        self.name = league["details"]["name"]

        self.teams: list[Team] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(Team, team["id"]): team for team in teams}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} teams",
            ):
                self.teams.append(future.result())

    def __repr__(self) -> str:
        id = self.id
        name = self.name
        teams = self.teams
        return f"League({id=!r}, {name=!r}, {teams=!r})"


p = League()
print(f"Teams: {len(p.teams)}, Players: { sum(len(team.players) for team in p.teams)}")

for team in p.teams[:1]:
    for player in team.players[:1]:
        print(player)
