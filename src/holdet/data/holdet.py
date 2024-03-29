import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import requests

from . import util

PRIMER_LEAGUE_2022_2023 = 422
PRIMER_LEAGUE_FALL_2022 = 629
PRIMER_LEAGUE_SPRING_2023 = 644

PRIMER_LEAGUE_2023_2024 = 446
PRIMER_LEAGUE_FALL_2023 = 666


@dataclass
class Game:
    id: int
    name: str
    start: datetime
    end: datetime


@dataclass
class Team:
    id: int
    name: str
    slug: str


@dataclass(frozen=True)  # Makes the class hashable
class Person:
    id: int
    first_name: str
    last_name: str
    slug: str

    @property
    def name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class Position(Enum):
    KEEPER = 6
    DEFENSE = 7
    MIDFIELDER = 8
    FORWARD = 9


class Side(Enum):
    HOME = 0
    AWAY = 1


@dataclass
class Player:
    id: int
    person: Person
    team: Team
    position: Position


@dataclass
class Round:
    game: Game
    number: int
    start: datetime
    end: datetime


@dataclass
class Tournament:
    id: int
    teams: list[Team]
    players: list[Player]
    persons: list[Person]


@dataclass
class Values:
    id: int
    value: int
    growth: int
    totalGrowth: int
    popularity: float
    trend: int


@dataclass
class Statistics:
    round: Round
    player: Player
    values: Values


@dataclass
class Schedule:
    start: datetime
    status: str
    away: Team
    home: Team

    def contains(self, team: Team) -> bool:
        return team == self.away or team == self.home

    def opponent(self, team: Team) -> Team:
        """Return team opponent in the schedule game"""
        if team == self.away:
            return self.home
        elif team == self.home:
            return self.away
        else:
            raise ValueError("Team not in schedule")

    def side(self, team: Team) -> Side:
        """Return the side of team in the schedule game"""
        if team == self.away:
            return Side.AWAY
        elif team == self.home:
            return Side.HOME
        else:
            raise ValueError("Team not in schedule")


@dataclass
class Points:
    position: Position

    @property
    def goal(self) -> int:
        if self.position.KEEPER:
            return 250000
        elif self.position.DEFENSE:
            return 175000
        elif self.position.MIDFIELDER:
            return 150000
        elif self.position.FORWARD:
            return 125000
        else:
            return 0

    @property
    def clean_sheet(self) -> int:
        if self.position.KEEPER:
            return 75000
        elif self.position.DEFENSE:
            return 50000
        else:
            return 0

    own_goal = -75000
    assist = 60000
    shot_on_goal = 10000
    scoring_victory = 30000
    scoring_draw = 15000
    yellow_card = -20000
    second_yellow_card = -20000
    direct_red_card = -50000
    team_win = 25000
    team_draw = 5000
    team_loss = -15000
    team_score = 10000
    opponent_score = -8000
    away_win = 10000
    home_loss = -10000
    on_field = 7000
    off_field = -5000
    goalkeeper_save = 5000
    penalty_save = 100000
    penalty_miss = -30000
    hattrick = 100000
    captain_bonus_multiplier = 2
    bank_interest_multiplier = 1.01


class Client:
    def __init__(self, base_url: str = "https://api.holdet.dk") -> None:
        self.base_url = base_url
        self.http = util.CachedLimiterSession.new(".holdet_cache")

        # Makes a http session without caching
        self.http_no_cache = requests.session()

        self._params: str = "appid=holdet"
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh)",
        }

    def _get(self, endpoint, params=None, no_cache=False):
        url = self.base_url + endpoint
        params = params + self._params if params else self._params

        if no_cache:
            response = self.http_no_cache.get(url, headers=self.headers, params=params)
        else:
            response = self.http.get(url, headers=self.headers, params=params)

        response.raise_for_status()
        return response.json()

    def rounds(self, game: Game) -> list[Round]:
        g = self._get(f"/games/{game.id}")

        rounds: list[Round] = []
        for index, round in enumerate(g["rounds"]):
            rounds.append(
                Round(
                    game=game,
                    number=index + 1,
                    # At holdet the start is the time which transfer opens while
                    # close is the time the round actually begins.
                    start=datetime.fromisoformat(round["close"]),
                    end=datetime.fromisoformat(round["end"]),
                )
            )
        return rounds

    def statistics(
        self,
        tournament: Tournament,
        game: Game,
        round: Round,
    ) -> list[Statistics]:
        stats = self._get(f"/games/{game.id}/rounds/{round.number}/statistics")

        statistics: list[Statistics] = []
        for stat in stats:
            try:
                player = next(
                    p for p in tournament.players if p.id == stat["player"]["id"]
                )
            except StopIteration:
                # Not all players in the tournament are in the game
                logging.debug(f"Player {stat['player']['id']} not found in tournament")
                continue

            values = Values(
                stat["values"]["id"],
                stat["values"]["value"],
                stat["values"]["growth"],
                stat["values"]["totalGrowth"],
                stat["values"]["popularity"],
                stat["values"]["trend"],
            )

            statistics.append(
                Statistics(
                    round=round,
                    player=player,
                    values=values,
                )
            )
        return statistics

    def game(self, id: int) -> Game:
        game = self._get(f"/games/{id}")

        return Game(
            id=game["id"],
            name=game["name"],
            start=datetime.fromisoformat(game["start"]),
            end=datetime.fromisoformat(game["end"]),
        )

    def tournament(self, id: int) -> Tournament:
        tournament = self._get(f"/tournaments/{id}")

        teams: list[Team] = []
        for team in tournament["teams"]:
            teams.append(
                Team(
                    team["id"],
                    team["name"],
                    team["slug"],
                )
            )

        persons: list[Person] = []
        for person in tournament["persons"]:
            persons.append(
                Person(
                    person["id"],
                    person["firstname"],
                    person["lastname"],
                    person["slug"],
                )
            )

        players: list[Player] = []
        for player in tournament["players"]:
            # Find the team in the tournament that the player belongs to
            team = next(t for t in teams if t.id == player["team"]["id"])

            # Find the person for the player in the tournament
            person = next(p for p in persons if p.id == player["person"]["id"])

            players.append(
                Player(
                    player["id"],
                    person,
                    team,
                    Position(player["position"]["id"]),
                )
            )

        return Tournament(
            tournament["id"],
            teams,
            players,
            persons,
        )

    def schedule(self, tournament: Tournament) -> list[Schedule]:
        schedules = self._get(
            f"/tournaments/{tournament.id}/schedule",
            # Using the cached session makes the json decoding fail for this
            # call. It works for every other call but not this, until we find
            # the cause disable cache for this call.
            no_cache=True,
        )
        s: list[Schedule] = []
        for schedule in schedules:
            # Find the teams in the tournament that are playing in this game and
            # add them to the schedule
            for participant in schedule["participants"]:
                match participant["type"]:
                    case "away":
                        away = next(
                            t
                            for t in tournament.teams
                            if t.id == participant["team"]["id"]
                        )
                    case "home":
                        home = next(
                            t
                            for t in tournament.teams
                            if t.id == participant["team"]["id"]
                        )
            s.append(
                Schedule(
                    start=datetime.fromisoformat(schedule["start"]),
                    status=schedule["status"],
                    away=away,
                    home=home,
                )
            )

        return s
