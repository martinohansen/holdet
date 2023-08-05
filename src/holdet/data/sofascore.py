import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import requests

from . import util

PRIMER_LEAGUE = 17
PRIMER_LEAGUE_2022_2023 = 41886
PRIMER_LEAGUE_2023_2024 = 52186


@dataclass
class Season:
    tournament_id: int
    id: int

    @property
    def endpoint(self) -> str:
        return f"/unique-tournament/{self.tournament_id}/season/{self.id}"


@dataclass
class Tournament:
    id: int

    @property
    def endpoint(self) -> str:
        return f"/unique-tournament/{self.id}"

    def season(self, season_id: int) -> Season:
        return Season(self.id, season_id)


@dataclass
class Team:
    id: int
    name: str
    slug: str
    shortName: str


class Side(Enum):
    HOME = 0
    AWAY = 1


@dataclass
class Game:
    id: int
    slug: str
    home: Team
    away: Team
    homeScore: int
    awayScore: int

    round: int
    tournament: Season

    startTimestamp: int

    @property
    def start(self) -> datetime:
        return datetime.fromtimestamp(self.startTimestamp, tz=timezone.utc)

    def __lt__(self, other: "Game"):
        return self.startTimestamp < other.startTimestamp


@dataclass
class Statistics:
    game: Game
    side: Side
    substitute: bool

    assists: int = 0
    expectedAssists: float = 0.0
    expectedGoals: float = 0.0
    goals: int = 0
    goalsPrevented: float = 0.0
    minutesPlayed: int = 0
    onTargetScoringAttempt: int = 0
    savedShotsFromInsideTheBox: int = 0
    saves: int = 0

    @property
    def win(self) -> bool:
        return (self.game.homeScore > self.game.awayScore) == (self.side == Side.HOME)

    @property
    def loss(self) -> bool:
        return (self.game.homeScore < self.game.awayScore) == (self.side == Side.HOME)

    @property
    def draw(self) -> bool:
        return self.game.homeScore == self.game.awayScore

    @property
    def team_goals(self) -> int:
        return self.game.homeScore if self.side == Side.HOME else self.game.awayScore

    @property
    def team_goals_conceded(self) -> int:
        return self.game.awayScore if self.side == Side.HOME else self.game.homeScore

    @property
    def clean_sheet(self) -> bool:
        return self.team_goals_conceded == 0

    # These methods are not perfect, but they are a good enough approximation
    # and they help favor players scoring goals versus defending players.
    @property
    def decisive_goal_for_draw(self) -> bool:
        if self.goals == 0:
            return False
        if self.draw:
            return (self.team_goals - self.goals) + 1 == self.team_goals_conceded
        return False

    @property
    def decisive_goal_for_win(self) -> bool:
        if self.goals == 0:
            return False
        if self.win:
            return (self.team_goals - self.goals) == self.team_goals_conceded
        return False

    # TODO: We need to read yellow and red cards from the incident call:
    # https://api.sofascore.com/api/v1/event/11227333/incidents

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __lt__(self, other):
        return self.game < other.game


@dataclass
class Player:
    id: int
    name: str
    slug: str
    shortName: str

    team: Team

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Player):
            raise NotImplementedError
        return self.id == __value.id


@dataclass
class Lineup:
    home: list[tuple[Player, Statistics]]
    away: list[tuple[Player, Statistics]]

    @property
    def all(self) -> list[tuple[Player, Statistics]]:
        return self.home + self.away


class Client:
    def __init__(
        self,
        base_url: str = "https://api.sofascore.com/api/v1",
    ) -> None:
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            # User-Agent needs to be faked to get a response
            "User-Agent": "Mozilla/5.0 (Macintosh)",
        }

        # Setup request cache and limiter to avoid hitting the API too often and
        # getting blocked
        self.http = util.CachedLimiterSession.new(".sofascore_cache")

    def _get(self, endpoint, params=None):
        url = self.base_url + endpoint
        response = self.http.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def teams(self, tournament: Season) -> list[Team]:
        response = self._get(f"{tournament.endpoint}/statistics/info")
        return [
            Team(
                id=team["id"],
                name=team["name"],
                slug=team["slug"],
                shortName=team["shortName"],
            )
            for team in response["teams"]
        ]

    def games(self, season: Season, round: int) -> list[Game]:
        response = self._get(f"{season.endpoint}/events/round/{round}")
        return [
            Game(
                id=game["id"],
                slug=game["slug"],
                tournament=season,
                round=game["roundInfo"]["round"],
                home=Team(
                    id=game["homeTeam"]["id"],
                    name=game["homeTeam"]["name"],
                    slug=game["homeTeam"]["slug"],
                    shortName=game["homeTeam"]["shortName"],
                ),
                homeScore=game["homeScore"]["current"],
                away=Team(
                    id=game["awayTeam"]["id"],
                    name=game["awayTeam"]["name"],
                    slug=game["awayTeam"]["slug"],
                    shortName=game["awayTeam"]["shortName"],
                ),
                awayScore=game["awayScore"]["current"],
                startTimestamp=game["startTimestamp"],
            )
            for game in response["events"]
            # Only include games that have been played
            if game["status"]["code"] == 100
        ]

    def lineup(self, game: Game) -> Lineup:
        try:
            response = self._get(f"/event/{game.id}/lineups")
        except requests.exceptions.HTTPError:
            logging.warning(f"Could not get lineup for {game}")
            return Lineup(home=[], away=[])
        return Lineup(
            home=[
                (
                    Player(
                        id=player["player"]["id"],
                        name=player["player"]["name"],
                        slug=player["player"]["slug"],
                        shortName=player["player"]["shortName"],
                        team=game.home,
                    ),
                    Statistics(
                        game=game,
                        side=Side.HOME,
                        substitute=player["substitute"],
                        **player["statistics"],
                    ),
                )
                for player in response["home"]["players"]
            ],
            away=[
                (
                    Player(
                        id=player["player"]["id"],
                        name=player["player"]["name"],
                        slug=player["player"]["slug"],
                        shortName=player["player"]["shortName"],
                        team=game.away,
                    ),
                    Statistics(
                        game=game,
                        side=Side.AWAY,
                        substitute=player["substitute"],
                        **player["statistics"],
                    ),
                )
                for player in response["away"]["players"]
            ],
        )

    def statistics(self, game: Game, player: Player) -> Statistics:
        response = self._get(f"/event/{game.id}/player/{player.id}/statistics")
        return Statistics(game=Game, **response["statistics"])

    def current_round(self, tournament: Tournament, season: Season) -> int:
        response = self._get(tournament.endpoint + f"/season/{season.id}/rounds")
        return response["currentRound"]["round"]

    def rounds(self, tournament: Tournament, season: Season) -> int:
        response = self._get(tournament.endpoint + f"/season/{season.id}/info")
        return response["numberOfRounds"]
