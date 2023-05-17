import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests
from pyrate_limiter import MemoryListBucket
from requests_cache import CacheMixin, FileCache
from requests_ratelimiter import LimiterMixin


@dataclass
class Tournament:
    id: int
    season_id: int

    @property
    def endpoint(self) -> str:
        return f"/unique-tournament/{self.id}/season/{self.season_id}"


@dataclass
class Team:
    id: int
    name: str
    slug: str
    shortName: str


@dataclass
class Game:
    id: int
    slug: str
    home: Team
    away: Team
    homeScore: int
    awayScore: int

    round: int
    tournament: Tournament

    startTimestamp: int

    @property
    def startTimestampHuman(self) -> str:
        return datetime.fromtimestamp(self.startTimestamp).strftime("%Y-%m-%d %H:%M")

    def __lt__(self, other):
        return self.startTimestamp < other.startTimestamp


@dataclass
class Statistics:
    game: Game

    assists: int = 0
    expectedAssists: float = 0.0
    expectedGoals: float = 0.0
    goals: int = 0
    goalsPrevented: float = 0.0
    minutesPlayed: int = 0
    onTargetScoringAttempt: int = 0
    savedShotsFromInsideTheBox: int = 0
    saves: int = 0

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


class CachedLimiterSession(CacheMixin, LimiterMixin, requests.Session):
    pass


class Client:
    def __init__(self, base_url: str = "https://api.sofascore.com/api/v1") -> None:
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            # User-Agent needs to be faked to get a response
            "User-Agent": "Mozilla/5.0 (Macintosh)",
        }

        # Setup request cache and limiter to avoid hitting the API too often and
        # getting blocked
        self.http = CachedLimiterSession(
            per_second=1,
            bucket_class=MemoryListBucket,
            expire_after=timedelta(days=300),
            backend=FileCache(".sofascore_cache"),
        )

    def _get(self, endpoint, params=None):
        url = self.base_url + endpoint
        response = self.http.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def teams(self, tournament: Tournament) -> list[Team]:
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

    def games(self, tournament: Tournament, round: int) -> list[Game]:
        response = self._get(f"{tournament.endpoint}/events/round/{round}")
        return [
            Game(
                id=game["id"],
                slug=game["slug"],
                tournament=tournament,
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
                        **player["statistics"],
                    ),
                )
                for player in response["away"]["players"]
            ],
        )

    def statistics(self, game: Game, player: Player) -> Statistics:
        response = self._get(f"/event/{game.id}/player/{player.id}/statistics")
        return Statistics(game=Game, **response["statistics"])
