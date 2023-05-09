from dataclasses import dataclass

import requests


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


@dataclass
class Player:
    id: int
    name: str
    slug: str
    shortName: str
    position: str


@dataclass
class Lineup:
    home: list[Player]
    away: list[Player]

    @property
    def all(self) -> list[Player]:
        return self.home + self.away


@dataclass
class Statistics:
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


class Client:
    def __init__(self, base_url: str = "https://api.sofascore.com/api/v1") -> None:
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            # User-Agent needs to be faked to get a response
            "User-Agent": "Mozilla/5.0 (Macintosh)",
        }

        self.http = requests.Session()

    def _get(self, endpoint, params=None):
        url = self.base_url + endpoint
        response = self.http.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def teams(self, t: Tournament) -> list[Team]:
        response = self._get(f"{t.endpoint}/statistics/info")
        return [
            Team(
                id=team["id"],
                name=team["name"],
                slug=team["slug"],
                shortName=team["shortName"],
            )
            for team in response["teams"]
        ]

    def games(self, t: Tournament, round: int) -> list[Game]:
        response = self._get(f"{t.endpoint}/events/round/{round}")
        return [
            Game(
                id=game["id"],
                slug=game["slug"],
                home=Team(
                    id=game["homeTeam"]["id"],
                    name=game["homeTeam"]["name"],
                    slug=game["homeTeam"]["slug"],
                    shortName=game["homeTeam"]["shortName"],
                ),
                away=Team(
                    id=game["awayTeam"]["id"],
                    name=game["awayTeam"]["name"],
                    slug=game["awayTeam"]["slug"],
                    shortName=game["awayTeam"]["shortName"],
                ),
            )
            for game in response["events"]
        ]

    def lineup(self, game: Game) -> Lineup:
        response = self._get(f"/event/{game.id}/lineups")
        return Lineup(
            home=[
                Player(
                    id=player["player"]["id"],
                    name=player["player"]["name"],
                    slug=player["player"]["slug"],
                    shortName=player["player"]["shortName"],
                    position=player["player"]["position"],
                )
                for player in response["home"]["players"]
            ],
            away=[
                Player(
                    id=player["player"]["id"],
                    name=player["player"]["name"],
                    slug=player["player"]["slug"],
                    shortName=player["player"]["shortName"],
                    position=player["player"]["position"],
                )
                for player in response["away"]["players"]
            ],
        )

    # TODO: Reduce needed calls by picking uo the stats directly from the lineup
    # call
    def statistics(self, game: Game, player: Player) -> Statistics:
        response = self._get(f"/event/{game.id}/player/{player.id}/statistics")
        return Statistics(**response["statistics"])
