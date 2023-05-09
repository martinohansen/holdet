from dataclasses import dataclass

import requests


@dataclass
class Character:
    name: str
    value: int
    growth: int
    totalGrowth: int
    popularity: float
    trend: int
    position: int
    team: str

    @property
    def keeper(self) -> bool:
        if self.position == 6:
            return True
        return False

    @property
    def defense(self) -> bool:
        if self.position == 7:
            return True
        return False

    @property
    def midfielder(self) -> bool:
        if self.position == 8:
            return True
        return False

    @property
    def forward(self) -> bool:
        if self.position == 9:
            return True
        return False


class Client:
    def __init__(self, base_url: str = "https://api.holdet.dk") -> None:
        self.base_url = base_url
        self.http = requests.Session()

        self._params: str = "?appid=holdet"
        self.headers = {
            "Content-Type": "application/json",
        }

    def _get(self, endpoint, params=None):
        url = self.base_url + endpoint
        params = params + self._params if params else self._params
        response = self.http.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def characters(
        self,
        tournament_id: int,
        game_id: int,
        game_round: int,
    ) -> list[Character]:
        tournament = self._get(f"/tournaments/{tournament_id}")
        game = self._get(f"/games/{game_id}/rounds/{game_round}/statistics")

        tournament_players = [
            player
            for player in tournament["players"]
            if any(character["player"]["id"] == player["id"] for character in game)
        ]

        characters: list[Character] = []
        for player in tournament_players:
            # Find the team in the tournament that the player belongs to
            team = next(
                t for t in tournament["teams"] if t["id"] == player["team"]["id"]
            )

            # Find the character for the player in the game
            character = next(
                character
                for character in game
                if character["player"]["id"] == player["id"]
            )

            # Find the person for the player in the tournament
            person = next(
                person
                for person in tournament["persons"]
                if person["id"] == player["person"]["id"]
            )

            characters.append(
                Character(
                    name=person["firstname"] + " " + person["lastname"],
                    growth=character["values"]["growth"],
                    totalGrowth=character["values"]["totalGrowth"],
                    value=character["values"]["value"],
                    popularity=character["values"]["popularity"],
                    trend=character["values"]["trend"],
                    position=player["position"]["id"],
                    team=team["name"],
                )
            )

        return characters
