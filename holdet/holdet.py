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
        return self.position == 6

    @property
    def defense(self) -> bool:
        return self.position == 7

    @property
    def midfielder(self) -> bool:
        return self.position == 8

    @property
    def forward(self) -> bool:
        return self.position == 9


@dataclass
class Points:
    character: Character

    @property
    def goal(self) -> int:
        if self.character.keeper:
            return 250000
        elif self.character.defense:
            return 175000
        elif self.character.midfielder:
            return 150000
        elif self.character.forward:
            return 125000
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
    clean_sheet_defense = 50000
    clean_sheet_goalkeeper = 75000
    goalkeeper_save = 5000
    penalty_save = 100000
    penalty_miss = -30000
    hattrick = 100000
    captain_bonus_multiplier = 2
    bank_interest_multiplier = 1.01


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
