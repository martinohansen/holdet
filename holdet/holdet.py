import requests


class App:
    def __init__(self) -> None:
        self.session = requests.Session()

        self.league_id: int = 47
        self.season_id: int = 879290

    class Error(Exception):
        pass

    def __repr__(self) -> str:
        return self.repr_string(self)

    def repr_string(
        self,
        obj,
        exclude: list[str] = [
            "session",
            "league_id",
            "season_id",
        ],
    ) -> str:
        attrs = []
        for attr, value in obj.__dict__.items():
            if attr not in exclude:
                attrs.append(f"{attr}={value!r}")
        return f"{obj.__class__.__name__}({', '.join(attrs)})"


class Character(App):
    def __init__(
        self,
        name: str,
        value: int,
        growth: int,
        totalGrowth: int,
        popularity: float,
        trend: int,
        position: int,
        team: str,
    ) -> None:
        self.name = name
        self.value = value
        self.growth = growth
        self.totalGrowth = totalGrowth
        self.popularity = popularity
        self.trend = trend
        self.position = position
        self.team = team

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


class Game(App):
    def __init__(
        self,
        tournament_id: int = 422,
        game_id: int = 644,
        game_round: int = 3,
    ) -> None:
        super().__init__()

        __params: str = "?appid=holdet"

        url = f"https://api.holdet.dk/tournaments/{tournament_id}{__params}"
        tournament = self.session.get(url).json()

        url = f"https://api.holdet.dk/games/{game_id}/rounds/{game_round}/statistics{__params}"
        game = self.session.get(url).json()

        self.Characters: list[Character] = self.__get_characters(tournament, game)

    def __get_characters(self, tournament, game) -> list[Character]:
        characters: list[Character] = []

        tournament_players = [
            player
            for player in tournament["players"]
            if any(character["player"]["id"] == player["id"] for character in game)
        ]

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
