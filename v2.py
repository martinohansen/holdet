#!/usr/bin/env python3

import concurrent.futures
import logging
import re

import requests
from tqdm import tqdm  # type: ignore


class App:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.logger = logging.Logger("App")

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
            "logger",
            "league_id",
            "season_id",
        ],
    ) -> str:
        attrs = []
        for attr, value in obj.__dict__.items():
            if attr not in exclude:
                attrs.append(f"{attr}={value!r}")
        return f"{obj.__class__.__name__}({', '.join(attrs)})"


class CareerStats(App):
    def __init__(self, stats: list) -> None:
        super().__init__()

        self.leagues: list[LeagueStats] = []
        if stats is not None:
            for stat in stats:
                self.leagues.append(LeagueStats(stat))


class LeagueStats(App):
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


class SeasonStats(App):
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


class Player(App):
    def __init__(self, id: int, team: str) -> None:
        super().__init__()

        resp = self.session.get(f"https://www.fotmob.com/api/playerData?id={id}")
        self.__player = resp.json()

        # Reduce memory usage, we dont have any need for match data and it
        # accounts for around 60% of the dict so lets drop it.
        del self.__player["recentMatches"]

        self.id = id
        self.team = team
        self.name: str = self.__player["name"]

        self.career_stats: CareerStats = CareerStats(self.__player["careerStatistics"])

        self.current_season = None
        for league in self.career_stats.leagues:
            if league.id == self.league_id:
                for season in league.seasons:
                    if season.id == self.season_id:
                        self.current_season = season

        self.character = HOLDET.find_closest_character(self.team, self.name)

    def __repr__(self) -> str:
        id = self.id
        name = self.name
        holdet = self.character
        current_season = self.current_season
        return f"Player({id=!r}, {name=!r}, {holdet=!r}, {current_season=!r}"

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


class Team(App):
    def __init__(self, id: int) -> None:
        super().__init__()

        resp = self.session.get(f"https://www.fotmob.com/api/teams?id={id}")
        team = resp.json()
        squad = team["squad"]

        self.id = id
        self.name: str = team["details"]["name"]

        tables = team["history"]["historicalTableData"]["ranks"]
        for table in tables:
            if table["stageId"] == self.season_id:
                stats = table["stats"]
                self.points: int = stats["points"]
                self.wins: int = stats["wins"]
                self.draws: int = stats["draws"]
                self.loss: int = stats["loss"]

        if self.wins == 0 and self.draws == 0 and self.loss == 0:
            raise self.Error("No matches have been played in the season yet")

        self.players: list[Player] = []
        for position in squad[1:]:  # Index 0 is the coach
            for player in position[1]:  # Index 0 is position name
                self.players.append(Player(player["id"], self.name))

        self.clean_sheets = sum(
            player.current_season.clean_sheets
            for player in self.players
            if player.current_season
        )

    def __repr__(self) -> str:
        return self.repr_string(self, exclude=["players"])


class League(App):
    def __init__(self) -> None:
        super().__init__()

        resp = self.session.get(
            f"https://www.fotmob.com/api/leagues?id={self.league_id}"
        )
        league = resp.json()
        teams = league["table"][0]["data"]["table"]["all"]

        self.id = league["details"]["id"]
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


class Character(App):
    def __init__(
        self,
        name: str,
        value: int,
        popularity: float,
        trend: int,
        position: int,
        team: str,
    ) -> None:
        self.name = name
        self.value = value
        self.popularity = popularity
        self.trend = trend
        self.position = position
        self.team = team


class Holdet(App):
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

        self.__characters: list[Character] = self.__get_characters(tournament, game)

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
                    value=character["values"]["value"],
                    popularity=character["values"]["popularity"],
                    trend=character["values"]["trend"],
                    position=player["position"]["id"],
                    team=team["name"],
                )
            )
        return characters

    def __jaccard_similarity(self, str1: str, str2: str) -> float:
        # Convert the strings to sets of characters
        set1 = set(str1)
        set2 = set(str2)

        # Calculate the size of the intersection of the sets
        intersection = set1 & set2
        intersection_size = len(intersection)

        # Calculate the size of the union of the sets
        union = set1 | set2
        union_size = len(union)

        # Calculate the Jaccard similarity
        similarity = intersection_size / union_size

        return similarity

    def find_closest_character(self, team_name: str, player_name: str):
        # Initialize a variable to store the maximum similarity
        max_similarity = 0.0

        # Initialize a variable to store the closest match
        closest_match_idx = None

        # Iterate over the players in the list
        for character in self.__characters:
            # Calculate the Jaccard similarity between the team name and the
            # character's team name
            team_similarity = self.__jaccard_similarity(team_name, character.team)

            # Calculate the Jaccard similarity between the character name and the
            # character's name
            name_similarity = self.__jaccard_similarity(player_name, character.name)

            # Calculate the overall similarity as the average of the team and
            # name similarities
            overall_similarity = (team_similarity + name_similarity) / 2

            # If the overall similarity is greater than the maximum similarity,
            # update the closest match
            if overall_similarity > max_similarity:
                max_similarity = overall_similarity
                closest_match_idx = self.__characters.index(character)

        return self.__characters.pop(closest_match_idx)  #  type: ignore


HOLDET = Holdet()


p = League()
print(f"Teams: {len(p.teams)}, Players: { sum(len(team.players) for team in p.teams)}")

for team in p.teams[:1]:
    print(team)
    for player in team.players[:2]:
        print(player)
