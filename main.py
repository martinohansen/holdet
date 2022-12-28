#!/usr/bin/env python3

import concurrent.futures
import sys
import urllib
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Union

import requests
from pulp import LpInteger, LpMaximize, LpProblem, LpVariable, value  # type: ignore

session = requests.Session()

url = "https://data.fotmob.com/stats/47/season/17664/expected_goals.json"
response = session.get(url)
expected_goals = response.json()

url = "https://data.fotmob.com/stats/47/season/17664/expected_assists.json"
response = session.get(url)
expected_assists = response.json()

url = "https://data.fotmob.com/stats/47/season/17664/clean_sheet.json"
response = session.get(url)
clean_sheet = response.json()

url = "https://data.fotmob.com/stats/47/season/17664/clean_sheet_team.json"
response = session.get(url)
clean_sheet_team = response.json()

url = "https://api.holdet.dk/tournaments/422?appid=holdet"
response = session.get(url)
tournament = response.json()

url = "https://api.holdet.dk/games/644/rounds/1/statistics?appid=holdet"
response = session.get(url)
game = response.json()

url = "https://www.fotmob.com/api/leagues?id=47"
response = session.get(url)
teams = response.json()


@dataclass
class Baller:
    name: str
    team: str
    value: int
    popularity: float
    trend: int
    position: int

    similarity_threshold: float = 0.8
    api_timeout: float = 5

    fotmob: Union[None, dict] = field(init=False)
    team_all: Union[None, dict] = field(init=False)
    team_home: Union[None, dict] = field(init=False)
    team_away: Union[None, dict] = field(init=False)

    def __post_init__(self):
        player = self.__fotmob_match()
        if player:
            fotmob_id = player["id"]
            response = self.__get_api(
                f"https://www.fotmob.com/api/playerData?id={fotmob_id}"
            )
            self.fotmob = response.json()
        else:
            self.fotmob = None

        self.team_all = None
        for team in teams["table"][0]["data"]["table"]["all"]:
            if self.__is_similar(self.team, team["name"]):
                self.team_all = team

        self.team_home = None
        for team in teams["table"][0]["data"]["table"]["home"]:
            if self.__is_similar(self.team, team["name"]):
                self.team_home = team

        self.team_away = None
        for team in teams["table"][0]["data"]["table"]["away"]:
            if self.__is_similar(self.team, team["name"]):
                self.team_away = team

    def __get_api(self, url) -> requests.Response:
        return session.get(url, timeout=self.api_timeout)

    def __search_fotmob(self, term: str) -> dict:
        response = self.__get_api(f"https://www.fotmob.com/api/searchData?term={term}")
        return response.json()

    def __fotmob_match(self) -> Union[None, dict]:
        search = self.__search_fotmob(urllib.parse.quote(self.name))
        if search.get("squad"):
            for player in search["squad"]["dataset"]:
                if self.__is_similar(self.name, player["name"]):
                    return player
        return None

    def __hash__(self):
        return hash(tuple(self.name))

    def __repr__(self) -> str:
        position = ""
        if self.keeper:
            position = "keeper"
        elif self.defense:
            position = "defense"
        elif self.midfielder:
            position = "midfielder"
        elif self.forward:
            position = "forward"

        if self.fotmob:
            fotmob = self.fotmob["id"]
        else:
            fotmob = False

        return (
            f"Baller(name={self.name!r}, team={self.team!r}, position={position!r},"
            f" value={self.value / 1000000:.1f}M,"
            f" popularity={self.popularity * 100:.1f}%,"
            f" xGrowth={self.xGrowth / 1000000:.3f}M, {fotmob=!r})"
        )

    def __populate_stat(self, stats) -> float:
        for stat in stats:
            if self.__is_similar(self.name, stat["ParticipantName"]):
                return stat["StatValue"]
        return 0

    def __populate_stat_team(self, stats) -> float:
        for stat in stats:
            if self.__is_similar(self.team, stat["ParticipantName"]):
                return stat["StatValue"]
        return 0

    def __similarity(self, a: str, b: str) -> float:
        """Returns the similarity between a and b in percent 0-1"""
        return SequenceMatcher(None, a, b).ratio()

    def __is_similar(self, a: str, b: str) -> bool:
        """Returns a bool if a and b is within the similarity threshold"""
        similarity = self.__similarity(a, b)
        if similarity > self.similarity_threshold:
            return True
        else:
            return False

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

    @property
    def games(self) -> int:
        if self.fotmob:
            if self.fotmob.get("lastLeague"):
                return int(self.fotmob["lastLeague"]["playerProps"][0]["value"])
        return 0

    @property
    def injury(self) -> bool:
        if self.fotmob:
            if self.fotmob["injuryInformation"] is not None:
                return True
        return False

    @property
    def wins_home(self) -> int:
        if self.team_home:
            return int(self.team_home["wins"])
        else:
            return 0

    @property
    def wins_away(self) -> int:
        if self.team_away:
            return int(self.team_away["wins"])
        else:
            return 0

    @property
    def wins(self) -> int:
        return self.wins_home + self.wins_away

    @property
    def draws_home(self) -> int:
        if self.team_home:
            return int(self.team_home["draws"])
        else:
            return 0

    @property
    def draws_away(self) -> int:
        if self.team_away:
            return int(self.team_away["draws"])
        else:
            return 0

    @property
    def draws(self) -> int:
        return self.draws_home + self.draws_away

    @property
    def losses_home(self) -> int:
        if self.team_home:
            return int(self.team_home["losses"])
        else:
            return 0

    @property
    def losses_away(self) -> int:
        if self.team_away:
            return int(self.team_away["losses"])
        else:
            return 0

    @property
    def total_games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def losses(self) -> int:
        return self.losses_home + self.losses_away

    @property
    def goals_scored(self) -> int:
        if self.team_all:
            scores: str = self.team_all["scoresStr"]
            scores.split("-")
            return int(scores[0])
        else:
            return 0

    @property
    def goals_conceded(self) -> int:
        if self.team_all:
            scores: str = self.team_all["scoresStr"]
            scores.split("-")
            return int(scores[1])
        else:
            return 0

    @property
    def participation_rate(self) -> float:
        if self.total_games != 0:
            return self.games / self.total_games
        else:
            return 0

    @property
    def CS(self) -> float:
        return self.__populate_stat_team(clean_sheet_team["TopLists"][0]["StatList"])

    @property
    def xGrowth(self) -> float:
        growth: float = 0

        if self.keeper:
            goal_points = 250000
        elif self.defense:
            goal_points = 175000
        elif self.midfielder:
            goal_points = 150000
        elif self.forward:
            goal_points = 125000
        else:
            goal_points = 0

        if self.keeper:
            clean_sheet_points = 75000
        elif self.defense:
            clean_sheet_points = 50000
        else:
            clean_sheet_points = 0

        growth += goal_points * self.xG
        growth += clean_sheet_points * self.xCS
        growth += 60000 * self.xA

        growth += 25000 * self.xWin
        growth += 5000 * self.xDraw
        growth += -15000 * self.xLoss
        growth += 10000 * self.xTeamGoals
        growth += -8000 * self.xTeamConceded
        growth += 10000 * self.xWinAway
        growth += -1000 * self.xLossHome

        growth += 7000 * self.xIn
        growth += -5000 * self.xOut

        return growth

    @property
    def xIn(self) -> float:
        return self.participation_rate * self.total_games

    @property
    def xOut(self) -> float:
        return self.total_games - self.xIn

    @property
    def xTeamGoals(self) -> float:
        return self.participation_rate * self.goals_scored

    @property
    def xTeamConceded(self) -> float:
        return self.participation_rate * self.goals_conceded

    @property
    def xWin(self) -> float:
        return self.participation_rate * self.wins

    @property
    def xDraw(self) -> float:
        return self.participation_rate * self.draws

    @property
    def xLoss(self) -> float:
        return self.participation_rate * self.losses

    @property
    def xLossHome(self) -> float:
        return self.participation_rate * self.losses_home

    @property
    def xWinAway(self) -> float:
        return self.participation_rate * self.wins_away

    @property
    def xCS(self) -> float:
        return self.participation_rate * self.CS

    @property
    def xG(self) -> float:
        return self.__populate_stat(expected_goals["TopLists"][0]["StatList"])

    @property
    def xA(self) -> float:
        return self.__populate_stat(expected_assists["TopLists"][0]["StatList"])


def find_optimal_team(ballers, value_limit):
    # Create a linear programming problem
    problem = LpProblem("OptimalTeam", LpMaximize)

    # Create a dictionary to store the variables for each baller
    variables = {}

    # Create a variable for each baller, with a lower bound of 0 and an upper bound of 1
    for baller in ballers:
        variables[baller] = LpVariable(baller.name, 0, 1, LpInteger)

    # Set the objective function to maximize the xGrowth
    problem += sum(variables[b] * b.xGrowth for b in ballers)

    # Add the constraint that the value must be less than or equal to the value limit
    problem += sum(variables[b] * b.value for b in ballers) <= value_limit

    # Add the constraint that there must be exactly 11 players in total
    problem += sum(variables[b] for b in ballers) == 11

    # Add the constraint that no player may be injured
    problem += sum(variables[b] for b in ballers if b.injury) == 0

    # Add the constraint that there must be exactly 1 keeper
    problem += sum(variables[b] for b in ballers if b.keeper) == 1

    # Add the constraint that there must be between 3 and 5 defenders
    problem += sum(variables[b] for b in ballers if b.defense) >= 3
    problem += sum(variables[b] for b in ballers if b.defense) <= 5

    # Add the constraint that there must be between 3 and 5 midfielders
    problem += sum(variables[b] for b in ballers if b.midfielder) >= 3
    problem += sum(variables[b] for b in ballers if b.midfielder) <= 5

    # Add the constraint that there must be between 1 and 3 forwards
    problem += sum(variables[b] for b in ballers if b.forward) >= 1
    problem += sum(variables[b] for b in ballers if b.forward) <= 3

    # Solve the problem
    problem.solve()

    # Initialize an empty list to store the optimal team
    optimal_team = []

    # Iterate through the ballers
    for baller in ballers:
        # If the variable for the baller is non-zero, add it to the optimal team
        if value(variables[baller]) > 0:
            optimal_team.append(baller)

    return optimal_team


def init_baller(player: dict, game: dict, tournament: dict) -> Baller:
    # Find the team in the tournament that the player belongs to
    team = next(t for t in tournament["teams"] if t["id"] == player["team"]["id"])

    # Find the character for the player in the game
    character = next(
        character for character in game if character["player"]["id"] == player["id"]
    )

    # Find the person for the player in the tournament
    person = next(
        person
        for person in tournament["persons"]
        if person["id"] == player["person"]["id"]
    )

    return Baller(
        name=person["firstname"] + " " + person["lastname"],
        value=character["values"]["value"],
        popularity=character["values"]["popularity"],
        trend=character["values"]["trend"],
        position=player["position"]["id"],
        team=team["name"],
    )


def debugger(type, value, tb):
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import pdb
        import traceback

        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)


if __name__ == "__main__":
    sys.excepthook = debugger

    ballers: list[Baller] = []

    # Find all players in the tournament who have a character in the game
    tournament_players = [
        player
        for player in tournament["players"]
        if any(character["player"]["id"] == player["id"] for character in game)
    ]

    # Use a ThreadPoolExecutor to run the init_baller function concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the init_baller function for each player to the executor
        results = [
            executor.submit(init_baller, player, game, tournament)
            for player in tournament_players
        ]

        # Iterate over the results and append the ballers to the ballers list
        for result in concurrent.futures.as_completed(results):
            baller = result.result()
            # Only append baller if they have value and at least 2% popularity
            if baller.value != 0 and baller.popularity > 0.02:
                ballers.append(baller)

    solution: list[Baller] = find_optimal_team(ballers, 50000000)

    team_by_position: dict[str, list[Baller]] = {
        "keepers": [],
        "defenses": [],
        "midfielders": [],
        "forwards": [],
    }
    for player in solution:
        if player.keeper:
            team_by_position["keepers"].append(player)
        if player.defense:
            team_by_position["defenses"].append(player)
        if player.midfielder:
            team_by_position["midfielders"].append(player)
        if player.forward:
            team_by_position["forwards"].append(player)

    for position, players in team_by_position.items():
        print(f"# {position.title()} ({len(players)})")
        for player in players:
            print(player)
        print()
    print(
        f"Combined value: {sum(p.value for p in solution) / 1000000:.2f}M, total"
        f" expected growth: {sum(p.xGrowth for p in solution) / 1000000:.2f}M, average"
        f" popularity = {(sum(p.popularity for p in solution) / 11) * 100:.2f}%,"
        f" players considered: {len(ballers)}"
    )
