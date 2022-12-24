#!/usr/bin/env python3

from dataclasses import dataclass
from difflib import SequenceMatcher

import requests
from pulp import LpInteger, LpMaximize, LpProblem, LpVariable, value  # type: ignore

url = "https://data.fotmob.com/stats/47/season/17664/expected_goals.json"
response = requests.request("GET", url)
expected_goals = response.json()

url = "https://data.fotmob.com/stats/47/season/17664/expected_assists.json"
response = requests.request("GET", url)
expected_assists = response.json()

url = "https://data.fotmob.com/stats/47/season/17664/clean_sheet.json"
response = requests.request("GET", url)
clean_sheet = response.json()

url = "https://api.holdet.dk/tournaments/422?appid=holdet"
response = requests.request("GET", url)
tournament = response.json()

url = "https://api.holdet.dk/games/644/rounds/1/statistics?appid=holdet"
response = requests.request("GET", url)
game = response.json()


@dataclass
class Baller:
    name: str
    value: int
    popularity: float
    trend: int
    position: int

    similarity_threshold: float = 0.8

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

        return (
            f"Baller(name={self.name!r}, position={position!r},"
            f" value={self.value / 1000000:.1f}M,"
            f" popularity={self.popularity * 100:.1f}%,"
            f" xGrowth={self.xGrowth / 1000000:.3f}M"
        )

    def __populate_stat(self, stats) -> float:
        for stat in stats:
            similarity = self.__similarity(stat["ParticipantName"])
            if similarity > self.similarity_threshold:
                return stat["StatValue"]
        return 0

    def __similarity(self, to):
        return SequenceMatcher(None, self.name, to).ratio()

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
    def xGrowth(self) -> float:
        growth: float = 0

        def __goal_bonus(self) -> float:
            if self.keeper:
                return 250000
            elif self.defense:
                return 175000
            elif self.midfielder:
                return 150000
            elif self.forward:
                return 125000
            else:
                return 0

        def __assist_bonus(self) -> float:
            return 60000

        def __clean_sheet_bonus(self) -> float:
            if self.keeper:
                return 75000
            elif self.defense:
                return 50000
            else:
                return 0

        growth += __goal_bonus(self) * self.xG
        growth += __assist_bonus(self) * self.xA
        growth += __clean_sheet_bonus(self) * self.CS

        return growth

    @property
    def xG(self) -> float:
        return self.__populate_stat(expected_goals["TopLists"][0]["StatList"])

    @property
    def xA(self) -> float:
        return self.__populate_stat(expected_assists["TopLists"][0]["StatList"])

    @property
    def CS(self) -> float:
        return self.__populate_stat(clean_sheet["TopLists"][0]["StatList"])


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


if __name__ == "__main__":
    ballers = []
    # Loop over all person in tournament
    for person in tournament["persons"]:
        whoami = person["firstname"] + " " + person["lastname"]

        # Loop over all players in tournament
        for player in tournament["players"]:
            # If player and person is the same
            if player["person"]["id"] == person["id"]:
                # Loop over characters in game
                for character in game:
                    # If character is player
                    if character["player"]["id"] == player["id"]:
                        x = Baller(
                            name=whoami,
                            value=character["values"]["value"],
                            popularity=character["values"]["popularity"],
                            trend=character["values"]["trend"],
                            position=player["position"]["id"],
                        )
                        if x.value != 0:
                            ballers.append(x)

    team: list[Baller] = find_optimal_team(ballers, 50000000)

    team_by_position: dict[str, list[Baller]] = {
        "keepers": [],
        "defenses": [],
        "midfielders": [],
        "forwards": [],
    }
    for player in team:
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
        f"Combined value: {sum(p.value for p in team) / 1000000:.2f}M,"
        f" total expected growth: {sum(p.xGrowth for p in team) / 1000000:.2f}M"
    )
