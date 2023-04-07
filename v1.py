#!/usr/bin/env python3

import concurrent.futures
import copy
import math
import sys
import urllib
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Union

import requests
from pulp import (  # type: ignore
    PULP_CBC_CMD,
    LpInteger,
    LpMaximize,
    LpProblem,
    LpVariable,
    value,
)
from tqdm import tqdm  # type: ignore

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

url = "https://data.fotmob.com/stats/47/season/17664/red_card.json"
response = session.get(url)
red_cards = response.json()

url = "https://data.fotmob.com/stats/47/season/17664/yellow_card.json"
response = session.get(url)
yellow_cards = response.json()

url = "https://api.holdet.dk/tournaments/422?appid=holdet"
response = session.get(url)
tournament = response.json()

url = "https://api.holdet.dk/games/644/rounds/14/statistics?appid=holdet"
response = session.get(url)
game = response.json()

url = "https://www.fotmob.com/api/leagues?id=47"
response = session.get(url)
league = response.json()


@dataclass
class Odds:
    Home: float
    Draw: float
    Away: float


@dataclass
class Match:
    Home: str
    Away: str
    Odds: Odds
    Round: int


@dataclass
class Baller:
    first_name: str
    last_name: str
    alt_name: str
    team: str
    value: int
    popularity: float
    trend: int
    position: int
    captain: bool = False
    on_team: bool = False

    fotmob: Union[None, dict] = field(init=False)
    next_match: Union[None, Match] = field(init=False)
    table_all: Union[None, dict] = field(init=False)
    table_home: Union[None, dict] = field(init=False)
    table_away: Union[None, dict] = field(init=False)

    def __post_init__(self):
        self.fotmob = self.__player()
        self.next_match = self.__next_match()

        self.table_all = self.__find_team_in_table(
            league["table"][0]["data"]["table"]["all"]
        )

        self.table_home = self.__find_team_in_table(
            league["table"][0]["data"]["table"]["home"]
        )

        self.table_away = self.__find_team_in_table(
            league["table"][0]["data"]["table"]["away"]
        )

    def __get_api(self, url, timeout: float = 30) -> requests.Response:
        return session.get(url, timeout=timeout)

    def __search_fotmob(self, term: str) -> dict:
        response = self.__get_api(f"https://www.fotmob.com/api/searchData?term={term}")
        return response.json()

    def __fotmob_match(self) -> Union[None, dict]:
        for term in [self.name, self.alt_name, self.last_name, self.first_name]:
            search = self.__search_fotmob(urllib.parse.quote(term))
            if search and search.get("squad"):
                for player in search["squad"]["dataset"]:
                    if self.__i_am(player["teamName"], player["name"]):
                        return player

        return None

    def __player(self) -> Union[None, dict]:
        player = self.__fotmob_match()
        if player:
            fotmob_id = player["id"]
            response = self.__get_api(
                f"https://www.fotmob.com/api/playerData?id={fotmob_id}"
            )
            return response.json()
        print(f"Unable to find fotmob match for {self.name} ({self.team})")
        return None

    def __odds(self, id: int) -> Odds:
        params = "&ccode3=DNK&bettingProvider=Bet365_Denmark"
        url = f"https://www.fotmob.com/api/matchOdds?matchId={id}{params}"
        response = self.__get_api(url)
        match = response.json()

        if match.get("coefficients"):
            return Odds(
                Home=float(match["coefficients"][0][1]),
                Draw=float(match["coefficients"][1][1]),
                Away=float(match["coefficients"][2][1]),
            )
        else:
            print(
                "Match have no odds yet, defaulting to 50% win chance:"
                f" {id=!r} ({self.name!r})"
            )
            return Odds(2, 2, 2)

    def __next_match(self) -> Union[None, Match]:
        """
        Find the most likely match the team is playing in the round by using
        the lowest levenshtein distance between the teams in the matches and the
        ballers team
        """
        matches = league["matches"]
        next_round: int = 30
        start_index: int = matches["firstUnplayedMatch"]["firstUnplayedMatchIndex"]

        closest_match = None
        max_similarity = 0.5
        for match in matches["allMatches"][start_index:]:
            match_round: int = match["round"]
            if match_round == next_round:
                home: str = match["home"]["name"]
                away: str = match["away"]["name"]
                for team in [home, away]:
                    similarity = self.__similarity(self.team, team)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        odds = self.__odds(int(match["id"]))
                        closest_match = Match(
                            home,
                            away,
                            Odds=odds,
                            Round=match_round,
                        )

        if closest_match is None:
            print(f"Unable to find next match for {self.name} ({self.team})")

        return closest_match

    def __find_team_in_table(self, teams: list) -> Union[None, dict]:
        closest_team = None
        max_similarity = 0.5
        for team in teams:
            similarity = self.__similarity(self.team, team["name"])
            if similarity > max_similarity:
                max_similarity = similarity
                closest_team = team

        if closest_team is None:
            print(f"Unable to find team table for {self.name} ({self.team})")

        return closest_team

    @property
    def name(self) -> str:
        return self.first_name + " " + self.last_name

    @property
    def stats(self) -> str:
        return (
            f"Stats(xG={self.xG!r}, xA={self.xA!r}, xCS={self.xCS!r},"
            f" xWin={self.xWin!r}, xDraw={self.xDraw!r}, xLoss={self.xLoss!r},"
            f" xIn={self.xIn!r}, xOut={self.xOut!r}, xYellow={self.xYellow!r},"
            f" xRed={self.xRed!r}, xHattrick={self.xHattrick!r})"
        )

    @property
    def transfer_fee(self) -> float:
        """Transfer fee is 1% of the value"""
        if self.on_team:
            return 0.0
        else:
            return self.value * 0.01

    @property
    def price(self) -> float:
        return self.value + self.transfer_fee

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
            f" xGrowth={self.xGrowth / 1000000:.3f}M,"
            f" xGrowthRound={self.xGrowthRound / 1000:.0f}K,"
            f" xWinProbability={self.xWinProbability * 100:.0f}%,"
            f" captain={self.captain!r}, on_team={self.on_team!r} {fotmob=!r})"
        )

    def __populate_stat(self, stats) -> float:
        if self.fotmob:
            for stat in stats:
                if stat["ParticiantId"] == self.fotmob["id"]:
                    return stat["StatValue"]
        return 0

    def __populate_stat_team(self, stats) -> float:
        closest_team = None
        max_similarity = 0.5
        for stat in stats:
            similarity = self.__similarity(self.team, stat["ParticipantName"])
            if similarity > max_similarity:
                max_similarity = similarity
                closest_team = stat["StatValue"]
        if closest_team is None:
            print(f"Unable to find team in stat for {self.name} ({self.team})")
            return 0
        else:
            return closest_team

    def __similarity(self, a: str, b: str) -> float:
        """Returns the similarity between a and b in percent 0-1"""
        return SequenceMatcher(None, a, b).ratio()

    def __is_similar(self, a: str, b: str, threshold: float = 0.8) -> bool:
        """Returns a bool if a and b is within the similarity threshold"""
        similarity = self.__similarity(a, b)
        if similarity >= threshold:
            return True
        else:
            return False

    def __i_am(self, team: str, name: str) -> bool:
        """Tries to determine if self is the same as team and name"""
        for my_name in [self.name, self.alt_name]:
            if self.__is_similar(self.team, team, threshold=0.5):
                if self.__is_similar(my_name, name, threshold=0.6):
                    return True
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
        if self.table_home:
            return int(self.table_home["wins"])
        else:
            return 0

    @property
    def wins_away(self) -> int:
        if self.table_away:
            return int(self.table_away["wins"])
        else:
            return 0

    @property
    def wins(self) -> int:
        return self.wins_home + self.wins_away

    @property
    def draws_home(self) -> int:
        if self.table_home:
            return int(self.table_home["draws"])
        else:
            return 0

    @property
    def draws_away(self) -> int:
        if self.table_away:
            return int(self.table_away["draws"])
        else:
            return 0

    @property
    def draws(self) -> int:
        return self.draws_home + self.draws_away

    @property
    def losses_home(self) -> int:
        if self.table_home:
            return int(self.table_home["losses"])
        else:
            return 0

    @property
    def losses_away(self) -> int:
        if self.table_away:
            return int(self.table_away["losses"])
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
        if self.table_all:
            scores: str = self.table_all["scoresStr"].split("-")
            return int(scores[0])
        else:
            return 0

    @property
    def goals_conceded(self) -> int:
        if self.table_all:
            scores: str = self.table_all["scoresStr"].split("-")
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
    def yellow(self) -> float:
        return self.__populate_stat(yellow_cards["TopLists"][0]["StatList"])

    @property
    def red(self) -> float:
        return self.__populate_stat(red_cards["TopLists"][0]["StatList"])

    @property
    def x_decisive_goals_win(self) -> float:
        # TODO: Implement math for decisive goals
        return 0

    @property
    def x_decisive_goals_draw(self) -> float:
        # TODO: Implement math for decisive goals
        return 0

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

        # Decisive goals
        growth += 30000 * self.x_decisive_goals_win
        growth += 15000 * self.x_decisive_goals_draw

        # Goals and assists
        growth += goal_points * self.xG
        growth += 60000 * self.xA

        # Fair play
        growth += -20000 * self.xYellow
        growth += -50000 * self.xRed

        # Team performance
        growth += 25000 * self.xWin
        growth += 5000 * self.xDraw
        growth += -15000 * self.xLoss
        growth += 10000 * self.xTeamGoals
        growth += -8000 * self.xTeamConceded
        growth += 10000 * self.xWinAway
        growth += -1000 * self.xLossHome

        # Special
        growth += clean_sheet_points * self.xCS
        growth += 7000 * self.xIn
        growth += -5000 * self.xOut
        growth += 100000 * self.xHattrick

        # Finance
        if self.captain:
            growth = growth * 2

        return growth

    @property
    def xGrowthRound(self) -> float:
        weight: float = 0.5
        growth = 0.0

        if self.total_games != 0 and self.next_match is not None:
            per_round = self.xGrowth / self.total_games
            next_round = per_round * self.xWinProbability
            round_weight = weight
            stats_weight = 1 - round_weight

            growth = (per_round * stats_weight) + (next_round * round_weight)
        return growth

    @property
    def xValueNextRound(self) -> float:
        return self.value + self.xGrowthRound

    def __get_match_side(self, match: Match) -> Union[None, str]:
        home_score = self.__similarity(self.team, match.Home)
        away_score = self.__similarity(self.team, match.Away)

        if home_score > away_score:
            return "Home"
        elif away_score > home_score:
            return "Away"
        return None

    @property
    def xWinProbability(self) -> float:
        if self.next_match:
            match_side = self.__get_match_side(self.next_match)
            if match_side == "Home":
                return 1 / self.next_match.Odds.Home
            elif match_side == "Away":
                return 1 / self.next_match.Odds.Away
        print(f"No match found for player, using win probability of 50% ({self.name})")
        return 0.5

    def __poisson_probability(self, lambda_, x):
        return (math.exp(-lambda_) * lambda_**x) / math.factorial(x)

    @property
    def xHattrick(self) -> float:
        if self.games == 0:
            return 0

        average_xG_per_game = self.xG / self.games

        # More then 9 goals in a match is very unlikely, so we just sum the
        # likelihood from 3 to 9 goals.
        probability = 0
        for x in range(3, 9):
            probability += self.__poisson_probability(average_xG_per_game, x)

        return probability

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

    @property
    def xYellow(self) -> float:
        if self.yellow != 0:
            return self.participation_rate / self.yellow
        else:
            return 0

    @property
    def xRed(self) -> float:
        if self.red != 0:
            return self.participation_rate / self.red
        else:
            return 0


def find_optimal_team(ballers: list[Baller], budget) -> list[Baller]:
    # Add the captain variant of each baller to the list of ballers
    captain_variant = copy.deepcopy(ballers)
    for baller in captain_variant:
        baller.captain = True
    ballers = ballers + captain_variant

    # Create a linear programming problem
    problem = LpProblem("OptimalTeam", LpMaximize)

    # Create a dictionary to store the variables for each baller
    variables = {}

    # Create a variable for each baller, with a lower bound of 0 and an upper
    # bound of 1
    for baller in ballers:
        identifier = f"{baller.name}_{str(baller.captain)}"
        variables[baller] = LpVariable(identifier, 0, 1, LpInteger)

    # Add the constraint that no two ballers with the same name can be selected
    for baller in ballers:
        problem += sum(variables[b] for b in ballers if b.name == baller.name) <= 1

    # Add the constraint that only 4 ballers from the same team is allowed
    for team in set(baller.team for baller in ballers):
        problem += sum(variables[b] for b in ballers if b.team == team) <= 4

    # Set the objective function to maximize the value after round
    problem += sum(variables[b] * b.xValueNextRound for b in ballers)

    # Add the constraint that the price must be less than or equal to the budget
    problem += sum(variables[b] * b.price for b in ballers) <= budget

    # Add the constraint that there must be exactly 11 players in total
    problem += sum(variables[b] for b in ballers) == 11

    # Add the constraint that there must be exactly 1 captain on the team
    problem += sum(variables[b] for b in ballers if b.captain) == 1

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
    problem.solve(PULP_CBC_CMD(msg=0))

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
        first_name=person["firstname"],
        last_name=person["lastname"],
        alt_name=person["slug"].replace("_", " "),
        value=character["values"]["value"],
        popularity=character["values"]["popularity"],
        trend=character["values"]["trend"],
        position=player["position"]["id"],
        team=team["name"],
    )


def levenshtein_distance(s1, s2):
    # Implement the Levenshtein distance algorithm here
    # See https://en.wikipedia.org/wiki/Levenshtein_distance for more information
    m = len(s1)
    n = len(s2)
    d = [[0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(1, m + 1):
        d[i][0] = i
    for j in range(1, n + 1):
        d[0][j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[m][n]


def get_closest_match(name, choices):
    # Find the player name with the smallest Levenshtein distance
    closest_match = None
    min_distance = float("inf")
    for choice in choices:
        distance = levenshtein_distance(name, choice)
        if distance < min_distance:
            closest_match = choice
            min_distance = distance
    return closest_match


def print_solution(solution: list[Baller]):
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

    print()
    for position, players in team_by_position.items():
        print(f"# {position.title()} ({len(players)})")
        for player in players:
            print(player)
        print()
    print(
        f"Combined value: {sum(p.value for p in solution) / 1000000:.2f}M, expected"
        f" after round: {sum(p.xValueNextRound for p in solution) /1000000:.2f}M,"
        f" transfer fee: {sum(p.transfer_fee for p in solution) / 1000:.0f}K, expected"
        f" growth: {sum(p.xGrowth for p in solution) / 1000000:.2f}M total /"
        f" {sum(p.xGrowthRound for p in solution) / 1000:.0f}K next round, average"
        f" popularity: {(sum(p.popularity for p in solution) / 11) * 100:.2f}%, players"
        f" considered: {len(ballers)}, fotmob matches:"
        f" {sum(1 for p in ballers if isinstance(p.fotmob, dict))}\n"
    )


def find_player(ballers: list[Baller], find: str) -> Union[None, Baller]:
    """Search for player by name"""
    player_names = [p.name for p in ballers]
    closest_match = get_closest_match(find, player_names)
    if closest_match:
        for p in ballers:
            if p.name == closest_match:
                return p
    return None


def list_from_file(file: str) -> list[str]:
    """Read a list of strings from a file"""
    with open(file, "r") as f:
        return [line.strip() for line in f.readlines()]


def find_and_print_solution(ballers: list[Baller], budget: int) -> None:
    print_solution(find_optimal_team(ballers, budget))


if __name__ == "__main__":
    ballers: list[Baller] = []

    # Find all players in the tournament who have a character in the game and a
    # popularity of 0.5% or more, this is done to remove players that dont even
    # play in the league anymore.
    tournament_players = [
        player
        for player in tournament["players"]
        if any(
            character["player"]["id"] == player["id"]
            and character["values"]["popularity"] >= 0.005
            for character in game
        )
    ]

    # Use a ThreadPoolExecutor to run the init_baller function concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the init_baller function for each player to the executor
        results = [
            executor.submit(init_baller, player, game, tournament)
            for player in tournament_players
        ]

        # Iterate over the results and append the ballers to the ballers list
        for result in tqdm(
            concurrent.futures.as_completed(results),
            total=len(results),
            bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} players",
        ):
            baller = result.result()
            ballers.append(baller)

    budget = 50000000
    # TODO: Use argparse instead
    if len(sys.argv) == 2:
        budget = int(sys.argv[1])

    # If a team file is provided, add the players to the team
    if len(sys.argv) == 3:
        budget = int(sys.argv[1])
        file = sys.argv[2]
        team = list_from_file(file)
        for player in team:
            player_found = find_player(ballers, player)
            if player_found:
                idx = ballers.index(player_found)
                ballers[idx].on_team = True
            else:
                print(f"No player found with name {player!r} from team file...")

    find_and_print_solution(ballers, budget)

    while True:
        input_value = input(
            "Valid options are:\n"
            " 'b' to change budget value\n"
            " 'r' to remove a player\n"
            " 'w' to write team to file\n"
            " 'q' to quit\n"
            "  Any player name to inspect\n\n"
            "> "
        )
        if input_value == "q":
            break
        elif input_value == "b":
            new_budget = input("Enter new budget value: ")
            budget = int(new_budget)
            find_and_print_solution(ballers, budget)
        elif input_value == "r":
            remove = input("Enter name of player to remove: ")
            player_found = find_player(ballers, remove)
            if player_found:
                print(f"Removed player {player_found.name!r}")
                ballers.remove(player_found)
                find_and_print_solution(ballers, budget)
            else:
                print(f"No player found with name {remove!r}")
        elif input_value == "w":
            file = input("Enter filename to write to: ")
            with open(file, "w") as f:
                for p in find_optimal_team(ballers, budget):
                    f.write(f"{p.name}\n")
        else:
            # Search for player by name
            player_found = find_player(ballers, input_value)
            if player_found is not None:
                print(player_found)
                print(player_found.next_match)
                print(player_found.stats)
                print()
            else:
                print(f"No player found with name {input_value!r}")
