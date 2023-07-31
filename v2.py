#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich import print
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from holdet import holdet
from sofascore import sofascore


@dataclass
class Holdet:
    player: holdet.Player
    stats: list[holdet.Statistics]


@dataclass
class Sofascore:
    player: sofascore.Player
    stats: list[sofascore.Statistics]


@dataclass
class Round:
    number: int
    values: holdet.Values
    stats: list[sofascore.Statistics]
    position: holdet.Position

    @property
    def xGrowth(self) -> float:
        growth: float = 0

        # The points vary depending on the position
        points = holdet.Points(self.position)

        for stat in self.stats:
            # Goals and assists
            growth += points.goal * stat.expectedGoals
            growth += points.assist * stat.expectedAssists
            growth += points.shot_on_goal * stat.onTargetScoringAttempt

            # Decisive goals
            growth += points.scoring_victory * stat.decisive_goal_for_win
            growth += points.scoring_draw * stat.decisive_goal_for_draw

            # Fair play
            # growth += points.yellow_card * stat.yellowCard
            # growth += -50000 * self.xRed

            # # Team performance
            growth += points.team_win * stat.win
            growth += points.team_draw * stat.draw
            growth += points.team_loss * stat.loss
            growth += points.team_score * stat.team_goals
            growth += points.opponent_score * stat.team_goals_conceded
            growth += points.away_win if stat.side == "away" and stat.win else 0
            growth += points.home_loss if stat.side == "home" and stat.loss else 0

            # # Special
            growth += points.clean_sheet if stat.clean_sheet else 0
            growth += points.on_field if stat.minutesPlayed != 0 else 0
            growth += points.off_field if stat.minutesPlayed == 0 else 0
            growth += points.hattrick if stat.expectedGoals >= 3 else 0

        return growth

    @property
    def growth(self) -> int:
        return self.values.growth

    @property
    def diff(self) -> float:
        return self.xGrowth - self.growth

    def __lt__(self, other: "Round") -> bool:
        return self.number < other.number

    def __repr__(self) -> str:
        return (
            f"Round(number={self.number}, games={len(self.stats)},"
            f" growth={self.values.growth / 1000:.0f}K,"
            f" xGrowth={self.xGrowth / 1000:.0f}K,"
            f" diff={self.diff / 1000:+.0f}K)"
        )


@dataclass
class Candidate:
    avatar: Holdet  # Avatar is the player from the game
    person: Sofascore  # Person is the player from real world

    captain: bool = False  # Whether the player is the captain or not
    on_team: bool = False  # Whether the player is on the team or not

    @property
    def id(self) -> int:
        return self.avatar.player.id

    @property
    def name(self) -> str:
        return self.avatar.player.person.name

    @property
    def team(self) -> str:
        return self.avatar.player.team.name

    @property
    def value(self) -> int:
        return self.rounds[-1].values.value

    @property
    def keeper(self) -> bool:
        return self.avatar.player.position == holdet.Position.KEEPER

    @property
    def defense(self) -> bool:
        return self.avatar.player.position == holdet.Position.DEFENSE

    @property
    def midfielder(self) -> bool:
        return self.avatar.player.position == holdet.Position.MIDFIELDER

    @property
    def forward(self) -> bool:
        return self.avatar.player.position == holdet.Position.FORWARD

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

    @property
    def emoji(self) -> str:
        if self.captain:
            return "👑"
        return "⚽️"

    @property
    def similarity(self) -> float:
        return _similarity(self.person.player, self.avatar.player)

    @property
    def rounds(self) -> list[Round]:
        """
        Zip the statistics from the avatar and the person together. Every stat
        that is within the same round is zipped together. Only include rounds
        that have ended.
        """

        def zip(stat: holdet.Statistics) -> Round:
            return Round(
                number=stat.round.number,
                values=stat.values,
                stats=[
                    s
                    for s in self.person.stats
                    if stat.round.start <= s.game.start <= stat.round.end
                ],
                position=stat.player.position,
            )

        return [
            zip(stat)
            for stat in self.avatar.stats
            if stat.round.end < datetime.now(tz=timezone.utc)
        ]

    def xGrowthEMA(self, alpha: float) -> float:
        """
        Predict the growth for the next game using exponential moving average
        (EMA). Set alpha to adjust the smoothing factor, between 0 and 1. Higher
        values give more weight to recent stats.
        """
        ema = 0.0
        for i, round in enumerate(sorted(self.rounds)):
            if i == 0:
                ema = round.xGrowth
            else:
                ema = alpha * round.xGrowth + (1 - alpha) * ema
        return ema

    @property
    def xGrowth(self) -> float:
        if self.captain:
            return self.xGrowthEMA(0.5) * 2
        return self.xGrowthEMA(0.5)

    @property
    def xValue(self) -> float:
        return self.value + self.xGrowth

    def __eq__(self, other: object):
        if not isinstance(other, Candidate):
            return NotImplemented
        return self.id == other.id

    def __lt__(self, other: "Candidate"):
        return self.xGrowth < other.xGrowth

    def __hash__(self):
        return hash(self.avatar.player.person)

    def __repr__(self) -> str:
        return (
            f"{self.emoji} {self.name} ({self.team}),"
            f" value={self.value / 1000000:.1f}M, xGrowth={self.xGrowth / 1000:.0f}K"
            # f" rounds={self.rounds}"
        )


class Formation:
    def __init__(self, solution: list[Candidate]):
        self.position: dict[str, list[Candidate]] = {
            "keeper": [],
            "defenses": [],
            "midfielders": [],
            "forwards": [],
        }
        self._populate(solution)

    def _populate(self, solution: list[Candidate]):
        for player in solution:
            if player.keeper:
                self.position["keeper"].append(player)
            if player.defense:
                self.position["defenses"].append(player)
            if player.midfielder:
                self.position["midfielders"].append(player)
            if player.forward:
                self.position["forwards"].append(player)

    def __iter__(self):
        return iter(self.position.items())

    def __repr__(self) -> str:
        return f"{len(self.position['defenses'])}-{len(self.position['midfielders'])}-{len(self.position['forwards'])}"

    def __rich__(self) -> Table:
        table = Table(title=f"XI ({self})")
        table.add_column("Position")
        table.add_column("Players")
        for index, (position, players) in enumerate(self):
            for player in players:
                table.add_row(position.capitalize(), str(player))
                position = ""  # Avoid repeating the position name in the same row

            # Add an empty row between positions
            if index < 3:
                table.add_row()

        return table


def _jaccard_similarity(str1: str, str2: str) -> float:
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


def _similarity(s: sofascore.Player, h: holdet.Player) -> float:
    team_similarity = _jaccard_similarity(s.team.name, h.team.name)
    name_similarity = _jaccard_similarity(s.name, h.person.name)

    # Calculate the overall similarity as the average of the team and
    # name similarities
    overall_similarity = (team_similarity + name_similarity) / 2

    return overall_similarity


def _find_closest_match(s: Sofascore, holdet_candidates: list[Holdet]) -> Holdet:
    max_similarity = 0.0
    closest_match_idx = None

    for h in holdet_candidates:
        # Read the similarity between players
        similarity = _similarity(s.player, h.player)

        # If the similarity is greater than the maximum similarity update the
        # closest match
        if similarity > max_similarity:
            max_similarity = similarity
            closest_match_idx = holdet_candidates.index(h)

    return holdet_candidates[closest_match_idx]  #  type: ignore


def get_sofascore(tournament: sofascore.Tournament) -> list[Sofascore]:
    client = sofascore.Client()

    players: list[Sofascore] = []
    for round in range(1, 37):
        for game in client.games(tournament, round):
            lineup = client.lineup(game).all
            for player, stats in lineup:
                found = False
                for p in players:
                    if player == p.player:
                        p.stats.append(stats)
                        found = True
                        break
                if not found:
                    players.append(Sofascore(player=player, stats=[stats]))

    return players


def get_holdet(game: holdet.Game) -> list[Holdet]:
    client = holdet.Client()

    # Use a dict for lookups to improve speed
    players_dict: dict[int, Holdet] = {}
    for round in client.rounds(game):
        stats = client.statistics(game, round)
        for stat in stats:
            player_id = stat.player.id
            if player_id in players_dict:
                players_dict[player_id].stats.append(stat)
            else:
                players_dict[player_id] = Holdet(player=stat.player, stats=[stat])

    return list(players_dict.values())


if __name__ == "__main__":
    # Setup logger with Rich formatting
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    console = Console()

    with console.status("Fetching data...") as status:
        holdet_candidates = get_holdet(
            holdet.Game(
                644,  # Spring 2023
                422,  # Premier League
            )
        )
        status.console.log(f"Found {len(holdet_candidates)} players on Holdet")
        sofascore_candidates = get_sofascore(
            sofascore.Tournament(
                17,  # Premier League
                41886,  # 2022/2023
            )
        )
        candidates: list[Candidate] = []
        for s in sofascore_candidates:
            # Find the closest match in Holdet for the current Sofascore player and
            # remove it from the list afterwards.
            h = _find_closest_match(s, holdet_candidates)
            holdet_candidates.remove(h)
            candidates.append(Candidate(h, s))
        status.console.log(f"Found {len(candidates)} players on Sofascore")

    # Extract and flatten the data
    data = []
    for candidate in candidates:
        for round in candidate.rounds:
            for stat in round.stats:
                row = {
                    "id": candidate.id,
                    "round": round.number,
                }
                row.update(stat.features)
                row.update({"growth": round.growth})
                data.append(row)

    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Define features (X) and target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error:", rmse)

    # with console.status("Finding optimal team..."):
    #     solution = lp.find_optimal_team(candidates, 70 * 1000000)
    #     status.console.log(f"Found optimal 11 out of {len(candidates)} players")

    # print(Formation(solution))
