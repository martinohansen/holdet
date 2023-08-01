#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich import print
from rich.logging import RichHandler
from rich.table import Table
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from holdet.data import holdet, sofascore
from holdet.solver import lp

from . import campaigns


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

    def features(self, stat: sofascore.Statistics) -> dict[str, float | int]:
        """
        Return a dictionary with all the features that can be used to train a
        model.
        """

        return {
            # Data know before hand which can be used to predict the outcome.
            # TODO: Add data from betting sites
            "team": stat.game.home.id
            if stat.side == sofascore.Side.HOME
            else stat.game.away.id,
            "opponent": stat.game.away.id
            if stat.side == sofascore.Side.HOME
            else stat.game.home.id,
            "side": stat.side.value,
            # Stats known after game has finished
            "substitute": int(stat.substitute),
            "assists": stat.assists,
            "expectedAssists": stat.expectedAssists,
            "expectedGoals": stat.expectedGoals,
            "goals": stat.goals,
            "goalsPrevented": stat.goalsPrevented,
            "minutesPlayed": stat.minutesPlayed,
            "onTargetScoringAttempt": stat.onTargetScoringAttempt,
            "savedShotsFromInsideTheBox": stat.savedShotsFromInsideTheBox,
            "saves": stat.saves,
            "team_goals": stat.team_goals,
            "team_goals_conceded": stat.team_goals_conceded,
            "win": int(stat.win),
            "loss": int(stat.loss),
            "draw": int(stat.draw),
            "clean_sheet": int(stat.clean_sheet),
            "decisive_goal_for_draw": int(stat.decisive_goal_for_draw),
            "decisive_goal_for_win": int(stat.decisive_goal_for_win),
        }

    def df(self, train: bool = False) -> pd.DataFrame:
        """
        Return a data frame with features for all rounds. Include the actual
        growth if train is True.
        """
        data = []
        for round in self.rounds:
            row: dict[str, int | float] = {
                "id": self.id,
                "round": round.number,
                "position": round.position.value,
            }

            # If the candidate has no stats for the round, skip it
            if len(round.stats) == 0:
                continue

            # Sum stats from multiple stats in the same round, this can
            # happen as some rounds have multiple games for some teams.
            for stat in round.stats:
                for key, value in self.features(stat).items():
                    if key in row:
                        row[key] += value
                    else:
                        row[key] = value

            if train:
                row.update({"growth": round.growth})

            data.append(row)
        return pd.DataFrame(data)

    def __eq__(self, other: object):
        if not isinstance(other, Candidate):
            return NotImplemented
        return self.id == other.id

    def __lt__(self, other: "Candidate"):
        return self.value < other.value

    def __hash__(self):
        return hash(self.avatar.player.person)

    def __repr__(self) -> str:
        return (
            f"{self.emoji} {self.name} ({self.team}), value={self.value / 1000000:.1f}M"
        )


class Formation:
    def __init__(self, solution: list[Candidate], xValue: lp.xValue):
        self.position: dict[str, list[Candidate]] = {
            "keeper": [],
            "defenses": [],
            "midfielders": [],
            "forwards": [],
        }
        self.xValue = xValue
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
        return (
            f"{len(self.position['defenses'])}-"
            f"{len(self.position['midfielders'])}-"
            f"{len(self.position['forwards'])}"
        )

    def __rich__(self) -> Table:
        table = Table(title=f"\nXI ({self})")
        table.add_column("Position")
        table.add_column("Players")
        table.add_column("xGrowth", justify="right")
        for index, (position, players) in enumerate(self):
            for player in players:
                # Calculate the xGrowth using the evaluator to show in the
                # output what the model predicts.
                xGrowth = self.xValue(player) - player.value

                table.add_row(
                    position.capitalize(), str(player), f"{xGrowth / 1000:.0f}K"
                )

                # Avoid repeating the position name in the same row
                position = ""

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

    if closest_match_idx is None:
        raise ValueError("No match found")
    return holdet_candidates[closest_match_idx]


class Game:
    def __init__(self, campaign: campaigns.Campaign) -> None:
        self.candidates: list[Candidate] = []

        logging.info("Fetching data...")
        holdet_candidates = self._get_holdet(campaign.holdet)
        for s in self._get_sofascore(campaign.sofascore):
            # Find the closest match in Holdet for the current Sofascore player and
            # remove it from the list afterwards.
            h = _find_closest_match(s, holdet_candidates)
            holdet_candidates.remove(h)
            self.candidates.append(Candidate(h, s))
        logging.info(f"Collected data from {len(self.candidates)} candidates")

    def _get_sofascore(self, tournament: sofascore.Tournament) -> list[Sofascore]:
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

    def _get_holdet(self, game: holdet.Game) -> list[Holdet]:
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

    def df(self) -> pd.DataFrame:
        """
        Return all candidates in a flattened data frame for training
        """
        data = []
        for candidate in self.candidates:
            data.append(candidate.df(train=True))

        df = pd.concat(data)
        logging.info(f"Created data frame: {len(df)} rows, {len(df.columns)} columns")
        return df


def xGrowthML(c: Candidate, model) -> float:
    # Catch players with no stats, most likely because they never played.
    # The model is going to complain if this happens so we need to just
    # return 0
    if len(c.df()) == 0:
        return 0.0
    return model.predict(c.df())[-1]


def xGrowthEMA(c: Candidate, alpha: float) -> float:
    """
    Predict the growth for the next game using exponential moving average
    (EMA). Set alpha to adjust the smoothing factor, between 0 and 1. Higher
    values give more weight to recent stats.
    """
    ema = 0.0
    for i, round in enumerate(sorted(c.rounds)):
        if i == 0:
            ema = round.xGrowth
        else:
            ema = alpha * round.xGrowth + (1 - alpha) * ema
    return ema


def main():
    # Setup logger and console with Rich formatting
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    # Init the game and get a dataframe of the candidates.
    game = Game(campaigns.PRIMER_LEAGUE_2023)
    df = game.df()

    # X is the data we're using to predict y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test data
    logging.info("Training model on data...")
    y_pred = model.predict(X_test)

    score = model.score(X_test, y_test)
    avg_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"Model evaluation: {avg_rmse=:.2f}, {score=:.4f}")

    # Simple evaluator using linear regression to predict the growth
    def xValue(c: Candidate) -> float:
        if c.captain:
            return c.value + (xGrowthML(c, model) * 2)
        return c.value + xGrowthML(c, model)

    logging.info("Finding optimal team...")
    solution = lp.find_optimal_team(game.candidates, xValue, 70 * 1000000)

    print(Formation(solution, xValue))
