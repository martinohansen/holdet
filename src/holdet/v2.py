#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense  # type: ignore
from keras.models import Sequential  # type: ignore
from rich import print
from rich.logging import RichHandler
from rich.table import Table
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from holdet.data import holdet, sofascore
from holdet.solver import lp


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
    def team_id(self) -> int:
        return self.avatar.player.team.id

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
            # TODO: Add data from betting sites
            "opponent": stat.game.away.id
            if stat.side == sofascore.Side.HOME
            else stat.game.home.id,
            "side": stat.side.value,
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

    def aggregate_features(self, round: Round) -> dict:
        """
        Aggregate all the features from a round into one. This is needed for
        rounds with multiple games in them. The features are summed together.

        Its not ideal for stuff like opponent or side which will simply be added
        together, but thats we can do for now.
        """
        round_stats: dict[str, int | float] = {}
        for stat in round.stats:
            for key, value in self.features(stat).items():
                round_stats[key] = round_stats.get(key, 0) + value
        return round_stats

    def generate_dataframe(self) -> pd.DataFrame:
        """
        Generate a dataframe for the candidate with features for every round
        """
        data = []

        for round in self.rounds:
            row = {
                "id": self.id,
                "round": round.number,
                "position": round.position.value,
                "team": self.team_id,
                "growth": round.growth,
            }

            # Append all features to the row
            features = self.aggregate_features(round)
            for key, value in features.items():
                row[key] = value

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
    def __init__(self) -> None:
        self.candidates: list[Candidate] = []
        self.holdet_client = holdet.Client()
        self.sofascore_client = sofascore.Client()

        logging.info("Fetching data from Holdet...")
        holdet_candidates = self._get_holdet(
            self.holdet_client.tournament(holdet.PRIMER_LEAGUE),
            [
                holdet.Game(holdet.PRIMER_LEAGUE_FALL_2022),
                holdet.Game(holdet.PRIMER_LEAGUE_SPRING_2023),
                holdet.Game(holdet.PRIMER_LEAGUE_FALL_2023),
            ],
        )

        logging.info("Fetching data from Sofascore...")
        tournament = sofascore.Tournament(sofascore.PRIMER_LEAGUE)
        for s in self._get_sofascore(
            tournament,
            seasons=[
                tournament.season(sofascore.PRIMER_LEAGUE_2022_2023),
                tournament.season(sofascore.PRIMER_LEAGUE_2023_2024),
            ],
        ):
            # Find the closest match in Holdet for the current Sofascore player and
            # remove it from the list afterwards.
            h = _find_closest_match(s, holdet_candidates)
            holdet_candidates.remove(h)
            self.candidates.append(Candidate(h, s))
        logging.info(f"Collected data from {len(self.candidates)} candidates")

    def _get_sofascore(
        self,
        tournament: sofascore.Tournament,
        seasons: list[sofascore.Season],
    ) -> list[Sofascore]:
        players: list[Sofascore] = []
        for season in seasons:
            logging.info(f"Fetching data for {season}...")
            # Get the current round for the season and iterate over all the
            # rounds so far to gather the player stats.
            current_round = self.sofascore_client.current_round(tournament, season)
            for round in range(1, current_round + 1):
                for game in self.sofascore_client.games(season, round):
                    lineup = self.sofascore_client.lineup(game).all
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

    def _get_holdet(
        self,
        tournament: holdet.Tournament,
        games: list[holdet.Game],
    ) -> list[Holdet]:
        # Use a dict for lookups to improve speed
        players_dict: dict[int, Holdet] = {}
        for game in games:
            logging.info(f"Fetching data for {game}...")
            for round in self.holdet_client.rounds(game):
                stats = self.holdet_client.statistics(tournament, game, round)
                for stat in stats:
                    player_id = stat.player.id
                    if player_id in players_dict:
                        players_dict[player_id].stats.append(stat)
                    else:
                        players_dict[player_id] = Holdet(
                            player=stat.player, stats=[stat]
                        )

        return list(players_dict.values())

    def generate_dataframe(self) -> pd.DataFrame:
        """
        Return a combined dataframe for all the candidates.
        """
        data = []
        for candidate in self.candidates:
            data.append(candidate.generate_dataframe())

        df = pd.concat(data)
        logging.info(f"Created data frame: {len(df)} rows, {len(df.columns)} columns")
        return df


def xGrowthML(c: Candidate, model) -> float:
    # Catch players with no stats, most likely because they never played.
    # The model is going to complain if this happens so we need to just
    # return 0
    if len(c.generate_dataframe()) == 0:
        return 0.0
    return model.predict(c.generate_dataframe())[-1]


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


class Model:
    def __init__(self, n_steps: int = 10) -> None:
        self.n_steps = n_steps

        self.model = Sequential()
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare the data for training the model. This includes normalizing the
        data and combining the data into a X and y array.
        """

        # NaN values is often because no stat was recorded for the given player
        # for that game. That means that the player was not on the pitch and
        # thus 0 is the proper value to use.
        df.fillna(0, inplace=True)

        # Normalize the data to gain better results. Using separate scaler to
        # allow for independently inverting again.
        df[df.columns.difference(["growth"])] = self.scaler_x.fit_transform(
            df[df.columns.difference(["growth"])]
        )
        df["growth"] = self.scaler_y.fit_transform(df[["growth"]])

        # Combine number of time_steps into a single array and set the target to the
        # growth of the following round. E.g. combine values from round 1-10 and set
        # the target to the growth of round 11.
        time_steps = self.n_steps
        features = []
        target = []
        for _, group in df.groupby("id"):
            group.drop(columns=["id", "round"], inplace=True)

            for i in range(time_steps, len(group)):
                features.append(group.iloc[i - time_steps : i].values)
                target.append(group.iloc[i]["growth"])

        return np.array(features), np.array(target)

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the LSTM model using the given data frame. Returns the model for use
        to predict the growth of a player.
        """
        X, y = self.prepare_data(df)

        # Build model with layers according to the data
        self.model.add(
            LSTM(
                units=50,
                return_sequences=True,
                input_shape=(
                    X.shape[1],  # Number of time steps
                    X.shape[2],  # Number of features
                ),
            )
        )
        self.model.add(LSTM(units=50))
        self.model.add(Dense(units=50, activation="relu"))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer="adam", loss="mean_squared_error")

        # Train the model
        logging.info("Training model...")
        self.model.fit(
            X,
            y,
            epochs=10,  # Number of iterations over the entire dataset
            validation_split=0.2,  # Use 20% of the data for validation
        )

    def evaluate(self, df: pd.DataFrame) -> str:
        """
        Return the loss and RMSE in a string for the given data frame.
        """
        X, y = self.prepare_data(df)
        loss = self.model.evaluate(X, y)
        return f"Model evaluation: {loss=!r}"

    def predict(self, df: pd.DataFrame):
        """
        Predict the growth for the next game using the given data frame. The
        data frame should contain the same columns as the one used for training
        the model.
        """
        X, _ = self.prepare_data(df)
        y = self.model.predict(X)
        return self.scaler_y.inverse_transform(y)


def main():
    # Setup logger and console with Rich formatting
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    game = Game()
    model = Model()
    breakpoint()
    df = game.generate_dataframe()
    model.train(df)
    print(model.evaluate(df))

    candidate = game.candidates[0]
    print(candidate.generate_dataframe())
    print(model.predict(candidate.generate_dataframe()))

    # def xValue(c: Candidate) -> float:
    #     """
    #     Simple evaluator using model to predict the growth
    #     """
    #     if c.captain:
    #         return c.value + (xGrowthML(c, model) * 2)
    #     return c.value + xGrowthML(c, model)

    # logging.info("Finding optimal team...")
    # solution = lp.find_optimal_team(game.candidates, xValue, 70 * 1000000)

    # print(Formation(solution, xValue))
