#!/usr/bin/env python3

import logging
from dataclasses import dataclass

from rich import print
from rich.logging import RichHandler
from rich.progress import track

from holdet import holdet
from sofascore import sofascore


@dataclass
class HoldetCandidate:
    player: holdet.Player
    stats: list[holdet.Statistics]


@dataclass
class SofascoreCandidate:
    player: sofascore.Player
    stats: list[sofascore.Statistics]


@dataclass
class Candidate:
    holdet: HoldetCandidate
    sofascore: SofascoreCandidate

    # Whether the player is the captain or not
    captain: bool = False

    @property
    def name(self) -> str:
        return self.holdet.player.person.name

    @property
    def team(self) -> str:
        return self.holdet.player.team.name

    @property
    def emoji(self) -> str:
        if self.captain:
            return "👑"
        if self.holdet.player.position == holdet.Position.KEEPER:
            return "🧤"
        return "⚽️"

    @property
    def similarity(self) -> float:
        return _similarity(self.sofascore.player, self.holdet.player)

    @property
    def growths(self) -> list[float]:
        return [stat.values.growth for stat in self.holdet.stats]

    @property
    def growthTotal(self) -> float:
        return sum(stat.values.growth for stat in self.holdet.stats)

    def xGrowth(self, stat) -> float:
        # The argument type hint did not work because the module and class
        # shares the same name. This works for now.
        assert isinstance(stat, sofascore.Statistics)

        points = holdet.Points(self.holdet.player.position)
        growth: float = 0

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

        # Finance
        if self.captain:
            growth = growth * 2

        return growth

    @property
    def xGrowths(self) -> list[float]:
        return [self.xGrowth(stat) for stat in sorted(self.sofascore.stats)]

    @property
    def xGrowthTotal(self) -> float:
        return sum(self.xGrowth(stat) for stat in self.sofascore.stats)

    def xGrowthPredict(self, alpha: float = 0.5) -> float:
        """
        Predict the growth for the next game using exponential moving average
        (EMA). Set alpha to adjust the smoothing factor, between 0 and 1. Higher
        values give more weight to recent stats.
        """
        ema = 0.0
        for i, stat in enumerate(sorted(self.sofascore.stats)):
            if i == 0:
                ema = self.xGrowth(stat)
            else:
                ema = alpha * self.xGrowth(stat) + (1 - alpha) * ema
        return ema

    def __lt__(self, other: "Candidate"):
        return self.xGrowthPredict() < other.xGrowthPredict()

    def __repr__(self) -> str:
        xGrowth_list = [f"{growth / 1000:.0f}K" for growth in self.xGrowths]
        growth_list = [f"{growth / 1000:.0f}K" for growth in self.growths]
        return (
            f"{self.emoji} {self.name} ({self.team}),"
            f" xGrowthPredict={self.xGrowthPredict() / 1000:.0f}K,"
            f" xGrowthTotal={self.xGrowthTotal / 1000000:.2f}M"
            f" ({', '.join(xGrowth_list)}),"
            f" growthTotal={self.growthTotal / 1000000:.2f}M ({', '.join(growth_list)})"
        )


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


def _find_closest_match(
    s: SofascoreCandidate, holdet_candidates: list[HoldetCandidate]
) -> HoldetCandidate:
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


def get_sofascore(tournament: sofascore.Tournament) -> list[SofascoreCandidate]:
    client = sofascore.Client()

    players: list[SofascoreCandidate] = []
    for round in track(range(1, 36)):
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
                    players.append(SofascoreCandidate(player=player, stats=[stats]))

    return players


def get_holdet(game: holdet.Game) -> list[HoldetCandidate]:
    client = holdet.Client()

    # Use a dict for lookups to improve speed
    players_dict: dict[int, HoldetCandidate] = {}
    for round in track(client.rounds(game)):
        stats = client.statistics(game, round)
        for stat in stats:
            player_id = stat.player.id
            if player_id in players_dict:
                players_dict[player_id].stats.append(stat)
            else:
                players_dict[player_id] = HoldetCandidate(
                    player=stat.player, stats=[stat]
                )

    return list(players_dict.values())


if __name__ == "__main__":
    # Setup logger with Rich formatting
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    holdet_candidates = get_holdet(
        holdet.Game(
            644,  # Spring 2023
            422,  # Premier League
        )
    )
    sofascore_candidates = get_sofascore(
        sofascore.Tournament(
            17,  # Premier League
            41886,  # 2022/2023
        )
    )

    candidates: list[Candidate] = []
    for s in track(sofascore_candidates):
        # Find the closest match in Holdet for the current Sofascore player and
        # remove it from the list afterwards.
        h = _find_closest_match(s, holdet_candidates)
        holdet_candidates.remove(h)
        candidates.append(Candidate(h, s))

    print(sorted(candidates)[-10:])
