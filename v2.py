#!/usr/bin/env python3

import logging
from dataclasses import dataclass

from rich import print
from rich.logging import RichHandler
from rich.progress import track

from holdet import holdet
from sofascore import sofascore


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


def _similarity(player: sofascore.Player, character: holdet.Character) -> float:
    # Calculate the Jaccard similarity between the team name and the
    # character's team name
    team_similarity = _jaccard_similarity(player.team.name, character.team)

    # Calculate the Jaccard similarity between the character name and the
    # character's name
    name_similarity = _jaccard_similarity(player.name, character.name)

    # Calculate the overall similarity as the average of the team and
    # name similarities
    overall_similarity = (team_similarity + name_similarity) / 2

    return overall_similarity


def _find_closest_character(
    player: sofascore.Player,
    characters: list[holdet.Character],
) -> holdet.Character:
    max_similarity = 0.0
    closest_match_idx = None

    for character in characters:
        # Read the similarity between the player and the character
        similarity = _similarity(player, character)

        # If the similarity is greater than the maximum similarity update the
        # closest match
        if similarity > max_similarity:
            max_similarity = similarity
            closest_match_idx = characters.index(character)

    return characters[closest_match_idx]  #  type: ignore


@dataclass
class Stats:
    # Goals and assists
    goals: int
    assists: int
    xGoals: float
    xAssists: float
    own_goals: int
    shots_on_target: int

    # Decisive goals
    winning_goals: int
    equalizing_goals: int

    goals_conceded: int

    # Fair play
    yellow_card: int
    red_card: int

    # Team performance
    win: int
    draw: int
    loss: int
    team_goal: int
    opponent_goal: int
    win_away: int
    loss_away: int

    # Special
    clean_sheet: int
    played: int
    subbed_in: int
    subbed_out: int
    saved_penalty: int
    missed_penalty: int
    hattrick: int

    @classmethod
    def from_sofascore(self, stats: list[sofascore.Statistics]) -> "Stats":
        return Stats(
            goals=sum(stat.goals for stat in stats),
            assists=sum(stat.assists for stat in stats),
            xGoals=sum(stat.expectedGoals for stat in stats),
            xAssists=sum(stat.expectedAssists for stat in stats),
            # TODO: Populate the rest of the stats with zeros
            own_goals=0,
            shots_on_target=0,
            winning_goals=0,
            equalizing_goals=0,
            goals_conceded=0,
            yellow_card=0,
            red_card=0,
            win=0,
            draw=0,
            loss=0,
            team_goal=0,
            opponent_goal=0,
            win_away=0,
            loss_away=0,
            clean_sheet=0,
            played=0,
            subbed_in=0,
            subbed_out=0,
            saved_penalty=0,
            missed_penalty=0,
            hattrick=0,
        )


@dataclass
class Candidate:
    holdet_character: holdet.Character
    sofascore_player: sofascore.Player
    sofascore_stats: list[sofascore.Statistics]
    captain: bool = False

    @property
    def name(self) -> str:
        return self.holdet_character.name

    @property
    def team(self) -> str:
        return self.holdet_character.team

    @property
    def stats(self) -> Stats:
        return Stats.from_sofascore(self.sofascore_stats)

    @property
    def emoji(self) -> str:
        if self.captain:
            return "👑"
        if self.holdet_character.keeper:
            return "🧤"
        return "⚽️"

    @property
    def similarity(self) -> float:
        return _similarity(self.sofascore_player, self.holdet_character)

    @property
    def warning(self) -> str:
        warnings: list[str] = []
        if self.similarity < 0.6:
            warnings.append(f"low similarity: {self.similarity * 100:.0f}%")

        if warnings:
            return "🚨 " + ", ".join(warnings)
        return ""

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Candidate):
            return self.name == __value.name
        if isinstance(__value, sofascore.Player):
            return self.sofascore_player == __value
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.emoji} {self.name} ({self.team}) {self.stats} " + (
            self.warning if self.warning else ""
        )


if __name__ == "__main__":
    # Setup logger with Rich formatting
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    h = holdet.Client()
    characters = h.characters(
        tournament_id=422,  # Premier League
        game_id=644,  # 2022/2023 end of season
        game_round=3,
    )
    pl = sofascore.Tournament(
        17,  # Premier League
        41886,  # 2022/2023
    )
    c = sofascore.Client()

    candidates: list[Candidate] = []
    for round in track(range(20, 35), description="Rounds..."):
        for game in c.games(pl, round):
            players = c.lineup(game).all
            for player, stats in players:
                if player not in candidates:
                    # Find and pop the closest character from the list. This is
                    # to reduce the chance of matching sofascore and holdet
                    # players wrongly.
                    character = _find_closest_character(player, characters)
                    characters.pop(characters.index(character))

                    candidates.append(
                        Candidate(
                            holdet_character=character,
                            sofascore_player=player,
                            sofascore_stats=[stats],
                        )
                    )
                else:
                    for candidate in candidates:
                        if candidate == player:
                            candidate.sofascore_stats.append(stats)

    print(candidates)
