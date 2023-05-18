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
    def similarity(self) -> float:
        return _similarity(self.sofascore_player, self.holdet_character)

    def xGrowth(self, stat: sofascore.Statistics) -> float:
        points = holdet.Points(self.holdet_character)
        growth: float = 0

        # Goals and assists
        growth += points.goal * stat.expectedGoals
        growth += points.assist * stat.expectedAssists
        growth += points.shot_on_goal * stat.onTargetScoringAttempt

        # Decisive goals
        # growth += 30000 * self.x_decisive_goals_win
        # growth += 15000 * self.x_decisive_goals_draw

        # Fair play
        # growth += points.yellow_card * stat.yellowCard
        # growth += -50000 * self.xRed

        # # Team performance
        # growth += 25000 * self.xWin
        # growth += 5000 * self.xDraw
        # growth += -15000 * self.xLoss
        # growth += 10000 * self.xTeamGoals
        # growth += -8000 * self.xTeamConceded
        # growth += 10000 * self.xWinAway
        # growth += -1000 * self.xLossHome

        # # Special
        # growth += clean_sheet_points * self.xCS
        # growth += 7000 * self.xIn
        # growth += -5000 * self.xOut
        # growth += 100000 * self.xHattrick

        # Finance
        if self.captain:
            growth = growth * 2

        return growth

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Candidate):
            return self.name == __value.name
        if isinstance(__value, sofascore.Player):
            return self.sofascore_player == __value
        raise NotImplementedError

    def __repr__(self) -> str:
        # Find suitable emoji to identify player
        if self.captain:
            emoji = "👑"
        if self.holdet_character.keeper:
            emoji = "🧤"
        emoji = "⚽️"

        return f"{emoji} {self.name} ({self.team}) " + ", ".join(
            str(self.xGrowth(stat)) for stat in sorted(self.sofascore_stats)
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
