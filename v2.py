#!/usr/bin/env python3


from dataclasses import dataclass

from rich import print

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
    win: bool
    draw: bool
    loss: bool
    team_goal: int
    opponent_goal: int
    win_away: bool
    loss_away: bool

    # Special
    clean_sheet: bool
    played: bool
    subbed_in: bool
    subbed_out: bool
    saved_penalty: int
    missed_penalty: int
    hattrick: int


@dataclass
class Candidate:
    holdet_character: holdet.Character
    sofascore_player: sofascore.Player
    sofascore_stats: list[sofascore.Statistics]

    @property
    def name(self) -> str:
        return self.holdet_character.name

    @property
    def similarity(self) -> float:
        return _similarity(self.sofascore_player, self.holdet_character)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Candidate):
            return self.name == __value.name
        if isinstance(__value, sofascore.Player):
            return self.sofascore_player == __value
        raise NotImplementedError


if __name__ == "__main__":
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
    for round in range(30, 35):
        for game in c.games(pl, round):
            # TODO: Fix 404 from missing lineups, I suspect this is from the
            # canceled/rescheduled games.
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

    print(candidates[:1])
