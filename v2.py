#!/usr/bin/env python3

from fotmob import fotmob
from holdet import holdet


def jaccard_similarity(str1: str, str2: str) -> float:
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


def find_closest_character(whoami: fotmob.Player, candidates: list[holdet.Character]):
    # Initialize a variable to store the maximum similarity
    max_similarity = 0.0

    # Initialize a variable to store the closest match
    closest_match_idx = None

    # Iterate over the players in the list
    for candidate in candidates:
        # Calculate the Jaccard similarity between the team name and the
        # candidate's team name
        team_similarity = jaccard_similarity(whoami.team, candidate.team)

        # Calculate the Jaccard similarity between the candidate name and the
        # candidate's name
        name_similarity = jaccard_similarity(whoami.name, candidate.name)

        # Calculate the overall similarity as the average of the team and
        # name similarities
        overall_similarity = (team_similarity + name_similarity) / 2

        # If the overall similarity is greater than the maximum similarity,
        # update the closest match
        if overall_similarity > max_similarity:
            max_similarity = overall_similarity
            closest_match_idx = candidates.index(candidate)

    return candidates.pop(closest_match_idx)  #  type: ignore


HOLDET: holdet.Holdet = holdet.Holdet()


class Baller:
    def __init__(self, fotmob: fotmob.Player) -> None:
        self.fotmob = fotmob
        self.holdet: holdet.Character = find_closest_character(
            self.fotmob, HOLDET.Characters
        )

    def __repr__(self) -> str:
        name = self.fotmob.name
        value = self.holdet.value
        return f"Baller({name=!r}, {value=!r})"


ballers: list[Baller] = []

premier_league = fotmob.League()
for team in premier_league.teams:
    for player in team.players:
        ballers.append(Baller(player))

for baller in ballers[:5]:
    print(baller)
