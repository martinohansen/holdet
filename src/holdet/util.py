from holdet.data import holdet, sofascore
from holdet.v2 import Holdet, Sofascore


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


def get_similarity(s: sofascore.Player, h: holdet.Player) -> float:
    team_similarity = jaccard_similarity(s.team.name, h.team.name)
    name_similarity = jaccard_similarity(s.name, h.person.name)

    # Calculate the overall similarity as the average of the team and
    # name similarities
    overall_similarity = (team_similarity + name_similarity) / 2

    return overall_similarity


def find_closest_match(s: Sofascore, holdet_candidates: list[Holdet]) -> Holdet:
    max_similarity = 0.0
    closest_match_idx = None

    for h in holdet_candidates:
        # Read the similarity between players
        similarity = get_similarity(s.player, h.player)

        # If the similarity is greater than the maximum similarity update the
        # closest match
        if similarity > max_similarity:
            max_similarity = similarity
            closest_match_idx = holdet_candidates.index(h)

    if closest_match_idx is None:
        raise ValueError("No match found")
    return holdet_candidates[closest_match_idx]
