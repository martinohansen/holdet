#!/usr/bin/env python3


import pickle

from fotmob import fotmob
from holdet import holdet


class Baller:
    def __init__(
        self,
        fotmob: fotmob.Player,
        holdet: holdet.Character,
        captain: bool = False,
    ) -> None:
        self.fotmob = fotmob
        self.holdet = holdet
        self.captain = captain

    def __repr__(self) -> str:
        name = self.fotmob.name
        fotmob = self.fotmob
        holdet = self.holdet
        xGrowth = self.xGrowth
        return f"Baller({name=!r}, {xGrowth=!r}, {fotmob=!r}, {holdet=!r})"

    @property
    def xGrowth(self) -> float:
        growth: float = 0

        if self.holdet.keeper:
            goal_points = 250000
        elif self.holdet.defense:
            goal_points = 175000
        elif self.holdet.midfielder:
            goal_points = 150000
        elif self.holdet.forward:
            goal_points = 125000
        else:
            goal_points = 0

        if self.holdet.keeper:
            clean_sheet_points = 75000
        elif self.holdet.defense:
            clean_sheet_points = 50000
        else:
            clean_sheet_points = 0

        # Season stats
        if self.fotmob.current_season:
            # Goals and assists
            growth += goal_points * self.fotmob.current_season.expected_goals
            growth += 60000 * self.fotmob.current_season.assists

            # Fair play
            growth += 20000 * self.fotmob.current_season.yellow_card
            growth += 50000 * self.fotmob.current_season.red_card

            # Special
            growth += clean_sheet_points * self.fotmob.current_season.clean_sheets
            growth += clean_sheet_points * self.fotmob.current_season.matches
            # TODO(Martin): Implement stats
            # growth += -5000 * self.xOut
            # growth += 100000 * self.xHattrick

        # Team performance
        growth += 25000 * self.fotmob.team.wins
        growth += 5000 * self.fotmob.team.draws
        growth += -15000 * self.fotmob.team.loss
        # TODO(Martin): Implement stats
        # growth += 10000 * self.fotmob.team.xTeamGoals
        # growth += -8000 * self.fotmob.team.xTeamConceded
        # growth += 10000 * self.fotmob.team.xWinAway
        # growth += -1000 * self.fotmob.team.xLossHome

        # Finance
        if self.captain:
            growth += growth * 2

        return growth


class BallerFactory:
    def __init__(self, fotmob: fotmob.League, holdet: holdet.Game) -> None:
        self.fotmob = fotmob
        self.holdet = holdet

    def __jaccard_similarity(self, str1: str, str2: str) -> float:
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

    def __find_closest_character(self, player: fotmob.Player):
        # Initialize a variable to store the maximum similarity
        max_similarity = 0.0

        # Initialize a variable to store the closest match
        closest_match_idx = None

        # Iterate over the players in the list
        for character in self.holdet.Characters:
            # Calculate the Jaccard similarity between the team name and the
            # character's team name
            team_similarity = self.__jaccard_similarity(
                player.team.name, character.team
            )

            # Calculate the Jaccard similarity between the character name and the
            # character's name
            name_similarity = self.__jaccard_similarity(player.name, character.name)

            # Calculate the overall similarity as the average of the team and
            # name similarities
            overall_similarity = (team_similarity + name_similarity) / 2

            # If the overall similarity is greater than the maximum similarity,
            # update the closest match
            if overall_similarity > max_similarity:
                max_similarity = overall_similarity
                closest_match_idx = self.holdet.Characters.index(character)

        return self.holdet.Characters.pop(closest_match_idx)  #  type: ignore

    def __call__(self) -> list[Baller]:
        ballers: list[Baller] = []
        for team in self.fotmob.teams:
            for player in team.players:
                ballers.append(
                    Baller(
                        player,
                        self.__find_closest_character(player),
                    )
                )
        return ballers


if __name__ == "__main__":
    cache_file = ".ballers.pkl"
    try:
        with open(cache_file, "rb") as file_rb:
            factory = pickle.load(file_rb)
    except FileNotFoundError:
        with open(cache_file, "wb") as file_wb:
            factory = BallerFactory(fotmob.League(), holdet.Game())
            pickle.dump(factory, file_wb)

    ballers = factory()
    print(ballers[2])
    breakpoint()
