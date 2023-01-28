#!/usr/bin/env python3


import pickle

import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

from fotmob import fotmob
from holdet import holdet


class Baller:
    def __init__(self, fotmob: fotmob.Player, holdet: holdet.Character) -> None:
        self.fotmob = fotmob
        self.holdet = holdet

    def __repr__(self) -> str:
        name = self.fotmob.name
        fotmob = self.fotmob
        holdet = self.holdet
        return f"Baller({name=!r}, {fotmob=!r}, {holdet=!r})"

    @property
    def goals(self) -> int:
        if self.fotmob.current_season:
            if self.fotmob.current_season.goals:
                return self.fotmob.current_season.goals
        return 0

    @property
    def assists(self) -> int:
        if self.fotmob.current_season:
            if self.fotmob.current_season.assists:
                return self.fotmob.current_season.assists
        return 0


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
            team_similarity = self.__jaccard_similarity(player.team, character.team)

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

    # Create a DataFrame from the Baller objects
    data = {
        "goals": [baller.goals for baller in ballers],
        "assists": [baller.assists for baller in ballers],
        "round_growth": [baller.holdet.growth for baller in ballers],
    }
    df = pd.DataFrame(data)

    # Split the data into training and test sets
    train_df = df.iloc[:3]
    test_df = df.iloc[3:]

    # Extract the input features and target variable
    X_train = train_df[["goals", "assists"]]
    y_train = train_df["round_growth"]
    X_test = test_df[["goals", "assists"]]
    y_test = test_df["round_growth"]

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    score = model.score(X_test, y_test)
    print(f"R^2 score: {score:.2f}")
