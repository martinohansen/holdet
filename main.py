#!/usr/bin/env python3

from dataclasses import dataclass
from difflib import SequenceMatcher

import requests

url = "https://data.fotmob.com/stats/47/season/17664/expected_goals.json"
response = requests.request("GET", url)
expected_goals = response.json()

url = "https://data.fotmob.com/stats/47/season/17664/expected_assists.json"
response = requests.request("GET", url)
expected_assists = response.json()

url = "https://data.fotmob.com/stats/47/season/17664/clean_sheet.json"
response = requests.request("GET", url)
clean_sheet = response.json()

url = "https://api.holdet.dk/tournaments/422?appid=holdet"
response = requests.request("GET", url)
tournament = response.json()

url = "https://api.holdet.dk/games/644/rounds/1/statistics?appid=holdet"
response = requests.request("GET", url)
game = response.json()


@dataclass
class Baller:
    name: str
    value: int
    popularity: float
    trend: int
    position: int

    similarity_threshold: float = 0.8

    def __populate_stat(self, stats) -> float:
        for stat in stats:
            similarity = self.__similarity(stat["ParticipantName"])
            if similarity > self.similarity_threshold:
                return stat["StatValue"]
        return 0

    def __similarity(self, to):
        return SequenceMatcher(None, self.name, to).ratio()

    def __per_value(self, stat: float) -> float:
        if self.value != 0 and stat != 0:
            return self.value / stat
        else:
            return 99999999

    @property
    def xG(self) -> float:
        return self.__populate_stat(expected_goals["TopLists"][0]["StatList"])

    @property
    def xGPerValue(self) -> float:
        return self.__per_value(self.xG)

    @property
    def xA(self) -> float:
        return self.__populate_stat(expected_assists["TopLists"][0]["StatList"])

    @property
    def xAPerValue(self) -> float:
        return self.__per_value(self.xA)

    @property
    def CS(self) -> float:
        return self.__populate_stat(clean_sheet["TopLists"][0]["StatList"])

    @property
    def CSPerValue(self) -> float:
        return self.__per_value(self.CS)


if __name__ == "__main__":
    keepers = []
    defenses = []
    midfielders = []
    forwards = []

    # Loop over all person in tournament
    for person in tournament["persons"]:
        whoami = person["firstname"] + " " + person["lastname"]

        # Loop over all players in tournament
        for player in tournament["players"]:
            # If player and person is the same
            if player["person"]["id"] == person["id"]:
                # Loop over characters in game
                for character in game:
                    # If character is player
                    if character["player"]["id"] == player["id"]:
                        x = Baller(
                            name=whoami,
                            value=character["values"]["value"],
                            popularity=character["values"]["popularity"],
                            trend=character["values"]["trend"],
                            position=player["position"]["id"],
                        )

                        # Map positions to separate lists of players
                        if x.position == 6:
                            keepers.append(x)
                        elif x.position == 7:
                            defenses.append(x)
                        elif x.position == 8:
                            midfielders.append(x)
                        elif x.position == 9:
                            forwards.append(x)

    print("# Top 5 keepers by clean sheets")
    for keeper in sorted(keepers, key=lambda x: x.CSPerValue)[:5]:
        print(f"{keeper.CSPerValue}: {keeper}")

    print("\n# Top 5 defense by clean sheets")
    for defense in sorted(defenses, key=lambda x: [x.CSPerValue])[:5]:
        print(f"{defense.CSPerValue}: {defense}")

    print("\n# Top 5 defense by xA")
    for defense in sorted(defenses, key=lambda x: [x.xAPerValue])[:5]:
        print(f"{defense.xAPerValue}: {defense}")

    print("\n# Top 5 defense by xG")
    for defense in sorted(defenses, key=lambda x: [x.xGPerValue])[:5]:
        print(f"{defense.xGPerValue}: {defense}")

    print("\n# Top 5 midfielder by xA")
    for midfielder in sorted(midfielders, key=lambda x: [x.xAPerValue])[:5]:
        print(f"{midfielder.xAPerValue}: {midfielder}")

    print("\n# Top 5 midfielder by xG")
    for midfielder in sorted(midfielders, key=lambda x: [x.xGPerValue])[:5]:
        print(f"{midfielder.xGPerValue}: {midfielder}")

    print("\n# Top 5 forwards by xG")
    for forward in sorted(forwards, key=lambda x: x.xGPerValue)[:5]:
        print(f"{forward.xGPerValue}: {forward}")
