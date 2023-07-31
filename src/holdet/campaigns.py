from dataclasses import dataclass

from holdet.data import holdet, sofascore


@dataclass
class League:
    holdet: int
    sofascore: int


@dataclass
class Season:
    holdet: int
    sofascore: int


@dataclass
class Campaign:
    name: str
    League: League
    Season: Season

    def __str__(self) -> str:
        return self.name

    @property
    def holdet(self) -> holdet.Game:
        return holdet.Game(self.League.holdet, self.Season.holdet)

    @property
    def sofascore(self) -> sofascore.Tournament:
        return sofascore.Tournament(self.League.sofascore, self.Season.sofascore)


PRIMER_LEAGUE_2023 = Campaign(
    "Primer League Spring 2023", League(644, 17), Season(422, 41886)
)
