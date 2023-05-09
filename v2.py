#!/usr/bin/env python3


from dataclasses import dataclass

from holdet import holdet
from sofascore import sofascore


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
class Player:
    name: str
    team: str

    stats: list[Stats]


if __name__ == "__main__":
    pl = sofascore.Tournament(
        17,  # Premier League
        41886,  # 2022/2023
    )
    c = sofascore.Client()
    games = c.games(pl, 33)
    for game in c.games(pl, 33):
        for player in c.lineup(game).all:
            print(f"{player.name}: {c.statistics(game, player)}")

    h = holdet.Client()
    characters = h.characters(tournament_id=422, game_id=644, game_round=3)
    for character in characters:
        print(character)
