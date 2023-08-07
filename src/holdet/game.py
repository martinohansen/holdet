#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Type

from holdet.candidate import Candidate
from holdet.data import holdet, sofascore


@dataclass
class Holdet:
    player: holdet.Player
    stats: list[holdet.Statistics]


@dataclass
class Sofascore:
    player: sofascore.Player
    stats: list[sofascore.Statistics]


@dataclass
class Round:
    number: int
    values: holdet.Values
    stats: list[sofascore.Statistics]
    position: holdet.Position

    @property
    def xGrowth(self) -> float:
        growth: float = 0

        # The points vary depending on the position
        points = holdet.Points(self.position)

        for stat in self.stats:
            # Goals and assists
            growth += points.goal * stat.expectedGoals
            growth += points.assist * stat.expectedAssists
            growth += points.shot_on_goal * stat.onTargetScoringAttempt

            # Decisive goals
            growth += points.scoring_victory * stat.decisive_goal_for_win
            growth += points.scoring_draw * stat.decisive_goal_for_draw

            # Fair play
            # growth += points.yellow_card * stat.yellowCard
            # growth += -50000 * self.xRed

            # # Team performance
            growth += points.team_win * stat.win
            growth += points.team_draw * stat.draw
            growth += points.team_loss * stat.loss
            growth += points.team_score * stat.team_goals
            growth += points.opponent_score * stat.team_goals_conceded
            growth += points.away_win if stat.side == "away" and stat.win else 0
            growth += points.home_loss if stat.side == "home" and stat.loss else 0

            # # Special
            growth += points.clean_sheet if stat.clean_sheet else 0
            growth += points.on_field if stat.minutesPlayed != 0 else 0
            growth += points.off_field if stat.minutesPlayed == 0 else 0
            growth += points.hattrick if stat.expectedGoals >= 3 else 0

        return growth

    @property
    def growth(self) -> int:
        return self.values.growth

    @property
    def diff(self) -> float:
        return self.xGrowth - self.growth

    def __lt__(self, other: "Round") -> bool:
        return self.number < other.number

    def __repr__(self) -> str:
        return (
            f"Round(number={self.number}, games={len(self.stats)},"
            f" growth={self.values.growth / 1000:.0f}K,"
            f" xGrowth={self.xGrowth / 1000:.0f}K,"
            f" diff={self.diff / 1000:+.0f}K)"
        )


@dataclass
class BaseCandidate:
    avatar: Holdet  # Avatar is the player from the game
    person: Sofascore  # Person is the player from real world

    captain: bool = False  # Whether the player is the captain or not
    on_team: bool = False  # Whether the player is on the team or not

    @property
    def id(self) -> int:
        return self.avatar.player.id

    @property
    def name(self) -> str:
        return self.avatar.player.person.name

    @property
    def team(self) -> str:
        return self.avatar.player.team.name

    @property
    def team_id(self) -> int:
        return self.avatar.player.team.id

    @property
    def value(self) -> int:
        return self.rounds[-1].values.value

    @property
    def keeper(self) -> bool:
        return self.avatar.player.position == holdet.Position.KEEPER

    @property
    def defense(self) -> bool:
        return self.avatar.player.position == holdet.Position.DEFENSE

    @property
    def midfielder(self) -> bool:
        return self.avatar.player.position == holdet.Position.MIDFIELDER

    @property
    def forward(self) -> bool:
        return self.avatar.player.position == holdet.Position.FORWARD

    @property
    def transfer_fee(self) -> float:
        """Transfer fee is 1% of the value"""
        if self.on_team:
            return 0.0
        else:
            return self.value * 0.01

    @property
    def price(self) -> float:
        return self.value + self.transfer_fee

    @property
    def emoji(self) -> str:
        if self.captain:
            return "👑"
        return "⚽️"

    @property
    def xValue(self) -> float:
        raise NotImplementedError("xValue not implemented")

    @property
    def similarity(self) -> float:
        return get_similarity(self.person.player, self.avatar.player)

    @property
    def rounds(self) -> list[Round]:
        """
        Zip the statistics from the avatar and the person together. Every stat
        that is within the same round is zipped together. Only include rounds
        that have ended.
        """

        def zip(stat: holdet.Statistics) -> Round:
            return Round(
                number=stat.round.number,
                values=stat.values,
                stats=[
                    s
                    for s in self.person.stats
                    if stat.round.start <= s.game.start <= stat.round.end
                ],
                position=stat.player.position,
            )

        return [
            zip(stat)
            for stat in self.avatar.stats
            if stat.round.end < datetime.now(tz=timezone.utc)
        ]

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.value < other.value

    def __hash__(self):
        return hash(self.avatar.player.person)

    def __repr__(self) -> str:
        return (
            f"{self.emoji} {self.name} ({self.team}), value={self.value / 1000000:.1f}M"
        )


class Game:
    def __init__(self, candidate: Type[BaseCandidate]) -> None:
        self.candidates: list[Candidate] = []
        self.holdet_client = holdet.Client()
        self.sofascore_client = sofascore.Client()

        logging.info("Fetching data from Holdet...")
        holdet_candidates = self._get_holdet(
            self.holdet_client.tournament(holdet.PRIMER_LEAGUE),
            [
                holdet.Game(holdet.PRIMER_LEAGUE_FALL_2022),
                holdet.Game(holdet.PRIMER_LEAGUE_SPRING_2023),
                holdet.Game(holdet.PRIMER_LEAGUE_FALL_2023),
            ],
        )

        logging.info("Fetching data from Sofascore...")
        tournament = sofascore.Tournament(sofascore.PRIMER_LEAGUE)
        for s in self._get_sofascore(
            tournament,
            seasons=[
                tournament.season(sofascore.PRIMER_LEAGUE_2022_2023),
                tournament.season(sofascore.PRIMER_LEAGUE_2023_2024),
            ],
        ):
            # Find the closest match in Holdet for the current Sofascore player and
            # remove it from the list afterwards.
            h = find_closest_match(s, holdet_candidates)
            holdet_candidates.remove(h)
            self.candidates.append(candidate(h, s))
        logging.info(f"Collected data from {len(self.candidates)} candidates")

    def _get_sofascore(
        self,
        tournament: sofascore.Tournament,
        seasons: list[sofascore.Season],
    ) -> list[Sofascore]:
        players: list[Sofascore] = []
        for season in seasons:
            logging.info(f"Fetching data for {season}...")
            # Get the current round for the season and iterate over all the
            # rounds so far to gather the player stats.
            current_round = self.sofascore_client.current_round(tournament, season)
            for round in range(1, current_round + 1):
                for game in self.sofascore_client.games(season, round):
                    lineup = self.sofascore_client.lineup(game).all
                    for player, stats in lineup:
                        found = False
                        for p in players:
                            if player == p.player:
                                p.stats.append(stats)
                                found = True
                                break
                        if not found:
                            players.append(Sofascore(player=player, stats=[stats]))

        return players

    def _get_holdet(
        self,
        tournament: holdet.Tournament,
        games: list[holdet.Game],
    ) -> list[Holdet]:
        # Use a dict for lookups to improve speed
        players_dict: dict[int, Holdet] = {}
        for game in games:
            logging.info(f"Fetching data for {game}...")
            for round in self.holdet_client.rounds(game):
                stats = self.holdet_client.statistics(tournament, game, round)
                for stat in stats:
                    player_id = stat.player.id
                    if player_id in players_dict:
                        players_dict[player_id].stats.append(stat)
                    else:
                        players_dict[player_id] = Holdet(
                            player=stat.player, stats=[stat]
                        )

        return list(players_dict.values())


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
