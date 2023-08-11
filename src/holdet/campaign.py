#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Type

from holdet import util
from holdet.data import holdet, sofascore


@dataclass
class Avatar:
    player: holdet.Player
    stats: list[holdet.Statistics]


@dataclass
class Person:
    player: sofascore.Player
    stats: list[sofascore.Statistics]


@dataclass
class Round:
    season: int
    number: int
    values: holdet.Values
    stats: list[sofascore.Statistics]
    position: holdet.Position

    start: datetime
    end: datetime

    @property
    def xGrowth(self) -> float:
        """
        Estimated round growth based on "expected" stats, e.g. xG, xA, etc.
        """
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

    @property
    def matches(self) -> int:
        return len(self.stats)

    def __lt__(self, other: "Round") -> bool:
        return self.number < other.number

    def __repr__(self) -> str:
        return (
            f"Round(season={self.season}, number={self.number},"
            f" matches={self.matches}, growth={self.values.growth / 1000:.0f}K,"
            f" xGrowth={self.xGrowth / 1000:.0f}K, diff={self.diff / 1000:+.0f}K)"
        )

    @classmethod
    def from_stat(cls, stat: holdet.Statistics) -> "Round":
        """
        Returns a Round object from a holdet.Statistics. The sofascore stats
        have to be added after the fact.
        """
        return Round(
            season=stat.round.game.id,
            number=stat.round.number,
            values=stat.values,
            stats=[],
            position=stat.player.position,
            start=stat.round.start,
            end=stat.round.end,
        )


@dataclass
class BaseCandidate:
    avatar: Avatar  # Avatar is the player from the game
    person: Person  # Person is the player from real world

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
        try:
            return self.rounds[-1].values.value
        except IndexError:
            return 0

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
            r = Round.from_stat(stat)
            # Append stats from sofascore if they are within the period of the
            # round.
            r.stats = [
                s
                for s in self.person.stats
                if stat.round.start <= s.game.start <= stat.round.end
            ]
            return r

        return [
            zip(stat)
            for stat in self.avatar.stats
            if stat.round.end < datetime.now(tz=timezone.utc)
        ]

    def next_rounds(self, n: int) -> list[Round]:
        """
        Return until n number of next rounds from the game.
        """
        n_rounds: list[Round] = []
        now = datetime.now(tz=timezone.utc)
        for stat in self.avatar.stats:
            if now <= stat.round.start:
                n_rounds.append(Round.from_stat(stat))
                if len(n_rounds) == n:
                    return n_rounds

        # Return max number of n_rounds we could find
        return n_rounds

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


def _get_avatars(
    client: holdet.Client,
    tournaments: list[holdet.Tournament],
    games: list[holdet.Game],
) -> list[Avatar]:
    # Use a dict for lookups to improve speed
    persons: dict[int, Avatar] = {}
    for game in games:
        logging.info(f"Fetching data for {game}...")
        for round in client.rounds(game):
            for tournament in tournaments:
                stats = client.statistics(tournament, game, round)
                for stat in stats:
                    person_id = stat.player.person.id
                    if person_id in persons:
                        persons[person_id].stats.append(stat)
                    else:
                        persons[person_id] = Avatar(player=stat.player, stats=[stat])

    return list(persons.values())


def _get_persons(
    client: sofascore.Client,
    tournament: sofascore.Tournament,
    seasons: list[sofascore.Season],
) -> list[Person]:
    players: list[Person] = []
    for season in seasons:
        logging.info(f"Fetching data for {season}...")
        # Get the current round for the season and iterate over all the
        # rounds so far to gather the player stats.
        current_round = client.current_round(tournament, season)
        for round in range(1, current_round + 1):
            for game in client.games(season, round):
                lineup = client.lineup(game).all
                for player, stats in lineup:
                    found = False
                    for p in players:
                        if player == p.player:
                            p.stats.append(stats)
                            found = True
                            break
                    if not found:
                        players.append(Person(player=player, stats=[stats]))

    return players


@dataclass
class Game:
    candidates: list[BaseCandidate]

    @classmethod
    def new(cls, candidate: Type[BaseCandidate]) -> "Game":
        # TODO: Accept the tournament, game and season values as arguments
        # in some clever way to make the user able to chose which and how
        # much data to use. Even better if we can make it data source
        # agnostic.
        holdet_client = holdet.Client()
        sofascore_client = sofascore.Client()

        candidates: list[BaseCandidate] = []

        logging.info("Fetching data from Holdet...")
        avatars = _get_avatars(
            holdet_client,
            [
                holdet_client.tournament(holdet.PRIMER_LEAGUE_2023_2024),
                holdet_client.tournament(holdet.PRIMER_LEAGUE_2022_2023),
            ],
            [
                # Putting the newest season first to make sure that the newest
                # data is used to create the candidates. E.g. if a player was
                # marked as a midfielder in the newest season but a forward in
                # the previous season we want to use the newest data.
                holdet_client.game(holdet.PRIMER_LEAGUE_FALL_2023),
                holdet_client.game(holdet.PRIMER_LEAGUE_SPRING_2023),
                holdet_client.game(holdet.PRIMER_LEAGUE_FALL_2022),
            ],
        )

        logging.info("Fetching data from Sofascore...")
        tournament = sofascore.Tournament(sofascore.PRIMER_LEAGUE)
        for p in _get_persons(
            sofascore_client,
            tournament,
            seasons=[
                tournament.season(sofascore.PRIMER_LEAGUE_2022_2023),
                tournament.season(sofascore.PRIMER_LEAGUE_2023_2024),
            ],
        ):
            # Find the closest match, if any, in Holdet for the current
            # Sofascore player and remove it from the list afterwards.
            a = find_closest_match(p, avatars)
            if a is not None:
                avatars.remove(a)
                candidates.append(candidate(a, p))
            else:
                logging.debug(f"Found no avatar in game for: {p}")

        logging.info(f"Collected data from {len(candidates)} candidates")
        return Game(candidates=candidates)

    def find_candidate(self, name: str) -> None | BaseCandidate:
        """Find candidate by name"""
        choices = [c.name for c in self.candidates]
        closest_match = util.get_closest_match(name, choices)
        if closest_match:
            for c in self.candidates:
                if c.name == closest_match:
                    return c
        return None


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


def get_similarity(p: sofascore.Player, a: holdet.Player) -> float:
    team_similarity = jaccard_similarity(p.team.name, a.team.name)
    name_similarity = jaccard_similarity(p.name, a.person.name)

    # Calculate the overall similarity as the average of the team and
    # name similarities
    overall_similarity = (team_similarity + name_similarity) / 2

    return overall_similarity


def find_closest_match(s: Person, avatars: list[Avatar]) -> Avatar | None:
    max_similarity = 0.0
    closest_match_idx = None

    for a in avatars:
        # Read the similarity between players
        similarity = get_similarity(s.player, a.player)

        # If the similarity is greater than the maximum similarity update the
        # closest match
        if similarity > max_similarity:
            max_similarity = similarity
            closest_match_idx = avatars.index(a)

    # If no match was found it might mean that the player is not in the game.
    # This will happen for the players that was on relegated teams for example.
    if closest_match_idx is None:
        return None
    return avatars[closest_match_idx]
