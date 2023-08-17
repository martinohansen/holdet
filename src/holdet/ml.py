from holdet.campaign import BaseCandidate, Round
from holdet.data import sofascore


class CandidateML(BaseCandidate):
    """
    Candidate with support for generating datasets for training machine learning
    models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def features(self, stat: sofascore.Statistics) -> dict[str, float | int]:
        """
        Return a dictionary with all the features that can be used to train a
        model.
        """

        # TODO: Add data from betting sites
        return {
            "substitute": int(stat.substitute),
            "assists": stat.assists,
            "expectedAssists": stat.expectedAssists,
            "expectedGoals": stat.expectedGoals,
            "goals": stat.goals,
            "goalsPrevented": stat.goalsPrevented,
            "minutesPlayed": stat.minutesPlayed,
            "onTargetScoringAttempt": stat.onTargetScoringAttempt,
            "savedShotsFromInsideTheBox": stat.savedShotsFromInsideTheBox,
            "saves": stat.saves,
            "team_goals": stat.team_goals,
            "team_goals_conceded": stat.team_goals_conceded,
            "win": int(stat.win),
            "loss": int(stat.loss),
            "draw": int(stat.draw),
            "clean_sheet": int(stat.clean_sheet),
            "decisive_goal_for_draw": int(stat.decisive_goal_for_draw),
            "decisive_goal_for_win": int(stat.decisive_goal_for_win),
        }

    def aggregate_stats(self, round: Round) -> dict:
        """
        Aggregate all the stats from a round into one. This is needed for rounds
        with multiple games in them. The features are summed together.

        Its not ideal for stuff like opponent or side which will simply be added
        together, but thats we can do for now.
        """
        round_stats: dict[str, int | float] = {}

        for stat in round.stats:
            for key, value in self.features(stat).items():
                round_stats[key] = round_stats.get(key, 0) + value
        return round_stats

    def round_features(self, round: Round) -> dict:
        """
        Return a dictionary with all the features that can be used to train a
        model.
        """

        features = {
            "id": self.id,
            "round": round.number,
            "position": round.position.value,
            "team": self.team_id,
            "games": len(round.schedules),
            "growth": round.growth,
        }

        # Assume that any player have at max 2 scheduled games in a round.
        # Append the opponent and side for each game or None to the features.
        team = self.avatar.player.team
        for idx in range(2):
            if idx < len(round.schedules):
                schedule = round.schedules[idx]
                features[f"opponent_{idx}"] = schedule.opponent(team).id
                features[f"side_{idx}"] = schedule.side(team).value

        # Aggregate all the features from the round
        stats = self.aggregate_stats(round)
        for key, value in stats.items():
            features[key] = value

        return features

    def dataset(self) -> list[dict]:
        """
        Dataset with features for each round
        """

        data = []
        for round in self.rounds:
            data.append(self.round_features(round))

        return data

    def future_dataset(self, n: int) -> list[dict]:
        """
        Dataset with mostly empty features for future rounds
        """
        data = []
        for round in self.next_rounds(n):
            data.append(self.round_features(round))

        return data
