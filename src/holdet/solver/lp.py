import copy
import logging
from typing import Sequence

from pulp import (  # type: ignore
    PULP_CBC_CMD,
    LpInteger,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    value,
)

from holdet.campaign import BaseCandidate


def find_optimal_team(candidates: Sequence[BaseCandidate], budget: int):
    """
    Takes a list of candidates and returns the 11 candidates that maximize the
    combined value within the rules of the game.
    """
    # Add the captain variant of candidate to the list of candidates
    captain_variant = [copy.deepcopy(c) for c in candidates]
    for c in captain_variant:
        c.captain = True
    candidates = list(candidates) + captain_variant

    # Create a linear programming problem
    problem = LpProblem("OptimalTeam", LpMaximize)

    # Create a dictionary to store the variables for each candidate
    variables = {}

    # Create a variable for each candidate. The variable is an integer between 0
    # and 1 where 1 means selected and 0 means not selected.
    for candidate in candidates:
        identifier = f"{candidate.id}_{int(candidate.captain)}"
        variables[(candidate.id, candidate.captain)] = LpVariable(
            identifier, 0, 1, LpInteger
        )

    # Set the objective function to maximize the combined xGrowth of the team
    problem += sum(variables[(c.id, c.captain)] * c.xValue for c in candidates)

    # Add the constraint that no two identical candidates can be selected
    for candidate in candidates:
        problem += (
            variables[(candidate.id, True)] + variables[(candidate.id, False)] <= 1
        )

    # Add the constraint that only 4 candidates from the same team is allowed
    for team in set(candidate.team for candidate in candidates):
        problem += (
            sum(variables[(c.id, c.captain)] for c in candidates if c.team == team) <= 4
        )

    # Add the constraint that the price must be less than or equal to the budget
    problem += sum(variables[(c.id, c.captain)] * c.price for c in candidates) <= budget

    # Add the constraint that there must be exactly 11 players in total
    problem += sum(variables[(c.id, c.captain)] for c in candidates) == 11

    # Add the constraint that there must be exactly 1 captain on the team
    problem += sum(variables[(c.id, c.captain)] for c in candidates if c.captain) == 1

    # Add the constraint that there must be exactly 1 keeper
    problem += sum(variables[(c.id, c.captain)] for c in candidates if c.keeper) == 1

    # Add the constraint that there must be between 3 and 5 defenders
    problem += sum(variables[(c.id, c.captain)] for c in candidates if c.defense) >= 3
    problem += sum(variables[(c.id, c.captain)] for c in candidates if c.defense) <= 5

    # Add the constraint that there must be between 3 and 5 midfielders
    problem += (
        sum(variables[(c.id, c.captain)] for c in candidates if c.midfielder) >= 3
    )
    problem += (
        sum(variables[(c.id, c.captain)] for c in candidates if c.midfielder) <= 5
    )

    # Add the constraint that there must be between 1 and 3 forwards
    problem += sum(variables[(c.id, c.captain)] for c in candidates if c.forward) >= 1
    problem += sum(variables[(c.id, c.captain)] for c in candidates if c.forward) <= 3

    # Solve the problem
    status = problem.solve(PULP_CBC_CMD(msg=0))
    logging.debug(f"LpStatus: {LpStatus[status]}")

    # Iterate through the selected candidates and add them to the optimal team
    optimal_team = []
    for candidate in candidates:
        if value(variables[candidate.id, candidate.captain]) > 0:
            optimal_team.append(candidate)
    return optimal_team
