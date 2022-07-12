import numpy as np

from portfolio_optimization.portfolio import *

__all__ = ['fast_nondominated_sorting']


def fast_nondominated_sorting(portfolios: list[Portfolio], first_front_only: bool = False) -> list[list[int]]:
    """ Fast non-dominated sorting.
    Sort the portfolios into different non-domination levels.
    Complexity O(MN^2) where M is the number of objectives and N the number of portfolios.
    :param portfolios: A list of portfolios to select from.
    :param first_front_only: If :obj:`True` sort only the first front and exit.
    :returns: A list of Pareto fronts (lists), the first list includes non-dominated portfolios.
    """
    fronts = []
    n = len(portfolios)

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=int)

    # for each portfolio a list of all portfolios that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):
        for j in range(i + 1, n):
            if portfolios[i].dominates(portfolios[j]):
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif portfolios[j].dominates(portfolios[i]):
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    if first_front_only:
        return fronts

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each portfolio in the current front
        for i in current_front:

            # all solutions that are dominated by this portfolio
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts
