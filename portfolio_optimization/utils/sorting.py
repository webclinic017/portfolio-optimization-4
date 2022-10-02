import numpy as np
from numba import jit

__all__ = ['dominate',
           'non_denominated_sort',
           'dominate_slow']


@jit(nopython=True)
def dominate(fitness_1: np.array, fitness_2: np.array) -> bool:
    """
    Return true if each objective of the current portfolio's fitness is not strictly worse than
    the corresponding objective of the other portfolio's fitness and at least one objective is
    strictly better.
    """
    not_equal = False
    for self_value, other_value in zip(fitness_1, fitness_2):
        if self_value > other_value:
            not_equal = True
        elif self_value < other_value:
            return False
    return not_equal


def dominate_slow(fitness_1: np.array, fitness_2: np.array) -> bool:
    return np.all(fitness_1 >= fitness_2) and np.any(fitness_1 > fitness_2)


@jit(nopython=True)
def non_denominated_sort(n: int, fitnesses: np.ndarray, first_front_only: bool) -> list[list[int]]:
    fronts = [[x for x in range(0)] for _ in range(0)]  # "trick” to instruct the numba typing mechanism
    if n == 0:
        return fronts

        # final rank that will be returned
    n_ranked = 0
    ranked = np.array([0 for _ in range(n)])

    # for each portfolio a list of all portfolios that are dominated by this one
    is_dominating = [[x for x in range(0)] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = [0 for _ in range(n)]

    current_front = [x for x in range(0)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominate(fitnesses[i], fitnesses[j]):
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif dominate(fitnesses[j], fitnesses[i]):
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
