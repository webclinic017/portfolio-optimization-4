import datetime as dt
import numpy as np
from numba import jit

__all__ = ['dominate',
           'non_denominated_sort',
           'dominate_slow',
           'prices_rebased',
           'portfolio_returns',
           'rand_weights',
           'rand_weights_dirichlet',
           'walk_forward']


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

# non_denominated_sort(n, fitnesses, False)

@jit(nopython=True)
def non_denominated_sort(n: int, fitnesses: np.ndarray, first_front_only: bool):
    fronts = [0]
    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.array([0 for _ in range(n)])

    # for each portfolio a list of all portfolios that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.array([0 for _ in range(n)])

    current_front = []

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


def dominate_slow(fitness_1: np.array, fitness_2: np.array) -> bool:
    return np.all(fitness_1 >= fitness_2) and np.any(fitness_1 > fitness_2)


def prices_rebased(returns: np.array) -> np.array:
    p = [1 + returns[0]]
    for i in range(1, len(returns)):
        p.append(p[-1] * (1 + returns[i]))
    return np.array(p)


def portfolio_returns(asset_returns: np.ndarray, weights: np.array) -> np.array:
    n, m = asset_returns.shape
    returns = np.zeros(m)
    for i in range(n):
        returns += asset_returns[i] * weights[i]
    return returns


def rand_weights(n: int, zeros: int = 0) -> np.array:
    """
    Produces n random weights that sum to 1 (non-uniform distribution over the simplex)
    """
    k = np.random.rand(n)
    if zeros > 0:
        zeros_idx = np.random.choice(n, zeros, replace=False)
        k[zeros_idx] = 0
    return k / sum(k)


def rand_weights_dirichlet(n: int) -> np.array:
    """
    Produces n random weights that sum to 1 with uniform distribution over the simplex
    """
    return np.random.dirichlet(np.ones(n))


def walk_forward(start_date: dt.date,
                 end_date: dt.date,
                 train_duration: int,
                 test_duration: int,
                 full_period: bool = True) -> tuple[tuple[dt.date, dt.date], tuple[dt.date, dt.date]]:
    """
    Yield train and test periods in a walk forward manner.

    The proprieties are:
        * The test periods are not overlapping
        * The test periods are directly following the train periods

    Example:
        0 --> Train
        1 -- Test

        00000111------
        ---00000111---
        ------00000111
    """
    train_start = start_date
    while True:
        train_end = train_start + dt.timedelta(days=train_duration)
        test_start = train_end
        test_end = test_start + dt.timedelta(days=test_duration)
        if test_end > end_date:
            if full_period or test_start >= end_date:
                return
            else:
                test_end = end_date
        yield (train_start, train_end), (test_start, test_end)
        train_start = train_start + dt.timedelta(days=test_duration)
