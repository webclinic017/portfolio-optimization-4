import datetime as dt
from typing import Union, Optional
import numpy as np

from portfolio_optimization.meta import *

__all__ = ['dominate',
           'dominate_slow',
           'prices_rebased',
           'portfolio_returns',
           'rand_weights',
           'rand_weights_dirichlet',
           'get_lower_and_upper_bounds',
           'get_investment_target',
           'walk_forward']


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


def get_lower_and_upper_bounds(
        weight_bounds: Union[tuple[np.ndarray, np.ndarray], tuple[Optional[float], Optional[float]]],
        assets_number: int) -> tuple[np.ndarray, np.ndarray]:
    # Upper and lower bounds
    lower_bounds, upper_bounds = weight_bounds
    if lower_bounds is None:
        lower_bounds = -1
    if upper_bounds is None:
        upper_bounds = 1
    if np.isscalar(lower_bounds):
        lower_bounds = np.array([lower_bounds] * assets_number)
    if np.isscalar(upper_bounds):
        upper_bounds = np.array([upper_bounds] * assets_number)

    return lower_bounds, upper_bounds


def get_investment_target(investment_type: InvestmentType) -> Optional[int]:
    # Upper and lower bounds

    # Sum of weights
    if investment_type == InvestmentType.FULLY_INVESTED:
        return 1
    elif investment_type == InvestmentType.MARKET_NEUTRAL:
        return 0


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
