import numpy as np

__all__ = ['dominate',
           'dominate_slow',
           'max_drawdown',
           'max_drawdown_slow',
           'prices_rebased',
           'portfolio_returns',
           'rand_weights']


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


def max_drawdown(prices: np.array) -> float:
    return np.max(1 - prices / np.maximum.accumulate(prices))


def max_drawdown_slow(prices: np.array) -> float:
    max_dd = 0
    max_seen = prices[0]
    for price in prices:
        max_seen = max(max_seen, price)
        max_dd = max(max_dd, 1 - price / max_seen)
    return max_dd


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


def rand_weights(n: int) -> np.array:
    """
    Produces n random weights that sum to 1
    """
    k = np.random.rand(n)
    return k / sum(k)
