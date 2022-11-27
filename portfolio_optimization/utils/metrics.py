import numpy as np

__all__ = ['semi_std',
           'max_drawdown',
           'cdar',
           'cvar']


def semi_std(returns: np.ndarray,
             returns_target: float | np.ndarray | None = None) -> float:
    """
    Downside standard deviation with a target return of Rf=0.
    Many implementations remove positive returns then compute the std of the remaining negative returns or replace
    the positive returns by 0 then compute the std. Both are incorrect
    :param returns: expected returns for each asset
    :type returns: np.ndarray of shape(Number of Assets)
    :param returns_target: the return target to distinguish "downside" and "upside"
    :type returns_target: float or np.ndarray of shape(Number of Assets)
    """
    assets_number = returns.shape[0]
    if returns_target is None:
        returns_target = np.mean(returns, axis=0)
    return np.sqrt(np.sum(np.power(np.minimum(0, returns - returns_target), 2)) / (assets_number - 1))


def max_drawdown(prices: np.ndarray) -> float:
    return np.max(1 - prices / np.maximum.accumulate(prices))


def cdar(returns: np.ndarray, beta: float = 0.95) -> float:
    """
    Calculate the Conditional Drawdown at Risk (CDaR) of a return series
    :param returns: returns series
    :param beta: drawdown confidence level (expected drawdown on the worst (1-beta)% days)
    """
    prices = np.cumsum(np.insert(returns, 0, 1))
    k = (1 - beta) * len(returns)
    ik = int(np.ceil(k) - 1)
    # We only need the first k elements so using partition is faster than sort (O(n) vs O(n log(n))
    drawdowns = np.partition(prices - np.maximum.accumulate(prices), ik)
    return -np.sum(drawdowns[:ik]) / k + drawdowns[ik] * (ik / k - 1)


def cvar(returns: np.ndarray, beta: float = 0.95) -> float:
    """
    Calculate the historical Conditional Value at Risk (CVaR) of a return series
    :param returns: returns series
    :param beta: var confidence level (expected var on the worst (1-beta)% days)
    """
    observations_number = len(returns)
    k = int(np.ceil((1 - beta) * observations_number))
    # We only need the first k elements so using partition is faster than sort (O(n) vs O(n log(n))
    return -np.sum(np.partition(returns, k)[:k]) / k
