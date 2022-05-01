from typing import Union, Optional
import numpy as np

__all__ = ['downside_std',
           'max_drawdown',
           'max_drawdown_slow',
           'cdar']


def downside_std(returns: np.ndarray,
                 returns_target: Optional[Union[float, np.ndarray]] = None) -> float:
    """
    Downside standard deviation with a target return of Rf=0.
    Many implementations remove positive returns then compute the std of the remaining negative returns or replace
    the positive returns by 0 then compute the std. Both are incorrect.

    :param returns: expected returns for each asset.
    :type returns: np.ndarray of shape(Number of Assets)

    :param returns_target: the return target to distinguish "downside" and "upside".
    :type returns_target: float or np.ndarray of shape(Number of Assets)
    """
    assets_number = returns.shape[0]
    if returns_target is None:
        returns_target = np.mean(returns, axis=0)
    return np.sqrt(np.sum(np.power(np.minimum(0, returns - returns_target), 2)) / (assets_number - 1))


def max_drawdown(prices: np.array) -> float:
    return np.max(1 - prices / np.maximum.accumulate(prices))


def max_drawdown_slow(prices: np.array) -> float:
    max_dd = 0
    max_seen = prices[0]
    for price in prices:
        max_seen = max(max_seen, price)
        max_dd = max(max_dd, 1 - price / max_seen)
    return max_dd


def cdar(prices, beta: float = 0.95):
    """
    Calculate the Conditional Drawdown at Risk (CDaR) of a price series.
    :param prices: prices series.
    :param beta: drawdown confidence level (expected drawdown on the worst (1-beta)% days)
    """
    observations_number = len(prices)
    p = int(np.ceil((1 - beta) * observations_number))

    drawdowns = np.sort(prices / np.maximum.accumulate(prices) - 1)
    cdar = -np.sum(drawdowns[:p]) / p
    return cdar
