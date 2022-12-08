import numpy as np

__all__ = ['semivariance',
           'max_drawdown',
           'cdar',
           'cvar',
           'mad']


def semivariance(returns: np.ndarray,
                 min_acceptable_return: float | None = None) -> float:
    r"""
    Calculate the Semi Variance.
    The Semi Variance is the variance of the returns below the min_acceptable_return.

    Many implementations remove positive returns then compute the std of the remaining negative returns or replace
    the positive returns by 0 then compute the std. Both are incorrect.

    Parameters
    ----------
    returns : 1d-array
              The returns array

    min_acceptable_return: float, optional
                           The minimum acceptable return which is the return target to distinguish "downside" and
                           "upside" returns. If not provided, the returns mean will be used.

    Returns
    -------
    value : float
            The Semi Variance
    """
    if min_acceptable_return is None:
        min_acceptable_return = np.mean(returns, axis=0)
    return np.sum(np.power(np.minimum(0, returns - min_acceptable_return), 2)) / (len(returns) - 1)


def cvar(returns: np.ndarray, beta: float = 0.95) -> float:
    r"""
    Calculate the historical CVaR (Conditional Value at Risk).

    Parameters
    ----------
    returns : 1d-array
              The returns array

    beta: float, default = 0.95
          The VaR (Value At Risk) confidence level (expected VaR on the worst (1-beta)% observations)

    Returns
    -------
    value : float
            The CVaR
    """
    k = (1 - beta) * len(returns)
    ik = int(np.ceil(k) - 1)
    # We only need the first k elements so using partition is faster than sort (O(n) vs O(n log(n))
    ret = np.partition(returns, ik)
    return -np.sum(ret[:ik]) / k + ret[ik] * (ik / k - 1)


def max_drawdown(prices: np.ndarray) -> float:
    return np.max(1 - prices / np.maximum.accumulate(prices))


def cdar(returns: np.ndarray, beta: float = 0.95) -> float:
    r"""
   Calculate the historical CDaR (Conditional Drawdown at Risk).

   Parameters
   ----------
   returns : 1d-array
             The returns array

   beta: float, default = 0.95
         The drawdown confidence level (expected drawdown on the worst (1-beta)% observations)

   Returns
   -------
   value : float
           The CVaR
   """
    prices = np.cumsum(np.insert(returns, 0, 1))
    k = (1 - beta) * len(returns)
    ik = int(np.ceil(k) - 1)
    # We only need the first k elements so using partition is faster than sort (O(n) vs O(n log(n))
    drawdowns = np.partition(prices - np.maximum.accumulate(prices), ik)
    return -np.sum(drawdowns[:ik]) / k + drawdowns[ik] * (ik / k - 1)


def mad(returns: np.ndarray) -> float:
    r"""
    Calculate the MAD (Mean Absolute Deviation).

    Parameters
    ----------
    returns : 1d-array
             The returns array

    Returns
   -------
   value : float
           The MAD
    """
    return float(np.mean(np.abs(returns - np.mean(returns, axis=0)), axis=0))
