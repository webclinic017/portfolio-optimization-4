import numpy as np

__all__ = ['semi_variance',
           'kurtosis',
           'semi_kurtosis',
           'max_drawdown',
           'cdar',
           'cvar',
           'mad']


def semi_variance(returns: np.ndarray,
                  min_acceptable_return: float | None = None) -> float:
    r"""
    Calculate the Semi Variance (Second Lower Partial Moment).
    The Semi Variance is the variance of the returns below a minimum acceptable return.

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


def kurtosis(returns: np.ndarray) -> float:
    r"""
    Calculate the Kurtosis (Fourth Central Moment).
    The Kurtosis is a measure of the heaviness of the tail of the distribution.
    Higher kurtosis corresponds to greater extremity of deviations (fat tails).

    Parameters
    ----------
    returns : 1d-array
              The returns array
    Returns
    -------
    value : float
            The Kurtosis
    """

    return np.sum(np.power(returns - np.mean(returns, axis=0), 4)) / len(returns)


def semi_kurtosis(returns: np.ndarray,
                  min_acceptable_return: float | None = None) -> float:
    r"""
    Calculate the Semi Kurtosis (Fourth Lower Partial Moment).
    The Semi Kurtosis is a measure of the heaviness of the downside tail of the returns below a minimum acceptable
    return.
    Higher Semi Kurtosis corresponds to greater extremity of downside deviations (downside fat tail).

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
            The Semi Kurtosis
    """
    if min_acceptable_return is None:
        min_acceptable_return = np.mean(returns, axis=0)
    return np.sum(np.power(np.minimum(0, returns - min_acceptable_return), 4)) / len(returns)


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
