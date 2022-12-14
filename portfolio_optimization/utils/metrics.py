import numpy as np
from scipy.optimize import minimize, Bounds

__all__ = ['semi_variance',
           'kurtosis',
           'semi_kurtosis',
           'max_drawdown',
           'cdar',
           'cvar',
           'mad',
           'value_at_risk',
           'worst_return',
           'first_lower_partial_moment',
           'entropic_risk_measure',
           'evar']


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


def first_lower_partial_moment(returns: np.ndarray,
                               min_acceptable_return: float | None = None) -> float:
    r"""
    Calculate the First Lower Partial Moment.
    The First Lower Partial Moment is the mean of the returns below a minimum acceptable return.

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
    return -np.sum(np.minimum(0, returns - min_acceptable_return)) / len(returns)


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


def value_at_risk(returns: np.ndarray, beta: float = 0.95) -> float:
    r"""
    Calculate the VaR (Value at Risk).

    Parameters
    ----------
    returns : 1d-array
              The returns array

    beta: float, default = 0.95
          The VaR confidence level (return on the worst (1-beta)% observation)

    Returns
    -------
    value : float
            The VaR
    """
    k = (1 - beta) * len(returns)
    ik = int(np.ceil(k) - 1)
    # We only need the first k elements so using partition is faster than sort (O(n) vs O(n log(n))
    ret = np.partition(returns, ik)
    return -ret[ik]


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


def entropic_risk_measure(returns: np.ndarray,
                          theta: float = 1,
                          beta: float = 0.95) -> float:
    r"""
    Calculate the Entropic Risk Measure.
    The Entropic Risk Measure is a risk measure which depends on the risk aversion of the user through
    the exponential utility function.

    Parameters
    ----------
    returns : 1d-array
              The returns array

    theta: float, default 1
          The risk aversion

    beta: float, default 0.95
          The confidence level

    Returns
    -------
    value : float
            The VaR
    """
    return theta * (np.log(np.mean(np.exp(-returns / theta), axis=0)) + np.log(1 / (1 - beta)))


def evar(returns: np.ndarray, beta: float = 0.95) -> tuple[float, float]:
    r"""
    Calculate the EVaR (Entropic Value at Risk) and its associated risk aversion.
    The EVaR is a coherent risk measure which is an upper bound for the VaR and the CVaR,
    obtained from the Chernoff inequality. The EVaR can be represented by using the concept of relative entropy.

    Parameters
    ----------
    returns : 1d-array
              The returns array

    beta: float, default 0.95
          The EVaR confidence level

    Returns
    -------
    value : tuple(float, float)
            The EVaR and its associated risk aversion
    """

    def func(x: float) -> float:
        return entropic_risk_measure(returns=returns, theta=x, beta=beta)

    result = minimize(func,
                      x0=np.array([1]),
                      method='SLSQP',
                      bounds=Bounds([1e-15], [np.inf]),
                      tol=1e-10)
    return result.fun, result.x[0]


def worst_return(returns: np.ndarray) -> float:
    r"""
    Calculate the Worst Return (Worst Realization).

    Parameters
    ----------
    returns : 1d-array
              The returns array

    Returns
    -------
    value : float
            The Worst Return
    """
    return -min(returns)


def max_drawdown(returns: np.ndarray, compounded: bool = False) -> float:
    r"""
    Calculate the Maximum Drawdown.

    Parameters
    ----------
    returns : 1d-array
              The returns array

    compounded: bool, default False
               If True, we use compounded cumulative returns otherwise we use uncompounded cumulative returns

    Returns
    -------
    value : float
            The Maximum Drawdown
    """
    if compounded:
        prices = np.cumprod(1 + np.insert(returns, 0, 0))
        mdd = np.max(1 - prices / np.maximum.accumulate(prices))
    else:
        prices = np.cumsum(np.insert(returns, 0, 1))
        mdd = np.max(np.maximum.accumulate(prices) - prices)
    return mdd


def cdar(returns: np.ndarray,
         beta: float = 0.95,
         compounded: bool = False) -> float:
    r"""
   Calculate the historical CDaR (Conditional Drawdown at Risk).

   Parameters
   ----------
   returns : 1d-array
             The returns array

   beta: float, default = 0.95
         The drawdown confidence level (expected drawdown on the worst (1-beta)% observations)

   compounded: bool, default False
               If True, we use compounded cumulative returns otherwise we use uncompounded cumulative returns

   Returns
   -------
   value : float
           The CVaR
   """
    k = (1 - beta) * len(returns)
    ik = int(np.ceil(k) - 1)

    if compounded:
        prices = np.cumprod(1 + np.insert(returns, 0, 0))
        drawdowns = np.partition(prices / np.maximum.accumulate(prices) - 1, ik)

    else:
        prices = np.cumsum(np.insert(returns, 0, 1))
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
