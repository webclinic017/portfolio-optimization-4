import numpy as np
from scipy.optimize import minimize, Bounds

__all__ = ['mean',
           'get_cumulative_returns',
           'get_drawdowns',
           'variance',
           'semi_variance',
           'std',
           'semi_std',
           'kurtosis',
           'semi_kurtosis',
           'cvar',
           'mad',
           'value_at_risk',
           'worst_realization',
           'first_lower_partial_moment',
           'entropic_risk_measure',
           'evar',
           'dar',
           'cdar',
           'max_drawdown',
           'avg_drawdown',
           'edar',
           'ulcer_index',
           'gini_mean_difference']


def mean(returns: np.ndarray, annualized_factor: float = 1) -> float:
    return returns.mean() * annualized_factor


def mad(returns: np.ndarray, annualized_factor: float = 1) -> float:
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
    return float(np.mean(np.abs(returns - np.mean(returns, axis=0)), axis=0)) * annualized_factor


def first_lower_partial_moment(returns: np.ndarray,
                               min_acceptable_return: float | None = None,
                               annualized_factor: float = 1) -> float:
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
    return -np.sum(np.minimum(0, returns - min_acceptable_return)) / len(returns) * annualized_factor


def variance(returns: np.ndarray,
             annualized_factor: float = 1) -> float:
    return returns.var(ddof=1) * annualized_factor


def semi_variance(returns: np.ndarray,
                  min_acceptable_return: float | None = None,
                  annualized_factor: float = 1) -> float:
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


def std(returns: np.ndarray,
        annualized_factor: float = 1) -> float:
    return np.sqrt(variance(returns=returns, annualized_factor=annualized_factor))


def semi_std(returns: np.ndarray,
             min_acceptable_return: float | None = None,
             annualized_factor: float = 1) -> float:
    return np.sqrt(semi_variance(returns=returns,
                                 min_acceptable_return=min_acceptable_return,
                                 annualized_factor=annualized_factor))


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
    Calculate the historical VaR (Value at Risk).
    The VaR is the maximum loss at a given confidence level (beta).

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
    ik = max(0, int(np.ceil(k) - 1))
    # We only need the first k elements so using partition is faster than sort (O(n) vs O(n log(n))
    ret = np.partition(returns, ik)
    return -ret[ik]


def cvar(returns: np.ndarray, beta: float = 0.95) -> float:
    r"""
    Calculate the historical CVaR (Conditional Value at Risk).
    The CVaR (or Tail VaR) represents the mean shortfall at a specified confidence level.

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
    ik = max(0, int(np.ceil(k) - 1))
    # We only need the first k elements so using partition is faster than sort (O(n) vs O(n log(n))
    ret = np.partition(returns, ik)
    return -np.sum(ret[:ik]) / k + ret[ik] * (ik / k - 1)


def entropic_risk_measure(returns: np.ndarray,
                          theta: float = 1,
                          beta: float = 0.95) -> float:
    r"""
    Calculate the Entropic Risk Measure.
    The Entropic Risk Measure is a risk measure which depends on the risk aversion defined by the onvestor (theat)
     througt the exponential utility function at a given confidence level (beta).

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


def worst_realization(returns: np.ndarray) -> float:
    r"""
    Calculate the Worst Realization (Worst Return).

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


def get_cumulative_returns(returns: np.ndarray,
                           compounded: bool = False) -> np.ndarray:
    if compounded:
        cumulative_returns = np.delete(np.cumprod(1 + np.insert(returns, 0, 0)), 0)
    else:
        cumulative_returns = np.cumsum(returns)
    return cumulative_returns


def get_drawdowns(returns: np.ndarray,
                  compounded: bool = False) -> np.ndarray:
    cumulative_returns = get_cumulative_returns(returns=returns, compounded=compounded)
    if compounded:
        drawdowns = cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1
    else:
        drawdowns = cumulative_returns - np.maximum.accumulate(cumulative_returns)
    return drawdowns


def dar(drawdowns: np.ndarray,
        beta: float = 0.95) -> float:
    r"""
    Calculate the Drawdown at Risk (DaR).
    The DaR is the maximum drawdown at a given confidence level (beta)

    Parameters
    ----------
    drawdowns: array, optional
               The drawdowns array

    beta: float, default = 0.95
     The drawdown confidence level (expected drawdown on the worst (1-beta)% observations)

    Returns
    -------
    value : float
       The DaR
    """
    return value_at_risk(returns=drawdowns, beta=beta)


def cdar(drawdowns: np.ndarray,
         beta: float = 0.95) -> float:
    r"""
   Calculate the historical CDaR (Conditional Drawdown at Risk).

   Parameters
   ----------
   drawdowns : 1d-array
             The drawdowns array

   beta: float, default = 0.95
         The drawdown confidence level (expected drawdown on the worst (1-beta)% observations)

   Returns
   -------
   value : float
           The CVaR
   """
    return cvar(returns=drawdowns, beta=beta)


def max_drawdown(drawdowns: np.ndarray) -> float:
    r"""
    Calculate the Maximum Drawdown.

    Parameters
    ----------
    drawdowns : 1d-array
                The drawdowns array

    Returns
    -------
    value : float
            The Maximum Drawdown
    """
    return dar(drawdowns=drawdowns, beta=1)


def avg_drawdown(drawdowns: np.ndarray) -> float:
    r"""
    Calculate the Average Drawdown.

    Parameters
    ----------
    drawdowns : 1d-array
             The drawdowns array

    Returns
    -------
    value : float
            The Average Drawdown
    """
    return cdar(drawdowns=drawdowns, beta=0)


def edar(drawdowns: np.ndarray,
         beta: float = 0.95) -> tuple[float, float]:
    r"""
    Calculate the EDaR (Entropic Drawdown at Risk) and its associated risk aversion.
    The EDaR is a coherent risk measure which is an upper bound for the DaR and the CDaR,
    obtained from the Chernoff inequality. The EDaR can be represented by using the concept of relative entropy.

    Parameters
    ----------
    drawdowns : 1d-array
                The drawdowns array

    beta: float, default 0.95
      The EDaR confidence level

    Returns
    -------
    value : tuple(float, float)
        The EDaR and its associated risk aversion
    """
    return evar(returns=drawdowns, beta=beta)


def ulcer_index(drawdowns: np.ndarray) -> float:
    r"""
    Calculate the Ulcer Index.

    Parameters
    ----------
    drawdowns : 1d-array
                The drawdowns array

    Returns
    -------
        The Ulcer Index
    """
    return np.sqrt(np.sum(np.power(drawdowns, 2)) / len(drawdowns))


def _owa_gmd_weights(t: int) -> np.ndarray:
    r"""
    Calculate the OWA weights to compute the Gini mean difference (GMD)

     Parameters
    ----------
    t : int
        Number of observations of the return series.

    Returns
    -------
    value : float
            The OWA GMD weights
    """
    return (4 * np.arange(1, t + 1) - 2 * (t + 1)) / (t * (t - 1))


def gini_mean_difference(returns: np.ndarray) -> float:
    r"""
    Calculate the Gini Mean Difference (GMD).
    The Gini Mean Difference is the expected absolute difference between two realisations.
    The GMD is a superior measure of variability  for non-normal distribution than the variance.
    It can be used to form necessary conditions for second-degree stochastic dominance, while the
    variance cannot.

    Parameters
    ----------
    returns : 1d-array
          The returns array
    Returns
    -------
    value : float
            The Gini Mean Difference

    """
    w = _owa_gmd_weights(len(returns))
    return w @ np.sort(returns, axis=0)
