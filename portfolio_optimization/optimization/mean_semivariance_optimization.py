from typing import Union, Optional
import numpy as np
import cvxpy as cp

from portfolio_optimization.utils.tools import *
from portfolio_optimization.meta import *

__all__ = ['mean_semivariance_optimization']


def mean_semivariance_optimization(mu: np.ndarray,
                                   returns: np.ndarray,
                                   returns_target: Union[float, np.ndarray],
                                   weight_bounds: Union[tuple[np.ndarray, np.ndarray],
                                                        tuple[Optional[float], Optional[float]]],
                                   investment_type: InvestmentType,

                                   population_size: int) -> np.array:
    """
    Optimization along the mean-semivariance frontier.
    :param mu: expected returns for each asset.
    :type mu: np.ndarray of shape(Number of Assets)

    :param returns: historic returns for all your assets
    :type returns: np.ndarray of shape(Number of Assets, Number of Observations)

    :param returns_target: the return target to distinguish "downside" and "upside".
    :type returns_target: float or np.ndarray of shape(Number of Assets)

    :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair if all identical.
                            No short selling --> (0, None)
    :type weight_bounds: tuple OR tuple list, optional

    :param investment_type: investment type (fully invested, market neutral, unconstrained)
    :type investment_type: InvestmentType

    :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
    :type population_size: int

    :return the portfolio weights that are in the efficient frontier

    """
    assets_number = len(mu)
    observations_number = returns.shape[1]

    # Additional matrix
    if not np.isscalar(returns_target):
        returns_target = returns_target[:, np.newaxis]
    B = (returns - returns_target) / np.sqrt(observations_number)

    # Variables
    w = cp.Variable(assets_number)
    p = cp.Variable(observations_number, nonneg=True)
    n = cp.Variable(observations_number, nonneg=True)

    # Parameters
    target_semivariance = cp.Parameter(nonneg=True)

    # Objectives
    portfolio_return = mu.T @ w
    objective = cp.Maximize(portfolio_return)

    # Constraints
    portfolio_semivariance = cp.sum(cp.square(n))
    lower_bounds, upper_bounds = get_lower_and_upper_bounds(weight_bounds=weight_bounds,
                                                            assets_number=assets_number)
    constraints = [portfolio_semivariance <= target_semivariance,
                   B.T @ w - p + n == 0,
                   w >= lower_bounds,
                   w <= upper_bounds]
    investment_target = get_investment_target(investment_type=investment_type)
    if investment_target is not None:
        constraints.append(cp.sum(w) == investment_target)

    # Problem
    problem = cp.Problem(objective, constraints)

    # Solve for different volatilities
    weights = []
    for annualized_volatility in np.logspace(-2.5, -0.5, num=population_size):
        target_semivariance.value = annualized_volatility ** 2 / 255
        problem.solve()
        weights.append(w.value)

    return np.array(weights)
