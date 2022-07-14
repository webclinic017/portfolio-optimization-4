from typing import Union, Optional
import numpy as np
import cvxpy as cp

from portfolio_optimization.meta import *
from portfolio_optimization.optimization.helpers import *

__all__ = ['mean_cdar']


def mean_cdar(expected_returns: np.ndarray,
              returns: np.ndarray,
              weight_bounds: Union[tuple[np.ndarray, np.ndarray],
                                   tuple[Optional[float], Optional[float]]],
              investment_type: InvestmentType,
              population_size: int,
              beta: float = 0.95) -> np.array:
    """
    Optimization along the mean-CDaR frontier (conditional drawdown-at-risk).

    :param expected_returns: expected returns for each asset.
    :type expected_returns: np.ndarray of shape(Number of Assets)

    :param returns: historic returns for all your assets
    :type returns: np.ndarray of shape(Number of Assets, Number of Observations)

    :param beta: drawdown confidence level (expected drawdown on the worst (1-beta)% days)
    :type beta: float

    :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair if all identical.
                            No short selling --> (0, None)
    :type weight_bounds: tuple OR tuple list, optional

    :param investment_type: investment type (fully invested, market neutral, unconstrained)
    :type investment_type: InvestmentType

    :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
    :type population_size: int

    :return the portfolio weights that are in the efficient frontier

    """
    assets_number, observations_number = returns.shape

    # Variables
    w = cp.Variable(assets_number)
    alpha = cp.Variable()
    u = cp.Variable(observations_number + 1)
    z = cp.Variable(observations_number)

    # Parameters
    target_cdar = cp.Parameter(nonneg=True)

    # Objectives
    portfolio_return = expected_returns.T @ w
    objective = cp.Maximize(portfolio_return)

    # Constraints
    portfolio_cdar = alpha + 1.0 / (observations_number * (1 - beta)) * cp.sum(z)
    lower_bounds, upper_bounds = get_lower_and_upper_bounds(weight_bounds=weight_bounds,
                                                            assets_number=assets_number)
    constraints = [portfolio_cdar <= target_cdar,
                   z >= u[1:] - alpha,
                   z >= 0,
                   u[1:] >= u[:-1] - returns.T @ w,
                   u[0] == 0,
                   u[1:] >= 0,
                   w >= lower_bounds,
                   w <= upper_bounds]
    investment_target = get_investment_target(investment_type=investment_type)
    if investment_target is not None:
        constraints.append(cp.sum(w) == investment_target)

    # Problem
    problem = cp.Problem(objective, constraints)

    # Solve for different volatilities
    weights = get_optimization_weights(problem=problem,
                                       variable=w,
                                       parameter=target_cdar,
                                       parameter_array=np.logspace(-3.5, -0.5, num=population_size))

    return weights
