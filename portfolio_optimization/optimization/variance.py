import logging
from typing import Union, Optional
import numpy as np
import cvxpy as cp

from portfolio_optimization.meta import *
from portfolio_optimization.optimization.helpers import *

__all__ = ['mean_variance']

logger = logging.getLogger('portfolio_optimization.mean_variance_optimization')


def maximize_portfolio_returns(expected_returns: np.ndarray,
                               w: cp.Variable,
                               costs: Optional[Union[float, np.ndarray]] = 0,
                               prev_w: Optional[np.ndarray] = None) -> cp.Maximize:
    n = len(expected_returns)
    portfolio_return = expected_returns @ w
    if costs == 0:
        portfolio_cost = 0
    else:
        if prev_w is None:
            prev_w = np.zeros(n)
        else:
            if not isinstance(prev_w, np.ndarray):
                raise TypeError(f'prev_w should be of type numpy.ndarray')
            if len(prev_w) != n:
                raise ValueError(f'prev_w should be of size {n} but received {len(prev_w)}')

        if np.isscalar(costs):
            portfolio_cost = costs * cp.norm(prev_w - w, 1)
        else:
            portfolio_cost = cp.norm(cp.multiply(costs, (prev_w - w)), 1)

    objective = cp.Maximize(portfolio_return - portfolio_cost)

    return objective


def mean_variance(expected_returns: np.ndarray,
                  cov: np.matrix,
                  weight_bounds: Union[tuple[np.ndarray, np.ndarray],
                                       tuple[Optional[float], Optional[float]]],
                  investment_type: InvestmentType,
                  population_size: Optional[int] = None,
                  target_variance: Optional[float] = None,
                  costs: Optional[Union[float, np.ndarray]] = 0,
                  prev_w: Optional[np.ndarray] = None) -> np.array:
    """
    Optimization along the mean-variance frontier (Markowitz optimization).

    :param expected_returns: expected returns for each asset.
    :type expected_returns: np.ndarray of shape(Number of Assets)

    :param cov: covariance of returns for each asset.
    :type cov: np.matrix of shape(Number of Assets, Number of Assets)

    :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair if all identical.
                            No short selling --> (0, None)
    :type weight_bounds: tuple OR tuple list, optional

    :param investment_type: investment type (fully invested, market neutral, unconstrained)
    :type investment_type: InvestmentType

    :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
    :type population_size: int, optional

    :param target_variance: minimize return for the targeted variance.
    :type target_variance: float, optional

    :return the portfolio weights that are in the efficient frontier

    """
    if population_size is None and target_variance is None:
        raise ValueError(f'You have to provide either population_size or target_variance')

    assets_number = len(expected_returns)

    # Variables
    w = cp.Variable(assets_number)

    # Parameters
    target_variance_param = cp.Parameter(nonneg=True)

    # Objectives
    objective = maximize_portfolio_returns(expected_returns=expected_returns,
                                           w=w,
                                           costs=costs,
                                           prev_w=prev_w)

    # Constraints
    portfolio_variance = cp.quad_form(w, cov)
    lower_bounds, upper_bounds = get_lower_and_upper_bounds(weight_bounds=weight_bounds,
                                                            assets_number=assets_number)
    constraints = [portfolio_variance <= target_variance_param,
                   w >= lower_bounds,
                   w <= upper_bounds]
    investment_target = get_investment_target(investment_type=investment_type)
    if investment_target is not None:
        constraints.append(cp.sum(w) == investment_target)

    # Problem
    problem = cp.Problem(objective, constraints)

    # Solve for a variance target
    if target_variance is not None:
        variances = [target_variance]
    else:
        annualized_volatilities = np.logspace(-2.5, -0.5, num=population_size)
        variances = annualized_volatilities ** 2 / 255

    weights = get_optimization_weights(problem=problem,
                                       variable=w,
                                       parameter=target_variance_param,
                                       parameter_array=variances)

    return weights
