import logging
from typing import Union, Optional
import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError

from portfolio_optimization.utils.tools import *
from portfolio_optimization.meta import *

__all__ = ['mean_variance']

logger = logging.getLogger('portfolio_optimization.mean_variance_optimization')


def mean_variance(expected_returns: np.ndarray,
                  cov: np.matrix,
                  weight_bounds: Union[tuple[np.ndarray, np.ndarray],
                                                    tuple[Optional[float], Optional[float]]],
                  investment_type: InvestmentType,
                  population_size: int) -> np.array:
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
    :type population_size: int

    :return the portfolio weights that are in the efficient frontier

    """
    assets_number = len(expected_returns)

    # Variables
    w = cp.Variable(assets_number)

    # Parameters
    target_variance = cp.Parameter(nonneg=True)

    # Objectives
    portfolio_return = expected_returns.T @ w
    objective = cp.Maximize(portfolio_return)

    # Constraints
    portfolio_variance = cp.quad_form(w, cov)
    lower_bounds, upper_bounds = get_lower_and_upper_bounds(weight_bounds=weight_bounds,
                                                            assets_number=assets_number)
    constraints = [portfolio_variance <= target_variance,
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
        target_variance.value = annualized_volatility ** 2 / 255
        try:
            problem.solve()
            if w.value is None:
                logger.warning(f'None return for annualized_volatility {annualized_volatility}')
            else:
                weights.append(w.value)
        except SolverError as e:
            logger.warning(f'SolverError for annualized_volatility {annualized_volatility}: {e}')

    return np.array(weights)
