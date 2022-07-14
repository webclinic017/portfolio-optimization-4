import logging
from typing import Union, Optional
import cvxpy as cp
import numpy as np
from cvxpy import SolverError
from scipy.sparse.linalg import ArpackNoConvergence

from portfolio_optimization.meta import *
from portfolio_optimization.exception import *

__all__ = ['get_lower_and_upper_bounds',
           'get_optimization_weights',
           'get_investment_target']

logger = logging.getLogger('portfolio_optimization.optimization')


def get_lower_and_upper_bounds(
        weight_bounds: Union[tuple[np.ndarray, np.ndarray], tuple[Optional[float], Optional[float]]],
        assets_number: int) -> tuple[np.ndarray, np.ndarray]:
    # Upper and lower bounds
    lower_bounds, upper_bounds = weight_bounds
    if lower_bounds is None:
        lower_bounds = -1
    if upper_bounds is None:
        upper_bounds = 1
    if np.isscalar(lower_bounds):
        lower_bounds = np.array([lower_bounds] * assets_number)
    if np.isscalar(upper_bounds):
        upper_bounds = np.array([upper_bounds] * assets_number)

    return lower_bounds, upper_bounds


def get_optimization_weights(problem: cp.Problem,
                             variable: cp.Variable,
                             parameter: cp.Parameter,
                             parameter_array: np.array) -> np.array:
    weights = []
    try:
        for value in parameter_array:
            parameter.value = value
            try:
                problem.solve(solver='ECOS')
                if variable.value is None:
                    logger.warning(f'None return for {value}')
                else:
                    weights.append(variable.value)
            except SolverError as e:
                logger.warning(f'SolverError for {value}: {e}')
    except ArpackNoConvergence:
        raise OptimizationError

    return np.array(weights, dtype=float)


def get_investment_target(investment_type: InvestmentType) -> Optional[int]:
    # Upper and lower bounds

    # Sum of weights
    if investment_type == InvestmentType.FULLY_INVESTED:
        return 1
    elif investment_type == InvestmentType.MARKET_NEUTRAL:
        return 0
