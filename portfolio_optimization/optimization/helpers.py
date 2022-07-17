import logging
from typing import Union, Optional
import cvxpy as cp
import numpy as np
from cvxpy import SolverError
from scipy.sparse.linalg import ArpackNoConvergence

from portfolio_optimization.meta import *
from portfolio_optimization.exception import *

__all__ = ['maximize_portfolio_returns',
           'get_lower_and_upper_bounds',
           'get_optimization_weights',
           'get_investment_target']

logger = logging.getLogger('portfolio_optimization.optimization')


def maximize_portfolio_returns(expected_returns: np.ndarray,
                               w: cp.Variable,
                               costs: Optional[Union[float, np.ndarray]] = None,
                               investment_duration_in_days: Optional[int] = None,
                               prev_w: Optional[np.ndarray] = None) -> cp.Maximize:
    """
    :param expected_returns: Expected returns for each asset.
    :type expected_returns: np.ndarray of shape(Number of Assets)

    :param w: Weight variable
    :type w: cp.Variable of shape(Number of Assets)

     :param costs: Transaction costs. Costs represent fixed costs charged on the notional amount invested.
                  Example:
                          - For a fixed entry cost of 1% based the nominal amount in the asset currency, then costs=0.01
                          - For a fixed entry cost of 5 Ccy (asset currency), then that cost needs to be converted
                            in a notional amount equivalent with costs = 5 * asset price
    :type costs: float or np.ndarray of shape(Number of Assets)

    :param investment_duration_in_days: The expected investment duration in business days.
              When costs are provided, they need to be converted to an average daily cost over the expected investment
              duration. This is because the optimization problem has no notion of investment duration.
              For example, lets assume that asset A has an expected daily return of 0.01%
              with a fixed entry cost of 1% and asset B has an expected daily return of 0.005%
              with a fixed entry cost of 0%. Both having same volatility and correlated with r=1.
              If the investment duration is only one month, we should allocate all the weights to asset B
              whereas if the investment duration is one year, we should allocate all the weights to asset A.
              Duration = 1 months (21 business days):
                    expected return A = (1+0.01%)^21 - 1 - 1% ≈ 0.01% * 21 - 1% ≈ -0.8%
                    expected return B ≈ 0.005% * 21 - 0% ≈ 0.1%
              Duration = 1 year (255 business days):
                    expected return A ≈ 0.01% * 255 - 1% ≈ 1.5%
                    expected return B ≈ 0.005% * 21 - 0% ≈ 1.3%

    :param prev_w: previous weights
    :type prev_w: np.ndarray of shape(Number of Assets), default None (equivalent to an array of zeros)


    """
    n = len(expected_returns)
    portfolio_return = expected_returns @ w
    if costs is None or (np.isscalar(costs) and costs == 0):
        portfolio_cost = 0
    else:
        if prev_w is None:
            prev_w = np.zeros(n)
        else:
            if not isinstance(prev_w, np.ndarray):
                raise TypeError(f'prev_w should be of type numpy.ndarray')
            if len(prev_w) != n:
                raise ValueError(f'prev_w should be of size {n} but received {len(prev_w)}')

        if investment_duration_in_days is None:
            raise ValueError(f'investment_duration_in_days cannot be missing when costs is provided')

        costs = costs / investment_duration_in_days
        if np.isscalar(costs):
            portfolio_cost = costs * cp.norm(prev_w - w, 1)
        else:
            portfolio_cost = cp.norm(cp.multiply(costs, (prev_w - w)), 1)

    objective = cp.Maximize(portfolio_return - portfolio_cost)

    return objective


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
