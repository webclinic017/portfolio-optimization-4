import logging
from typing import Union, Optional
import numpy as np
import cvxpy as cp

from portfolio_optimization.meta import *
from portfolio_optimization.optimization.helpers import *

__all__ = ['mean_variance']

logger = logging.getLogger('portfolio_optimization.mean_variance_optimization')


def mean_variance(expected_returns: np.ndarray,
                  cov: np.matrix,
                  weight_bounds: Union[tuple[np.ndarray, np.ndarray],
                                       tuple[Optional[float], Optional[float]]],
                  investment_type: InvestmentType,
                  population_size: Optional[int] = None,
                  target_variance: Optional[float] = None,
                  costs: Optional[Union[float, np.ndarray]] = None,
                  investment_duration_in_days: Optional[int] = None,
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
             In order to take that into account, the costs provided are divided by the expected investment duration in
             days in the optimization problem.

    :param prev_w: previous weights
    :type prev_w: np.ndarray of shape(Number of Assets), default None (equivalent to an array of zeros)

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
                                           prev_w=prev_w,
                                           investment_duration_in_days=investment_duration_in_days)

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
