from typing import Union, Optional
import numpy as np
import cvxpy as cp

from portfolio_optimization.utils.tools import *
from portfolio_optimization.meta import *

__all__ = ['mean_cdar']


def mean_cdar(expected_returns: np.ndarray,
              returns: np.ndarray,
              weight_bounds: Union[tuple[np.ndarray, np.ndarray],
                                   tuple[Optional[float], Optional[float]]],
              investment_type: InvestmentType,
              population_size: int,
              beta: float = 0.95,
              ) -> np.array:
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

    self._alpha = cp.Variable()
    self._u = cp.Variable(len(self.returns) + 1)
    self._z = cp.Variable(len(self.returns))

    assets_number = len(expected_returns)
    observations_number = returns.shape[1]

    # Additional matrix
    if not np.isscalar(returns_target):
        returns_target = returns_target[:, np.newaxis]
    B = (returns - returns_target) / np.sqrt(observations_number)
    cdar = self._alpha + 1.0 / (len(self.returns) * (1 - self._beta)) * cp.sum(
        self._z
    )
    target_cdar_par = cp.Parameter(
        value=target_cdar, name="target_cdar", nonneg=True
    )
    self.add_constraint(lambda _: cdar <= target_cdar_par)
    self.add_constraint(lambda _: self._z >= self._u[1:] - self._alpha)
    self.add_constraint(
        lambda w: self._u[1:] >= self._u[:-1] - self.returns.values @ w
    )
    self.add_constraint(lambda _: self._u[0] == 0)
    self.add_constraint(lambda _: self._z >= 0)
    self.add_constraint(lambda _: self._u[1:] >= 0)


    # Variables
    w = cp.Variable(assets_number)
    alpha = cp.Variable()
    u = cp.Variable(len(self.returns) + 1)
    z = cp.Variable(len(self.returns))

    # Parameters
    target_semivariance = cp.Parameter(nonneg=True)

    # Objectives
    portfolio_return = expected_returns.T @ w
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
