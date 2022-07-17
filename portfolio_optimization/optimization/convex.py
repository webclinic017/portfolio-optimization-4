import logging
from typing import Union, Optional
import numpy as np
import cvxpy as cp
from cvxpy import SolverError
from scipy.sparse.linalg import ArpackNoConvergence

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.exception import *

__all__ = ['Optimization']

logger = logging.getLogger('portfolio_optimization.optimization')


class Optimization:
    def __init__(self,
                 assets: Assets,
                 investment_type: InvestmentType,
                 weight_bounds: Union[tuple[np.ndarray, np.ndarray],
                                      tuple[Optional[float], Optional[float]]],
                 costs: Optional[Union[float, np.ndarray]] = None,
                 investment_duration_in_days: Optional[int] = None,
                 prev_w: Optional[np.ndarray] = None):
        """
        :param investment_type: investment type (fully invested, market neutral, unconstrained)
        :type investment_type: InvestmentType

        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair if all identical.
                            No short selling --> (0, None)
        :type weight_bounds: tuple OR tuple list, optional

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
        """
        self.assets = assets
        self.investment_type = investment_type
        self.weight_bounds = weight_bounds
        self.costs = costs
        self.investment_duration_in_days = investment_duration_in_days
        self.prev_w = prev_w
        self._validation()

    def _validation(self):
        if self.assets.asset_nb < 2:
            raise ValueError(f'assets should contains more than one asset')
        if not isinstance(self.weight_bounds, tuple) or len(self.weight_bounds) != 2:
            raise ValueError(f'weight_bounds should be a tuple of size 2')
        if (isinstance(self.weight_bounds[0], np.ndarray) and not isinstance(self.weight_bounds[1], np.ndarray)
                or isinstance(self.weight_bounds[1], np.ndarray) and not isinstance(self.weight_bounds[0], np.ndarray)):
            raise ValueError(f'if one element of weight_bounds is an numpy array, '
                             f'the other one should also be a numpy array')
        if isinstance(self.weight_bounds[0], np.ndarray):
            for i in [0, 1]:
                if len(self.weight_bounds[i]) != self.assets.asset_nb:
                    raise ValueError(f'the weight_bounds arrays should be of size {self.assets.asset_nb}, '
                                     f'but received {len(self.weight_bounds[i])}')
        if self.costs is not None or not (np.isscalar(self.costs) and self.costs == 0):
            if self.investment_duration_in_days is None:
                raise ValueError(f'investment_duration_in_days cannot be missing when costs is provided')
        if self.prev_w is not None:
            if not isinstance(self.prev_w, np.ndarray):
                raise TypeError(f'prev_w should be of type numpy.ndarray')
            if len(self.prev_w) != self.assets.asset_nb:
                raise ValueError(f'prev_w should be of size {self.assets.asset_nb} but received {len(self.prev_w)}')

    def _maximize_portfolio_returns(self, w: cp.Variable) -> cp.Maximize:
        portfolio_return = self.assets.expected_returns @ w
        if self.costs is None or (np.isscalar(self.costs) and self.costs == 0):
            portfolio_cost = 0
        else:
            if self.prev_w is None:
                prev_w = np.zeros(self.assets.asset_nb)
            else:
                prev_w = self.prev_w
            daily_costs = self.costs / self.investment_duration_in_days
            if np.isscalar(daily_costs):
                portfolio_cost = daily_costs * cp.norm(prev_w - w, 1)
            else:
                portfolio_cost = cp.norm(cp.multiply(daily_costs, (prev_w - w)), 1)

        objective = cp.Maximize(portfolio_return - portfolio_cost)

        return objective

    def _get_lower_and_upper_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Upper and lower bounds
        lower_bounds, upper_bounds = self.weight_bounds
        if lower_bounds is None:
            lower_bounds = -1
        if upper_bounds is None:
            upper_bounds = 1
        if np.isscalar(lower_bounds):
            lower_bounds = np.array([lower_bounds] * self.assets.asset_nb)
        if np.isscalar(upper_bounds):
            upper_bounds = np.array([upper_bounds] * self.assets.asset_nb)

        return lower_bounds, upper_bounds

    def _get_optimization_weights(self,
                                  problem: cp.Problem,
                                  w: cp.Variable,
                                  parameter: cp.Parameter,
                                  parameter_array: np.array) -> np.array:
        weights = []
        try:
            for value in parameter_array:
                parameter.value = value
                try:
                    problem.solve(solver='ECOS')
                    if w.value is None:
                        logger.warning(f'None return for {value}')
                    else:
                        weights.append(w.value)
                except SolverError as e:
                    logger.warning(f'SolverError for {value}: {e}')
        except ArpackNoConvergence:
            raise OptimizationError

        return np.array(weights, dtype=float)

    def _get_investment_target(self) -> Optional[int]:
        # Sum of weights
        if self.investment_type == InvestmentType.FULLY_INVESTED:
            return 1
        elif self.investment_type == InvestmentType.MARKET_NEUTRAL:
            return 0

    def mean_variance(self,
                      population_size: Optional[int] = None,
                      target_variance: Optional[float] = None) -> np.array:
        """
        Optimization along the mean-variance frontier (Markowitz optimization).

        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int, optional

        :param target_variance: minimize return for the targeted variance.
        :type target_variance: float, optional

        :return the portfolio weights that are in the efficient frontier
        """
        if population_size is None and target_variance is None:
            raise ValueError(f'You have to provide either population_size or target_variance')

        # Variables
        w = cp.Variable(self.assets.asset_nb)

        # Parameters
        target_variance_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = self._maximize_portfolio_returns(w=w)

        # Constraints
        portfolio_variance = cp.quad_form(w, self.assets.cov)
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        constraints = [portfolio_variance <= target_variance_param,
                       w >= lower_bounds,
                       w <= upper_bounds]
        investment_target = self._get_investment_target()
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

        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_variance_param,
                                                 parameter_array=variances)

        return weights
