import logging
from typing import Union, Optional
import numpy as np
import cvxpy as cp
from cvxpy import SolverError
from scipy.sparse.linalg import ArpackNoConvergence

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.exception import *
from portfolio_optimization.utils.tools import *

__all__ = ['Optimization']

logger = logging.getLogger('portfolio_optimization.optimization')


class Optimization:
    def __init__(self,
                 assets: Assets,
                 investment_type: InvestmentType = InvestmentType.FULLY_INVESTED,
                 weight_bounds: tuple[Optional[Union[float, np.ndarray]],
                                      Optional[Union[float, np.ndarray]]] = (None, None),
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
                              * For a fixed entry cost of 1% based the nominal amount in the asset currency,
                                then costs=0.01
                              * For a fixed entry cost of 5 Ccy (asset currency), then that cost needs to be converted
                                in a notional amount equivalent with costs = 5 * asset price
        :type costs: float or np.ndarray of shape(Number of Assets)

        :param investment_duration_in_days: The expected investment duration in business days.
                  When costs are provided, they need to be converted to an average daily cost over
                  the expected investment duration. This is because the optimization problem has no notion of investment
                  duration.
                  For example, lets assume that asset A has an expected daily return of 0.01%
                  with a fixed entry cost of 1% and asset B has an expected daily return of 0.005%
                  with a fixed entry cost of 0%. Both having same volatility and correlated with r=1.
                  If the investment duration is only one month, we should allocate all the weights to asset B
                  whereas if the investment duration is one year, we should allocate all the weights to asset A.
                  Duration = 1 months (21 business days):
                        * expected return A = (1+0.01%)^21 - 1 - 1% ≈ 0.01% * 21 - 1% ≈ -0.8%
                        * expected return B ≈ 0.005% * 21 - 0% ≈ 0.1%
                  Duration = 1 year (255 business days):
                        * expected return A ≈ 0.01% * 255 - 1% ≈ 1.5%
                        * expected return B ≈ 0.005% * 21 - 0% ≈ 1.3%
                 In order to take that into account, the costs provided are divided by the expected investment duration
                 in days in the optimization problem.

        :param prev_w: previous weights
        :type prev_w: np.ndarray of shape(Number of Assets), default None (equivalent to an array of zeros)
        """
        self.assets = assets
        self.investment_type = investment_type
        self.weight_bounds = weight_bounds
        self.costs = costs
        self.investment_duration_in_days = investment_duration_in_days
        self.prev_w = prev_w
        self.loaded = True
        self._validation()

    def update(self, **kwargs):
        self.loaded = False
        valid_kwargs = ['assets',
                        'investment_type',
                        'weight_bounds',
                        'costs',
                        'investment_duration_in_days',
                        'prev_w']
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError(f'Invalid keyword argument {k}')
            setattr(self, k, v)
        self._validation()
        self.loaded = True

    def __setattr__(self, name, value):
        if name != 'loaded' and self.__dict__.get('loaded'):
            logger.warning(f'Attributes should be updated with the update() method to allow proper validation')
        super().__setattr__(name, value)

    def _validation(self):

        if self.assets.asset_nb < 2:
            raise ValueError(f'assets should contains more than one asset')

        if not isinstance(self.weight_bounds, tuple) or len(self.weight_bounds) != 2:
            raise ValueError(f'weight_bounds should be a tuple of size 2')

        for i in [0, 1]:
            if isinstance(self.weight_bounds[i], np.ndarray) and len(self.weight_bounds[i]) != self.assets.asset_nb:
                raise ValueError(f'the weight_bounds arrays should be of size {self.assets.asset_nb}, '
                                 f'but received {len(self.weight_bounds[i])}')

        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()

        if not np.all(lower_bounds <= upper_bounds):
            raise ValueError(f'All elements of the lower bounds should be less or equal than all elements of the upper '
                             f'bound')

        investment_target = self._get_investment_target()
        if investment_target is not None:
            if sum(upper_bounds) < investment_target:
                raise ValueError(f'When investment_type is {self.investment_type.value}, the sum of all upper bounds '
                                 f'should be greater or equal to {investment_target}')
            if sum(lower_bounds) > investment_target:
                raise ValueError(f'When investment_type is {self.investment_type.value}, the sum of all lower bounds '
                                 f'should be less or equal to {investment_target}')

        if self.costs is not None and not (np.isscalar(self.costs) and self.costs == 0):
            if self.investment_duration_in_days is None:
                raise ValueError(f'investment_duration_in_days cannot be missing when costs is provided')

        if self.prev_w is not None:
            if not isinstance(self.prev_w, np.ndarray):
                raise TypeError(f'prev_w should be of type numpy.ndarray')
            if len(self.prev_w) != self.assets.asset_nb:
                raise ValueError(f'prev_w should be of size {self.assets.asset_nb} but received {len(self.prev_w)}')

    def _portfolio_returns(self,
                           w: cp.Variable,
                           l1_coef: Optional[float] = None,
                           l2_coef: Optional[float] = None) -> cp.Expression:
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

        # Norm L1
        if l1_coef is None or l1_coef == 0:
            l1_regularization = 0
        else:
            l1_regularization = l1_coef * cp.norm(w, 1)

        # Norm L2
        if l2_coef is None or l2_coef == 0:
            l2_regularization = 0
        else:
            l2_regularization = l2_coef * cp.sum_squares(w)

        portfolio_return = (self.assets.expected_returns @ w
                            - portfolio_cost
                            - l1_regularization
                            - l2_regularization)

        return portfolio_return

    def _get_lower_and_upper_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Upper and lower bounds
        lower_bounds, upper_bounds = self.weight_bounds
        if lower_bounds is None:
            lower_bounds = -np.inf
        if upper_bounds is None:
            upper_bounds = np.inf

        if np.isscalar(lower_bounds):
            lower_bounds = np.array([lower_bounds] * self.assets.asset_nb)
        if np.isscalar(upper_bounds):
            upper_bounds = np.array([upper_bounds] * self.assets.asset_nb)

        return lower_bounds, upper_bounds

    @staticmethod
    def _get_optimization_weights(problem: cp.Problem,
                                  w: cp.Variable,
                                  parameter: cp.Parameter,
                                  target: Union[float, np.ndarray],
                                  ignore_none: bool = True) -> list[Union[np.ndarray, None]]:

        if np.isscalar(target):
            parameter_array = [target]
        else:
            parameter_array = target

        weights = []
        for value in parameter_array:
            parameter.value = value
            weight = None
            try:
                problem.solve(solver='ECOS')
                if w.value is None:
                    logger.warning(f'None return for {value}')
                weight = w.value
            except SolverError as e:
                logger.warning(f'SolverError for {value}: {e}')
            except ArpackNoConvergence as e:
                logger.warning(f'ArpackNoConvergence for {value}: {e}')
            if weight is not None or not ignore_none:
                weights.append(weight)

        if np.isscalar(target):
            if len(weights) == 0:
                raise OptimizationError(f'Optimization did not converge')
            return weights[0]

        return weights

    def _get_investment_target(self) -> Optional[int]:
        # Sum of weights
        if self.investment_type == InvestmentType.FULLY_INVESTED:
            return 1
        elif self.investment_type == InvestmentType.MARKET_NEUTRAL:
            return 0

    def _validate_args(self, **kwargs):
        population_size = kwargs.get('population_size')
        targets_names = [k for k, v in kwargs.items() if k.startswith('target_')]
        not_none_targets_names = [k for k in targets_names if kwargs[k] is not None]
        if len(not_none_targets_names) > 1:
            raise ValueError(f'Only one target has to be provided but received {" AND ".join(not_none_targets_names)}')
        elif len(not_none_targets_names) == 1:
            target_name = targets_names[0]
        else:
            target_name = None

        if ((population_size is None and target_name is None) or
                (population_size is not None and target_name is not None)):
            raise ValueError(f'You have to provide population_size OR {" OR ".join(not_none_targets_names)}')

        if population_size is not None and population_size <= 1:
            raise ValueError('f population_size should be strictly greater than one')

        if target_name is not None:
            target = kwargs[target_name]
            if np.isscalar(target):
                if target <= 0:
                    raise ValueError(f'{target_name} should be strictly positive')
            elif isinstance(target, np.ndarray) or isinstance(target, list):
                if np.any(np.array(target) <= 0):
                    raise ValueError(f'All values of {target_name} should be strictly positive')
            else:
                raise ValueError(f'{target_name} should be a scalar, numpy.ndarray or list. '
                                 f'But received {type(target)}')

        for k, v in kwargs.items():
            if k.endswith('coef') and v is not None:
                if v < 0:
                    raise ValueError(f'{k} cannot be negative')
                elif v > 0 and np.all(np.array(self.weight_bounds) >= 0):
                    logger.warning(f'Positive {k} will have no impact with positive or null lower bounds')

    def mean_variance(self,
                      target_volatility: Optional[Union[float, list, np.ndarray]] = None,
                      population_size: Optional[int] = None,
                      l1_coef: Optional[float] = None,
                      l2_coef: Optional[float] = None,
                      ignore_none: bool = True) -> Union[list[np.ndarray], np.ndarray]:
        """
        Optimization along the mean-variance frontier (Markowitz optimization).

        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int, optional

        :param target_volatility: minimize return for the targeted daily volatility of the portfolio.
        :type target_volatility: float or list or numpy.ndarray optional

        :param l1_coef: L1 regularisation coefficient. Increasing this coef will reduce the number of non-zero weights.
                        It's like the L1 regularisation in Lasso.
                        If both l1_coef and l2_coef are strictly positive, it's like the regularisation in Elastic-Net.
        :type l1_coef: float, default to None

        :param l2_coef: L2 regularisation coefficient. It's like the L2 regularisation in Ridge.
                        If both l1_coef and l2_coef are strictly positive, it's like the regularisation in Elastic-Net.
        :type l1_coef: float, default to None

        :param ignore_none: if True, None are removed from the list of weights results when the optimization failed
        :type ignore_none: bool, default True

        :return the portfolio weights that are in the efficient frontier.
        :rtype: list of numpy.ndarray or numpy.ndarray
        """
        self._validate_args(**{k: v for k, v in locals().items() if k != 'self'})

        min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(self.assets.expected_cov)))

        if np.isscalar(target_volatility) and target_volatility < min_volatility:
            raise ValueError(f'The minimum volatility is {min_volatility:.3f}. '
                             f'Please use a higher target_volatility')

        # Variables
        w = cp.Variable(self.assets.asset_nb)

        # Parameters
        target_variance_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = cp.Maximize(self._portfolio_returns(w=w, l1_coef=l1_coef, l2_coef=l2_coef))

        # Constraints
        portfolio_variance = cp.quad_form(w, self.assets.expected_cov)
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        constraints = [portfolio_variance <= target_variance_param,
                       w >= lower_bounds,
                       w <= upper_bounds]
        investment_target = self._get_investment_target()
        if investment_target is not None:
            constraints.append(cp.sum(w) == investment_target)

        # Problem
        problem = cp.Problem(objective, constraints)

        if target_volatility is not None:
            if np.isscalar(target_volatility):
                target = target_volatility ** 2
            else:
                target = np.array(target_volatility) ** 2

        else:
            start = np.log10(min_volatility * 1.3)  # We start at min * 130% to increase proba of convergence
            end = np.log10(0.3 / np.sqrt(255))  # We stop at 30% annualized volatility
            volatilities = np.logspace(start, end, num=population_size)
            target = volatilities ** 2

        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_variance_param,
                                                 target=target,
                                                 ignore_none=ignore_none)

        return weights

    def maximum_sharpe(self) -> np.ndarray:
        """
        Maximize the sharpe ratio.

        :return the portfolio weights that maximize the sharpe ratio of the portfolio.
        :rtype: numpy.ndarray
        """

        if self.investment_type != InvestmentType.FULLY_INVESTED:
            raise ValueError('maximum_sharpe() can be solved only for investment_type=InvestmentType.FULLY_INVESTED'
                             '  --> you can find an approximation by computing the efficient frontier with '
                             ' mean_variance(population=30) and finding the portfolio with the highest sharpe ratio.')

        if self.costs is not None:
            raise ValueError('maximum_sharpe() cannot be solved with costs '
                             '  --> you can find an approximation by computing the efficient frontier with '
                             ' mean_variance(population=30) and finding the portfolio with the highest sharpe ratio.')

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        k = cp.Variable()

        # Objectives
        objective = cp.Minimize(cp.quad_form(w, self.assets.expected_cov))

        # Constraints
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        constraints = [self._portfolio_returns(w=w) == 1,
                       w >= lower_bounds * k,
                       w <= upper_bounds * k,
                       cp.sum(w) == k,
                       k >= 0]

        # Problem
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver='ECOS')
            if w.value is None or k.value is None:
                logger.warning(f'None return')
                raise OptimizationError
            return np.array(w.value / k.value, dtype=float)
        except SolverError as e:
            logger.warning(f'SolverError for: {e}')
            raise OptimizationError
        except ArpackNoConvergence as e:
            logger.warning(f'ArpackNoConvergence for: {e}')
            raise OptimizationError

    def mean_semivariance(self,
                          returns_target: Optional[Union[float, np.ndarray]] = None,
                          target_semideviation: Optional[Union[float, list, np.ndarray]] = None,
                          population_size: Optional[int] = None,
                          ignore_none: bool = True) -> Union[list[np.ndarray], np.ndarray]:
        """
        Optimization along the mean-semivariance frontier.

        :param returns_target: the return target to distinguish "downside" and "upside".
        :type returns_target: float or np.ndarray of shape(Number of Assets)

        :param target_semideviation: minimize return for the targeted semideviation of the portfolio.
        :type target_semideviation: float or list or numpy.ndarray optional

        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int

        :param ignore_none: if True, None are removed from the list of weights results when the optimization failed
        :type ignore_none: bool, default True

        :return the portfolio weights that are in the efficient frontier
        :rtype: list of numpy.ndarray or numpy.ndarray
        """
        self._validate_args(population_size=population_size,
                            target_semideviation=target_semideviation)

        if returns_target is None:
            returns_target = self.assets.expected_returns

        # Additional matrix
        if not np.isscalar(returns_target):
            returns_target = returns_target[:, np.newaxis]
        b = (self.assets.returns - returns_target) / np.sqrt(self.assets.date_nb)

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        p = cp.Variable(self.assets.date_nb, nonneg=True)
        n = cp.Variable(self.assets.date_nb, nonneg=True)

        # Parameters
        target_semivariance_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = cp.Maximize(self._portfolio_returns(w=w))

        # Constraints
        portfolio_semivariance = cp.sum(cp.square(n))
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()

        constraints = [portfolio_semivariance <= target_semivariance_param,
                       b.T @ w - p + n == 0,
                       w >= lower_bounds,
                       w <= upper_bounds]

        investment_target = self._get_investment_target()
        if investment_target is not None:
            constraints.append(cp.sum(w) == investment_target)

        # Problem
        problem = cp.Problem(objective, constraints)

        if target_semideviation is not None:
            if np.isscalar(target_semideviation):
                target = target_semideviation ** 2
            else:
                target = np.array(target_semideviation) ** 2
        else:
            # Solve for multiple semideviations
            min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(self.assets.expected_cov)))
            start = np.log10(min_volatility * 1.3)  # We start at min_volatility * 130% to increase proba of convergence
            end = np.log10(0.3 / np.sqrt(255))  # We stop at 30% annualized semideviation
            semideviations = np.logspace(start, end, num=population_size)
            target = semideviations ** 2

        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_semivariance_param,
                                                 target=target,
                                                 ignore_none=ignore_none)

        return weights

    def mean_cvar(self,
                  beta: float = 0.95,
                  target_cvar: Optional[Union[float, list, np.ndarray]] = None,
                  population_size: Optional[int] = None,
                  ignore_none: bool = True) -> Union[list[np.ndarray], np.ndarray]:
        """
        Optimization along the mean-CVaR frontier (Conditional Value-at-Risk or Expected Shortfall).
        CVaR is the average of the “extreme” losses beyond the VaR threshold.

        :param beta: var confidence level (expected VaR on the worst (1-beta)% days)
        :type beta: float

        :param target_cvar: minimize return for the targeted cvar.
        :type target_cvar: float or list or numpy.ndarray optional

        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int

        :param ignore_none: if True, None are removed from the list of weights results when the optimization failed
        :type ignore_none: bool, default True

        :return the portfolio weights that are in the efficient frontier
        :rtype: list of numpy.ndarray or numpy.ndarray
        """

        self._validate_args(population_size=population_size,
                            target_cvar=target_cvar)

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        alpha = cp.Variable()
        u = cp.Variable(self.assets.date_nb)

        # Parameters
        target_cvar_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = cp.Maximize(self._portfolio_returns(w=w))

        # Constraints
        portfolio_cvar = alpha + 1.0 / (self.assets.date_nb * (1 - beta)) * cp.sum(u)
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()

        constraints = [portfolio_cvar <= target_cvar_param,
                       self.assets.returns.T @ w + alpha + u >= 0,
                       u >= 0,
                       w >= lower_bounds,
                       w <= upper_bounds]

        investment_target = self._get_investment_target()
        if investment_target is not None:
            constraints.append(cp.sum(w) == investment_target)

        # Problem
        problem = cp.Problem(objective, constraints)

        if target_cvar is not None:
            if np.isscalar(target_cvar):
                target = target_cvar
            else:
                target = np.array(target_cvar)
        else:
            target = np.logspace(-2, -0.5, num=population_size)

        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_cvar_param,
                                                 target=target,
                                                 ignore_none=ignore_none)

        return weights

    def mean_cdar(self,
                  beta: float = 0.95,
                  target_cdar: Optional[Union[float, list, np.ndarray]] = None,
                  population_size: Optional[int] = None,
                  ignore_none: bool = True) -> Union[list[np.ndarray], np.ndarray]:
        """
        Optimization along the mean-CDaR frontier (Conditional Drawdown-at-Risk).
        The Conditional Drawdown-at-Risk is the average drawdown for all the days that drawdown exceeds a threshold.

        :param beta: drawdown confidence level (expected drawdown on the worst (1-beta)% days)
        :type beta: float

        :param target_cdar: minimize return for the targeted cdar.
        :type target_cdar: float or list or numpy.ndarray optional

        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int

        :param ignore_none: if True, None are removed from the list of weights results when the optimization failed
        :type ignore_none: bool, default True

        :return the portfolio weights that are in the efficient frontier
        :rtype: list of numpy.ndarray or numpy.ndarray

        """

        self._validate_args(population_size=population_size,
                            target_cdar=target_cdar)

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        alpha = cp.Variable()
        u = cp.Variable(self.assets.date_nb + 1)
        z = cp.Variable(self.assets.date_nb)

        # Parameters
        target_cdar_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = cp.Maximize(self._portfolio_returns(w=w))

        # Constraints
        portfolio_cdar = alpha + 1.0 / (self.assets.date_nb * (1 - beta)) * cp.sum(z)
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()

        constraints = [portfolio_cdar <= target_cdar_param,
                       z >= u[1:] - alpha,
                       z >= 0,
                       u[1:] >= u[:-1] - self.assets.returns.T @ w,
                       u[0] == 0,
                       u[1:] >= 0,
                       w >= lower_bounds,
                       w <= upper_bounds]

        investment_target = self._get_investment_target()
        if investment_target is not None:
            constraints.append(cp.sum(w) == investment_target)

        # Problem
        problem = cp.Problem(objective, constraints)

        if target_cdar is not None:
            if np.isscalar(target_cdar):
                target = target_cdar
            else:
                target = np.array(target_cdar)
        else:
            # Solve for multiple cdar
            target = np.logspace(-2, -0.5, num=population_size)

        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_cdar_param,
                                                 target=target,
                                                 ignore_none=ignore_none)

        return weights

    def inverse_volatility(self) -> np.ndarray:
        """
        Asset Weights are proportional to 1 / asset volatility and sums to 1
        """
        weights = 1 / self.assets.std
        weights = weights / sum(weights)
        return weights

    def equal_weighted(self) -> np.ndarray:
        """
        Equal Weighted, summing to 1
        """
        weights = np.ones(self.assets.asset_nb) / self.assets.asset_nb
        return weights

    def random(self) -> np.ndarray:
        """
        Random positive weights that sum to 1 and respects the bounds.
        """
        # Produces n random weights that sum to 1 with uniform distribution over the simplex
        weights = rand_weights_dirichlet(n=self.assets.asset_nb)
        # Respecting bounds
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        weights = np.minimum(np.maximum(weights, lower_bounds), upper_bounds)
        weights = weights / sum(weights)
        return weights
