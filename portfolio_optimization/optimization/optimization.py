import logging
from typing import Union, Optional
import numpy as np
import cvxpy as cp
from cvxpy import SolverError
from scipy.sparse.linalg import ArpackNoConvergence
from enum import Enum, unique
from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.exception import *
from portfolio_optimization.utils.tools import *

__all__ = ['Optimization']

logger = logging.getLogger('portfolio_optimization.optimization')


@unique
class RiskMeasures(Enum):
    VARIANCE = 0
    SEMI_VARIANCE = 1
    CVAR = 2
    CDAR = 3


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
        Convex portfolio optimization
        :param investment_type: investment type (fully invested, market neutral, unconstrained)
        :type investment_type: InvestmentType
        :param weight_bounds: Minimum and maximum weight of each asset OR single min/max pair if all identical.
                              None lower bound is defaulted to -1.
                              None upper bound is defaulted to 1.
                              Default is (None, None) --> (-1, 1)
                              For example, for no short selling --> (0, None)
        :type weight_bounds: tuple OR tuple list, optional
        :param costs: Transaction costs. Costs represent fixed costs charged on the notional amount invested.
                      Example:
                              * costs = 0.01: for a fixed entry cost of 1% based on the notional amount in the asset
                              currency
                              * costs = 5 * asset price: for a fixed entry cost of 5 Ccy (asset currency) (that cost
                              needs to be converted in a notional amount equivalent)
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
                 in days in the optimization problem
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
        """
        Update the class attributes then re-validate them
        """
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
        """
        Validate the class attributes
        """
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

    def _validate_args(self, **kwargs):
        """
        Validate function arguments
        """
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
                if target < 0:
                    raise ValueError(f'{target_name} should be positive')
            elif isinstance(target, np.ndarray) or isinstance(target, list):
                if np.any(np.array(target) < 0):
                    raise ValueError(f'All values of {target_name} should be positive')
            else:
                raise ValueError(f'{target_name} should be a scalar, numpy.ndarray or list. '
                                 f'But received {type(target)}')

        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        for k, v in kwargs.items():
            if k.endswith('coef') and v is not None:
                if v < 0:
                    raise ValueError(f'{k} cannot be negative')
                elif v > 0 and np.all(lower_bounds >= 0):
                    logger.warning(f'Positive {k} will have no impact with positive or null lower bounds')

    def _portfolio_expected_return(self,
                                   w: cp.Variable,
                                   l1_coef: Optional[float] = None,
                                   l2_coef: Optional[float] = None) -> cp.Expression:
        """
        CVXPY Expression of the portfolio expected return with l1 and l2 regularization.
        """
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
        """
        Format lower and upper bounds
        """
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

    @staticmethod
    def _solve_problem(problem: cp.Problem, w: cp.Variable) -> tuple[float, np.ndarray]:
        """
        Solve CVXPY Problem without variables
        """
        try:
            problem.solve(solver='ECOS')
            if w.value is None:
                raise OptimizationError('None return')
            weights = w.value
            result = problem.value
        except (OptimizationError, SolverError, ArpackNoConvergence) as e:
            logger.warning(f'No solution found: {e}')
            weights = np.empty(w.shape) * np.nan
            result = np.nan

        return result, weights

    @staticmethod
    def _get_optimization_weights(problem: cp.Problem,
                                  w: cp.Variable,
                                  parameter: cp.Parameter,
                                  target: Union[float, np.ndarray]) -> np.ndarray:
        """
        Solve CVXPY Problem with variables
        :param problem: CVXPY Problem
        :param w: CVXPY Variable representing the weights
        :param parameter: CVXPY Parameter
        :param target: parameter's value(s)
        :returns: weights array
        """
        if np.isscalar(target):
            parameter_array = [target]
        else:
            parameter_array = target

        weights = []
        for value in parameter_array:
            parameter.value = value
            try:
                problem.solve(solver='ECOS')
                if w.value is None:
                    raise OptimizationError('None return')
                weights.append(w.value)
            except (OptimizationError, SolverError, ArpackNoConvergence) as e:
                logger.warning(f'No solution found for {value:e}: {e}')
                weights.append(np.empty(w.shape) * np.nan)

        if np.isscalar(target):
            weights = weights[0]

        return np.array(weights)

    def _get_investment_target(self) -> Optional[int]:
        """
        Convert the investment target into 0, 1 or None
        """
        # Sum of weights
        if self.investment_type == InvestmentType.FULLY_INVESTED:
            return 1
        elif self.investment_type == InvestmentType.MARKET_NEUTRAL:
            return 0

    def maximum_sharpe(self) -> np.ndarray:
        """
        Find the asset weights that maximize the portfolio sharpe ratio
        :returns: the asset weights that maximize the portfolio sharpe ratio.
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
        constraints = [self._portfolio_expected_return(w=w) == 1,
                       w >= lower_bounds * k,
                       w <= upper_bounds * k,
                       cp.sum(w) == k,
                       k >= 0]

        # Problem
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver='ECOS')
            if w.value is None or k.value is None:
                raise OptimizationError('None return')
            weights = np.array(w.value / k.value, dtype=float)
        except (OptimizationError, SolverError, ArpackNoConvergence) as e:
            logger.warning(f'No solution found: {e}')
            weights = np.empty(w.shape) * np.nan

        return weights

    def minimum_variance(self) -> tuple[float, np.ndarray]:
        """
        Find the asset weights that minimize the portfolio variance and the value of the minimum
        variance.
        :returns: the tuple (minimum variance, weights of the minimum variance portfolio)
        """
        # Variables
        w = cp.Variable(self.assets.asset_nb)

        # Objectives
        portfolio_variance = cp.quad_form(w, self.assets.expected_cov)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        constraints = [w >= lower_bounds,
                       w <= upper_bounds]
        investment_target = self._get_investment_target()
        if investment_target is not None:
            constraints.append(cp.sum(w) == investment_target)

        # Problem
        problem = cp.Problem(objective, constraints)

        min_variance, weights = self._solve_problem(problem=problem, w=w)

        return min_variance, weights

    def minimum_semivariance(self,
                             returns_target: Optional[Union[float, np.ndarray]] = None) -> tuple[float, np.ndarray]:
        """
        Find the asset weights that minimize the portfolio semivariance (downside variance) and the value of the minimum
        semivariance
        :param returns_target: the target(s) to distinguish "downside" and "upside" returns
        :type returns_target: float or np.ndarray of shape(Number of Assets)
        :returns: the tuple (minimum semivariance, weights of the minimum semivariance portfolio)
        """
        if returns_target is None:
            returns_target = self.assets.expected_returns

        # Additional matrix
        if not np.isscalar(returns_target) and returns_target.shape != (len(returns_target), 1):
            returns_target = returns_target[:, np.newaxis]
        b = (self.assets.returns - returns_target) / np.sqrt(self.assets.date_nb)

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        p = cp.Variable(self.assets.date_nb, nonneg=True)
        n = cp.Variable(self.assets.date_nb, nonneg=True)

        # Objectives
        portfolio_semivariance = cp.sum(cp.square(n))
        objective = cp.Minimize(portfolio_semivariance)

        # Constraints
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        constraints = [b.T @ w - p + n == 0,
                       w >= lower_bounds,
                       w <= upper_bounds]

        investment_target = self._get_investment_target()
        if investment_target is not None:
            constraints.append(cp.sum(w) == investment_target)

        # Problem
        problem = cp.Problem(objective, constraints)

        min_semivariance, weights = self._solve_problem(problem=problem, w=w)

        return min_semivariance, weights

    def minimum_cvar(self, beta: float = 0.95) -> tuple[float, np.ndarray]:
        """
        Find the asset weights that minimize the portfolio CVaR (Conditional Value-at-Risk or Expected Shortfall)
        and the value of the minimum CVaR.
        The CVaR is the average of the “extreme” losses beyond the VaR threshold
        :param beta: VaR confidence level (expected VaR on the worst (1-beta)% days)
        :type beta: float
        :returns: the tuple (minimum CVaR, weights of the minimum CVaR portfolio)
        """

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        alpha = cp.Variable()
        u = cp.Variable(self.assets.date_nb)

        # Objectives
        portfolio_cvar = alpha + 1.0 / (self.assets.date_nb * (1 - beta)) * cp.sum(u)
        objective = cp.Minimize(portfolio_cvar)

        # Constraints
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()

        constraints = [self.assets.returns.T @ w + alpha + u >= 0,
                       u >= 0,
                       w >= lower_bounds,
                       w <= upper_bounds]

        investment_target = self._get_investment_target()
        if investment_target is not None:
            constraints.append(cp.sum(w) == investment_target)

        # Problem
        problem = cp.Problem(objective, constraints)

        min_cvar, weights = self._solve_problem(problem=problem, w=w)

        return min_cvar, weights

    def minimum_cdar(self, beta: float = 0.95) -> tuple[float, np.ndarray]:
        """
        Find the asset weights that minimize the portfolio CDaR (Conditional Drawdown-at-Risk)
        and the value of the minimum CDaR.
        The CDaR is the average drawdown for all the days that drawdown exceeds a threshold
        :param beta: drawdown confidence level (expected drawdown on the worst (1-beta)% days)
        :type beta: float
        :returns: the tuple (minimum CDaR, weights of the minimum CDaR portfolio)
       """

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        alpha = cp.Variable()
        u = cp.Variable(self.assets.date_nb + 1)
        z = cp.Variable(self.assets.date_nb)

        # Objectives
        portfolio_cdar = alpha + 1.0 / (self.assets.date_nb * (1 - beta)) * cp.sum(z)
        objective = cp.Minimize(portfolio_cdar)

        # Constraints
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()

        constraints = [z >= u[1:] - alpha,
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

        min_cdar, weights = self._solve_problem(problem=problem, w=w)

        return min_cdar, weights

    def mean_variance(self,
                      target_variance: Optional[Union[float, list, np.ndarray]] = None,
                      population_size: Optional[int] = None,
                      l1_coef: Optional[float] = None,
                      l2_coef: Optional[float] = None) -> np.ndarray:
        """
        Optimization along the mean-variance frontier (Markowitz optimization)
        :param target_variance: the targeted daily variance of the portfolio: the portfolio expected return is maximized
        under this target constraint
        :type target_variance: float or list or numpy.ndarray optional
        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int, optional
        :param l1_coef: L1 regularisation coefficient. Increasing this coef will reduce the number of non-zero weights.
                        It is similar to the L1 regularisation in Lasso.
                        If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation
                        in Elastic-Net
        :type l1_coef: float, default to None
        :param l2_coef: L2 regularisation coefficient. It is similar to the L2 regularisation in Ridge.
                        If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation in
                        Elastic-Net
        :type l1_coef: float, default to None
        :returns: the portfolio weights that are in the efficient frontier.
        :rtype: numpy.ndarray of shape (asset number,) if target is a scalar,
                otherwise numpy.ndarray of shape (population size, asset number) or (len(target_variance), asset number)
        """
        self._validate_args(**{k: v for k, v in locals().items() if k != 'self'})

        # Variables
        w = cp.Variable(self.assets.asset_nb)

        # Parameters
        target_variance_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = cp.Maximize(self._portfolio_expected_return(w=w, l1_coef=l1_coef, l2_coef=l2_coef))

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

        # Target
        if target_variance is not None:
            if not np.isscalar(target_variance):
                target_variance = np.array(target_variance)
        else:
            min_variance, _ = self.minimum_variance()
            if np.isnan(min_variance):
                raise OptimizationError(f'Unable to find the minimum variance portfolio used as the starting point '
                                        f'of the pareto frontier --> you can input your own target_variance array')
            max_variance = max(0.4 ** 2 / 255, min_variance * 10)  # max(40% annualized volatility, 10 x min variance)
            target_variance = np.logspace(np.log10(min_variance), np.log10(max_variance), num=population_size)

        # weights
        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_variance_param,
                                                 target=target_variance)

        return weights

    def mean_semivariance(self,
                          returns_target: Optional[Union[float, np.ndarray]] = None,
                          target_semivariance: Optional[Union[float, list, np.ndarray]] = None,
                          population_size: Optional[int] = None,
                          l1_coef: Optional[float] = None,
                          l2_coef: Optional[float] = None) -> np.ndarray:
        """
         Optimization along the mean-semivariance frontier
        :param returns_target: the target(s) to distinguish "downside" and "upside" returns
        :type returns_target: float or np.ndarray of shape(Number of Assets)
        :param target_semivariance: the targeted daily semivariance (downside variance) of the portfolio:
        the portfolio expected return is maximized under this target constraint
        :type target_semivariance: float or list or numpy.ndarray optional
        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int, optional
        :param l1_coef: L1 regularisation coefficient. Increasing this coef will reduce the number of non-zero weights.
                        It is similar to the L1 regularisation in Lasso.
                        If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation
                        in Elastic-Net
        :type l1_coef: float, default to None
        :param l2_coef: L2 regularisation coefficient. It is similar to the L2 regularisation in Ridge.
                        If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation in
                        Elastic-Net
        :type l1_coef: float, default to None
        :returns: the portfolio weights that are in the efficient frontier.
        :rtype: numpy.ndarray of shape (asset number,) if the target is a scalar, otherwise numpy.ndarray of
        shape (population size, asset number) or (len(target_semivariance), asset number)
        """
        self._validate_args(**{k: v for k, v in locals().items() if k != 'self'})

        if returns_target is None:
            returns_target = self.assets.expected_returns

        # Additional matrix
        if not np.isscalar(returns_target) and returns_target.shape != (len(returns_target), 1):
            returns_target = returns_target[:, np.newaxis]
        b = (self.assets.returns - returns_target) / np.sqrt(self.assets.date_nb)

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        p = cp.Variable(self.assets.date_nb, nonneg=True)
        n = cp.Variable(self.assets.date_nb, nonneg=True)

        # Parameters
        target_semivariance_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = cp.Maximize(self._portfolio_expected_return(w=w, l1_coef=l1_coef, l2_coef=l2_coef))

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

        # Target
        if target_semivariance is not None:
            if not np.isscalar(target_semivariance):
                target_semivariance = np.array(target_semivariance)
        else:
            min_semivariance, _ = self.minimum_semivariance(returns_target=returns_target)
            if np.isnan(min_semivariance):
                raise OptimizationError(f'Unable to find the minimum semivariance portfolio used as the starting point '
                                        f'of the pareto frontier --> you can input your own target_semivariance array')
            max_semivariance = max(0.4 ** 2 / 255, min_semivariance * 10)  # 40% annualized semivolatility
            target_semivariance = np.logspace(np.log10(min_semivariance),
                                              np.log10(max_semivariance),
                                              num=population_size)

        # weights
        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_semivariance_param,
                                                 target=target_semivariance)

        return weights

    def mean_cvar(self,
                  beta: float = 0.95,
                  target_cvar: Optional[Union[float, list, np.ndarray]] = None,
                  population_size: Optional[int] = None,
                  l1_coef: Optional[float] = None,
                  l2_coef: Optional[float] = None) -> np.ndarray:
        """
        Optimization along the mean-CVaR frontier (Conditional Value-at-Risk or Expected Shortfall).
        The CVaR is the average of the “extreme” losses beyond the VaR threshold
        :param beta: VaR confidence level (expected VaR on the worst (1-beta)% days)
        :type beta: float
        :param target_cvar: the targeted CVaR of the portfolio: the portfolio expected return is maximized under this
        target constraint
        :type target_cvar: float or list or numpy.ndarray optional
        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int, optional
        :param l1_coef: L1 regularisation coefficient. Increasing this coef will reduce the number of non-zero weights.
                        It is similar to the L1 regularisation in Lasso.
                        If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation
                        in Elastic-Net
        :type l1_coef: float, default to None
        :param l2_coef: L2 regularisation coefficient. It is similar to the L2 regularisation in Ridge.
                        If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation in
                        Elastic-Net
        :type l1_coef: float, default to None
        :returns: the portfolio weights that are in the efficient frontier
        :rtype: numpy.ndarray of shape (asset number,) if the target is a scalar, otherwise numpy.ndarray of
        shape (population size, asset number) or (len(target_cvar), asset number)
        """

        self._validate_args(**{k: v for k, v in locals().items() if k != 'self'})

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        alpha = cp.Variable()
        u = cp.Variable(self.assets.date_nb)

        # Parameters
        target_cvar_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = cp.Maximize(self._portfolio_expected_return(w=w, l1_coef=l1_coef, l2_coef=l2_coef))

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

        # Target
        if target_cvar is not None:
            if not np.isscalar(target_cvar):
                target_cvar = np.array(target_cvar)
        else:
            min_cvar, _ = self.minimum_cvar(beta=beta)
            if np.isnan(min_cvar):
                raise OptimizationError(f'Unable to find the minimum CVaR portfolio used as the starting point '
                                        f'of the pareto frontier --> you can input your own target_cvar array')
            max_cvar = max(0.3, min_cvar * 10)  # 30% CVaR
            target_cvar = np.logspace(np.log10(min_cvar), np.log10(max_cvar), num=population_size)

        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_cvar_param,
                                                 target=target_cvar)

        return weights

    def mean_cdar(self,
                  beta: float = 0.95,
                  target_cdar: Optional[Union[float, list, np.ndarray]] = None,
                  population_size: Optional[int] = None,
                  l1_coef: Optional[float] = None,
                  l2_coef: Optional[float] = None) -> np.ndarray:
        """
        Optimization along the mean-CDaR frontier (Conditional Drawdown-at-Risk).
        The CDaR is the average drawdown for all the days that drawdown exceeds a threshold
        :param beta: drawdown confidence level (expected drawdown on the worst (1-beta)% days)
        :type beta: float
        :param target_cdar: the targeted CDaR of the portfolio: the portfolio expected return is maximized under this
        target constraint
        :type target_cdar: float or list or numpy.ndarray optional
        :param population_size: number of pareto optimal portfolio weights to compute along the efficient frontier
        :type population_size: int, optional
        :param l1_coef: L1 regularisation coefficient. Increasing this coef will reduce the number of non-zero weights.
                        It is similar to the L1 regularisation in Lasso.
                        If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation
                        in Elastic-Net
        :type l1_coef: float, default to None
        :param l2_coef: L2 regularisation coefficient. It is similar to the L2 regularisation in Ridge.
                        If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation in
                        Elastic-Net
        :type l1_coef: float, default to None
        :returns: the portfolio weights that are in the efficient frontier
        :rtype: numpy.ndarray of shape (asset number,) if the target is a scalar, otherwise numpy.ndarray of
        shape (population size, asset number) or (len(target_cdar), asset number)
        """

        self._validate_args(**{k: v for k, v in locals().items() if k != 'self'})

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        alpha = cp.Variable()
        u = cp.Variable(self.assets.date_nb + 1)
        z = cp.Variable(self.assets.date_nb)

        # Parameters
        target_cdar_param = cp.Parameter(nonneg=True)

        # Objectives
        objective = cp.Maximize(self._portfolio_expected_return(w=w, l1_coef=l1_coef, l2_coef=l2_coef))

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

        # Target
        if target_cdar is not None:
            if not np.isscalar(target_cdar):
                target_cdar = np.array(target_cdar)
        else:
            min_cdar, _ = self.minimum_cdar(beta=beta)
            if np.isnan(min_cdar):
                raise OptimizationError(f'Unable to find the minimum CDaR portfolio used as the starting point '
                                        f'of the pareto frontier --> you can input your own target_cdar array')
            max_cdar = max(0.3, min_cdar * 10)  # 30% CDaR
            target_cdar = np.logspace(np.log10(min_cdar), np.log10(max_cdar), num=population_size)

        weights = self._get_optimization_weights(problem=problem,
                                                 w=w,
                                                 parameter=target_cdar_param,
                                                 target=target_cdar)

        return weights

    def inverse_volatility(self) -> np.ndarray:
        """
        Inverse volatility portfolio
        :returns: assets weights of the inverse volatility portfolio, summing to 1
        """
        weights = 1 / self.assets.std
        weights = weights / sum(weights)
        return weights

    def equal_weighted(self) -> np.ndarray:
        """
        Equally weighted portfolio
        :returns: assets weights of the equally weighted portfolio, summing to 1
        """
        weights = np.ones(self.assets.asset_nb) / self.assets.asset_nb
        return weights

    def random(self) -> np.ndarray:
        """
        Randomly weighted portfolio
        :returns:  Random positive weights summing to 1 that respect the bounds constraints
        """
        # Produces n random weights that sum to 1 with uniform distribution over the simplex
        weights = rand_weights_dirichlet(n=self.assets.asset_nb)
        # Respecting bounds
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        weights = np.minimum(np.maximum(weights, lower_bounds), upper_bounds)
        weights = weights / sum(weights)
        return weights
