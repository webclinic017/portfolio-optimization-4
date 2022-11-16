import logging
import numpy as np
import cvxpy as cp
from cvxpy import SolverError
from cvxpy.constraints.constraint import Constraint
from scipy.sparse.linalg import ArpackNoConvergence
from enum import Enum, unique

from portfolio_optimization.meta import *
from portfolio_optimization.assets import *
from portfolio_optimization.exception import *
from portfolio_optimization.utils.tools import *

__all__ = ['Optimization',
           'RiskMeasure',
           'ObjectiveFunction']

logger = logging.getLogger('portfolio_optimization.optimization')


class RiskMeasure(Enum):
    VARIANCE = 'variance'
    SEMI_VARIANCE = 'semi_variance'
    CVAR = 'cvar'
    CDAR = 'cdar'


class ObjectiveFunction(Enum):
    MIN_RISK = 'min_risk'
    MAX_RETURN = 'max_return'
    RATIO = 'ratio'
    UTILITY = 'utility'


WeightBounds = tuple[float | np.ndarray | None, float | np.ndarray | None]
Costs = float | np.ndarray
Duration = int | None
PrevWeight = np.ndarray | None
Rate = float
Solvers = list[str] | None
solverParams = dict | None
Coef = float | None
Target = float | list | np.ndarray | None
ParametersValues = list[tuple[cp.Parameter, float | np.ndarray]] | None
PopulationSize = int | None
OptionalVariable = cp.Variable | None
Result = np.ndarray | tuple[float | np.ndarray, np.ndarray]


class Optimization:
    def __init__(self,
                 assets: Assets,
                 investment_type: InvestmentType = InvestmentType.FULLY_INVESTED,
                 weight_bounds: WeightBounds = (-1, 1),
                 transaction_costs: Costs = 0,
                 investment_duration: Duration = None,
                 previous_weights: PrevWeight = None,
                 risk_free_rate: Rate = 0,
                 logarithmic_returns: bool = False,
                 solvers: Solvers = None,
                 solver_params: solverParams = None):
        r"""
        Class for convex portfolio optimization

       Parameters
       ----------
        assets: Assets
                The Assets object containing the assets market data and mean/covariance models

        investment_type: InvestmentType, default InvestmentType.FULLY_INVESTED
                         The investment type of the portfolio.

                         The possible values are:
                          * fully invested: the sum of weights equal one
                          * market neutral: the sum of weights equal zero
                          * unconstrained: the sum of weights has no constraints

        weight_bounds: tuple[float | ndarray | None, float | ndarray | None], default (-1, 1)
                       Minimum and maximum weight of each asset OR single min/max pair if they are all identical.
                       A value of None for that lower bound is equivalent to -Inf (no lower bound). And a value
                       of None for the upper bound is equivalent to +Inf (no upper bound).

                       Example:
                            * no short selling (long only portfolio): (0, None)
                            * short only: (None, 0)
                            * no bound (0, 0)
                            * no short selling with maximum weight of 200%: (0, 2)
                            * all weights between -100% and 100% of the total notional: (-1, 1)

        transaction_costs: float or ndarray of shape (number_of_assets), default 0
                           Transaction costs are fixed costs charged when buying or selling an asset.
                           When that value is different from 0, you also have to provide `investment_duration` and
                           `previous_weights`.
                           They are used to compute the cost of rebalancing the portfolio which is:
                                * transaction_costs / investment_duration * |previous_weights - new_weights|
                           The costs is expressed in the same time unit as the returns (see investment_duration for more
                           details).

                      Example:
                              * if your broker charges you 0.5% for asset A and 0.6% for Asset B on the notional amount
                              that you buy or sell, you have to input a transaction_costs of [0.005, 0.006]
                              * if your broker charges you 3.1 EUR per share of asset A and 4.8 EUR per share of
                              asset B, you have to convert it into percentage of notional amount and input a
                              transaction_costs of [3.1 / price_asset_A, 4.8 / price_asset_B]

        investment_duration: The expected investment duration expressed in the same period as the returns.
                  If the returns are daily, then it is the expected investment duration in business days.
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

        previous_weight: np.ndarray of shape(Number of Assets), default None (equivalent to an array of zeros)
        :type
        """
        self.assets = assets
        self.investment_type = investment_type
        self.weight_bounds = weight_bounds
        self.transaction_costs = transaction_costs
        self.risk_free_rate = risk_free_rate
        self.logarithmic_returns = logarithmic_returns
        self.investment_duration = investment_duration
        self.previous_weights = previous_weights
        if solvers is None:
            self.solvers = ['ECOS', 'SCS', 'OSQP', 'CVXOPT']
        else:
            self.solvers = solvers
        if solver_params is None:
            solver_params = {}
        self.solver_params = {k: solver_params.get(k, {}) for k in self.solvers}
        self.N = 1000
        self.loaded = True
        self._validation()

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

        if self.transaction_costs is not None and not (np.isscalar(self.transaction_costs) and self.transaction_costs == 0):
            if self.investment_duration is None:
                raise ValueError(f'investment_duration_in_days cannot be missing when costs is provided')

        if self.previous_weights is not None:
            if not isinstance(self.previous_weights, np.ndarray):
                raise TypeError(f'prev_w should be of type numpy.ndarray')
            if len(self.previous_weights) != self.assets.asset_nb:
                raise ValueError(f'prev_w should be of size {self.assets.asset_nb} but received {len(self.previous_weights)}')

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
                                   l1_coef: Coef = None,
                                   l2_coef: Coef = None,
                                   k: OptionalVariable = None,
                                   gr: OptionalVariable = None) -> cp.Expression:
        """
        CVXPY Expression of the portfolio expected return with l1 and l2 regularization.
        """
        if self.transaction_costs is None or (np.isscalar(self.transaction_costs) and self.transaction_costs == 0):
            portfolio_cost = 0
        else:
            if self.previous_weights is None:
                prev_w = np.zeros(self.assets.asset_nb)
            else:
                prev_w = self.previous_weights
            daily_costs = self.transaction_costs / self.investment_duration
            if np.isscalar(daily_costs):
                portfolio_cost = daily_costs * cp.norm(prev_w - w, 1)
            else:
                portfolio_cost = cp.norm(cp.multiply(daily_costs, (prev_w - w)), 1)

        # Norm L1
        if l1_coef is None or l1_coef == 0:
            l1_regularization = 0
        else:
            # noinspection PyTypeChecker
            l1_regularization = l1_coef * cp.norm(w, 1)

        # Norm L2
        if l2_coef is None or l2_coef == 0:
            l2_regularization = 0
        else:
            l2_regularization = l2_coef * cp.sum_squares(w)

        if self.logarithmic_returns:
            if k is not None:
                # noinspection PyTypeChecker
                ret = 1 / self.assets.date_nb * cp.sum(gr) - self.risk_free_rate * k
            else:
                ret = 1 / self.assets.date_nb * cp.sum(cp.log(1 + self.assets.returns @ w))
        else:
            ret = self.assets.expected_returns @ w

        portfolio_return = ret - portfolio_cost - l1_regularization - l2_regularization
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

    def _get_investment_target(self) -> int | None:
        """
        Convert the investment target into 0, 1 or None
        """
        # Sum of weights
        if self.investment_type == InvestmentType.FULLY_INVESTED:
            return 1
        elif self.investment_type == InvestmentType.MARKET_NEUTRAL:
            return 0

    def _get_weight_constraints(self, w: cp.Variable, k: OptionalVariable = None) -> list[Constraint]:
        lower_bounds, upper_bounds = self._get_lower_and_upper_bounds()
        if k is None:
            constraints = [w >= lower_bounds,
                           w <= upper_bounds]
        else:
            constraints = [w >= lower_bounds * k,
                           w <= upper_bounds * k]

        investment_target = self._get_investment_target()
        if investment_target is not None:
            if k is None:
                constraints.append(cp.sum(w) == investment_target)
            else:
                # noinspection PyTypeChecker
                constraints.append(cp.sum(w) == investment_target * k)

        return constraints

    def _solve_problem(self,
                       problem: cp.Problem,
                       w: cp.Variable,
                       parameters_values: ParametersValues = None,
                       k: OptionalVariable = None,
                       objective_values: bool = False) -> Result:
        """
        Solve CVXPY Problem with variables
        :param problem: CVXPY Problem
        :param w: CVXPY Variable representing the weights
        :returns: weights array
        """
        n = 0
        is_scalar = True
        if parameters_values is None or len(parameters_values) == 0:
            new_parameters_values = [(None, None)]
        else:
            new_parameters_values = []
            prev_n = None
            for parameter, value in parameters_values:
                if np.isscalar(value):
                    n = 0
                    new_parameters_values.append((parameter, [value]))
                else:
                    is_scalar = False
                    n = len(value)
                    new_parameters_values.append((parameter, value))
                if prev_n is not None and prev_n != n:
                    raise ValueError(f'All elements from parameters_values should have same length')
                prev_n = n

        weights = []
        results = []
        n = max(1, n)
        for i in range(n):
            for parameter, values in new_parameters_values:
                if parameter is not None:
                    parameter.value = values[i]
            for solver in self.solvers:
                try:
                    problem.solve(solver=solver, **self.solver_params[solver])
                except (SolverError, ArpackNoConvergence) as e:
                    logger.warning(f'Solver {solver} failed with error: {e}')
                    pass
                if w.value is not None:
                    break
            if w.value is None:
                raise OptimizationError('No solution found')

            if k is None:
                weights.append(np.array(w.value, dtype=float))
            else:
                weights.append(np.array(w.value / k.value, dtype=float))

            results.append(problem.value / self.N)

        if is_scalar:
            weights = weights[0]
            results = results[0]
        else:
            weights = np.array(weights, dtype=float)
            results = np.array(results, dtype=float)

        if objective_values:
            return results, weights
        else:
            return weights

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

    def optimize(self,
                 risk_measure: RiskMeasure,
                 objective_function: ObjectiveFunction,
                 gamma: float = 2,
                 l1_coef: Coef = None,
                 l2_coef: Coef = None,
                 min_return: Target = None,
                 max_variance: Target = None,
                 max_semi_variance: Target = None,
                 semi_variance_returns_target: Target = None,
                 max_cvar: Target = None,
                 cvar_beta: float = 0.95,
                 max_cdar: Target = None,
                 cdar_beta: float = 0.95,
                 objective_values: bool = False) -> Result:

        if not isinstance(risk_measure, RiskMeasure):
            raise TypeError('risk_measure should be of type RiskMeasure')
        if not isinstance(objective_function, ObjectiveFunction):
            raise TypeError('objective_function should be of type ObjectiveFunction')

        w = cp.Variable(self.assets.asset_nb)
        n = self.assets.date_nb

        constraints = []

        if objective_function == ObjectiveFunction.RATIO:
            k = cp.Variable()
            gr = cp.Variable(self.assets.date_nb)
        else:
            k = None
            gr = None

        ret = self._portfolio_expected_return(w=w,
                                              l1_coef=l1_coef,
                                              l2_coef=l2_coef,
                                              k=k,
                                              gr=gr)

        # min_return constraint
        if min_return is not None:
            if k is None:
                # noinspection PyTypeChecker
                constraints.append(ret >= min_return)
            else:
                # noinspection PyTypeChecker
                constraints.append(ret >= min_return * k)

        # weight constraints
        constraints += self._get_weight_constraints(w=w, k=k)

        # risk and risk constraints
        risk = None
        parameters_values = []
        for r_m in RiskMeasure:
            risk_limit = locals()[f'max_{r_m.value}']
            # risk_limit = {'max_semi_variance': max_semi_variance}.get(f'max_{r_m.value}')
            if risk_measure == r_m or risk_limit is not None:
                risk_func = getattr(self, f'_{r_m.value}_risk')
                args = {}
                for arg_name in risk_func.__code__.co_varnames[:risk_func.__code__.co_argcount]:
                    if arg_name == 'self':
                        pass
                    elif arg_name == 'w':
                        args[arg_name] = w
                    elif arg_name == 'k':
                        args[arg_name] = k
                    else:
                        args[arg_name] = locals()[arg_name]
                risk_i, constraints_i = risk_func(**args)
                constraints += constraints_i
                if risk_limit is not None:
                    parameter = cp.Parameter(nonneg=True)
                    if k is not None:
                        # noinspection PyTypeChecker
                        constraints += [risk_i <= parameter * k]
                    else:
                        # noinspection PyTypeChecker
                        constraints += [risk_i <= parameter]
                    parameters_values.append((parameter, risk_limit))
                if risk_measure == r_m:
                    risk = risk_i

        match objective_function:
            case ObjectiveFunction.MAX_RETURN:
                # noinspection PyTypeChecker
                objective = cp.Maximize(ret * self.N)
            case ObjectiveFunction.MIN_RISK:
                objective = cp.Minimize(risk * self.N)
            case ObjectiveFunction.UTILITY:
                objective = cp.Maximize(ret - gamma * risk)
            case ObjectiveFunction.RATIO:
                if self.logarithmic_returns:
                    constraints += [risk <= 1,
                                    cp.constraints.ExpCone(gr, np.ones((n, 1)) @ k, k + self.assets.returns @ w)]
                    # noinspection PyTypeChecker
                    objective = cp.Maximize(ret * self.N)
                else:
                    # noinspection PyTypeChecker
                    constraints += [self._portfolio_expected_return(w=w, l1_coef=l1_coef, l2_coef=l2_coef)
                                    - self.risk_free_rate * k == 1]
                    objective = cp.Minimize(risk * self.N)
            case _:
                raise ValueError(f'objective_function {objective_function} is not valid')

        # problem
        problem = cp.Problem(objective, constraints)

        # results
        return self._solve_problem(problem=problem,
                                   w=w,
                                   k=k,
                                   parameters_values=parameters_values,
                                   objective_values=objective_values)

    def _variance_risk(self, w: cp.Variable):
        v = cp.Variable(nonneg=True)
        z = np.linalg.cholesky(self.assets.expected_cov)
        risk = v ** 2
        constraints = [cp.SOC(v, z.T @ w)]
        return risk, constraints

    def _semi_variance_risk(self, w: cp.Variable,
                            semi_variance_returns_target: Target = None):
        if semi_variance_returns_target is None:
            semi_variance_returns_target = self.assets.expected_returns

            # Additional matrix
        if (not np.isscalar(semi_variance_returns_target)
                and semi_variance_returns_target.shape != (len(semi_variance_returns_target), 1)):
            semi_variance_returns_target = semi_variance_returns_target[:, np.newaxis]

        v = cp.Variable(self.assets.date_nb)
        risk = cp.sum_squares(v) / (self.assets.date_nb - 1)
        # noinspection PyTypeChecker
        constraints = [(self.assets.returns - semi_variance_returns_target).T @ w >= -v,
                       v >= 0]
        return risk, constraints

    def _cvar_risk(self, w: cp.Variable, cvar_beta: float):
        alpha = cp.Variable()
        v = cp.Variable(self.assets.date_nb)
        risk = alpha + 1.0 / (self.assets.date_nb * (1 - cvar_beta)) * cp.sum(v)
        # noinspection PyTypeChecker
        constraints = [self.assets.returns.T @ w + alpha + v >= 0,
                       v >= 0]
        return risk, constraints

    def _cdar_risk(self, w: cp.Variable, cdar_beta: float):
        alpha = cp.Variable()
        v = cp.Variable(self.assets.date_nb + 1)
        z = cp.Variable(self.assets.date_nb)
        risk = alpha + 1.0 / (self.assets.date_nb * (1 - cdar_beta)) * cp.sum(z)
        # noinspection PyTypeChecker
        constraints = [z * self.N >= v[1:] * self.N - alpha * self.N,
                       z * self.N >= 0,
                       v[1:] * self.N >= v[:-1] * self.N - self.assets.returns.T @ w * self.N,
                       v[1:] * self.N >= 0,
                       v[0] * self.N == 0]
        return risk, constraints

    def maximum_sharpe(self) -> np.ndarray:
        """
        Find the asset weights that maximize the portfolio sharpe ratio
        :returns: the asset weights that maximize the portfolio sharpe ratio.
        """

        if self.investment_type != InvestmentType.FULLY_INVESTED:
            raise ValueError('maximum_sharpe() can be solved only for investment_type=InvestmentType.FULLY_INVESTED'
                             '  --> you can find an approximation by computing the efficient frontier with '
                             ' mean_variance(population=30) and finding the portfolio with the highest sharpe ratio.')

        if self.transaction_costs is not None:
            raise ValueError('maximum_sharpe() cannot be solved with costs '
                             '  --> you can find an approximation by computing the efficient frontier with '
                             ' mean_variance(population=30) and finding the portfolio with the highest sharpe ratio.')

        # Variables
        w = cp.Variable(self.assets.asset_nb)
        k = cp.Variable()

        # Objectives
        objective = cp.Minimize(cp.quad_form(w, self.assets.expected_cov))

        # Constraints
        constraints = self._get_weight_constraints(w=w, k=k)
        # noinspection PyTypeChecker
        constraints.extend([self._portfolio_expected_return(w=w) == 1,
                            k >= 0])

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

    def minimum_variance(self, objective_values: bool = False) -> Result:
        """
        Find the asset weights that minimize the portfolio semivariance (downside variance) and the value of the minimum
        semivariance
        :param returns_target: the target(s) to distinguish "downside" and "upside" returns
        :type returns_target: float or np.ndarray of shape(Number of Assets)
        :returns: the tuple (minimum semivariance, weights of the minimum semivariance portfolio)
        """
        return self.optimize(risk_measure=RiskMeasure.VARIANCE,
                             objective_function=ObjectiveFunction.MIN_RISK,
                             objective_values=objective_values)

    def minimum_semivariance(self,
                             semi_variance_returns_target: Target = None,
                             objective_values: bool = False) -> Result:
        """
        Find the asset weights that minimize the portfolio semivariance (downside variance) and the value of the minimum
        semivariance
        :param returns_target: the target(s) to distinguish "downside" and "upside" returns
        :type returns_target: float or np.ndarray of shape(Number of Assets)
        :returns: the tuple (minimum semivariance, weights of the minimum semivariance portfolio)
        """
        return self.optimize(risk_measure=RiskMeasure.SEMI_VARIANCE,
                             objective_function=ObjectiveFunction.MIN_RISK,
                             semi_variance_returns_target=semi_variance_returns_target,
                             objective_values=objective_values)

    def minimum_cvar(self, beta: float = 0.95, objective_values: bool = False) -> Result:
        """
        Find the asset weights that minimize the portfolio CVaR (Conditional Value-at-Risk or Expected Shortfall)
        and the value of the minimum CVaR.
        The CVaR is the average of the “extreme” losses beyond the VaR threshold
        :param beta: VaR confidence level (expected VaR on the worst (1-beta)% days)
        :type beta: float
        :returns: the tuple (minimum CVaR, weights of the minimum CVaR portfolio)
        """

        return self.optimize(risk_measure=RiskMeasure.CVAR,
                             objective_function=ObjectiveFunction.MIN_RISK,
                             cvar_beta=beta,
                             objective_values=objective_values)

    def minimum_cdar(self, beta: float = 0.95, objective_values: bool = False) -> Result:
        """
        Find the asset weights that minimize the portfolio CDaR (Conditional Drawdown-at-Risk)
        and the value of the minimum CDaR.
        The CDaR is the average drawdown for all the days that drawdown exceeds a threshold
        :param beta: drawdown confidence level (expected drawdown on the worst (1-beta)% days)
        :type beta: float
        :returns: the tuple (minimum CDaR, weights of the minimum CDaR portfolio)
       """

        return self.optimize(risk_measure=RiskMeasure.CDAR,
                             objective_function=ObjectiveFunction.MIN_RISK,
                             cvar_beta=beta,
                             objective_values=objective_values)

    def mean_variance(self,
                      target_variance: Target = None,
                      target_return: Target = None,
                      population_size: PopulationSize = None,
                      l1_coef: Coef = None,
                      l2_coef: Coef = None,
                      objective_values: bool = False) -> Result:
        r"""
        Optimization along the mean-variance frontier (Markowitz optimization)

        .. math::
            \begin{align}
            &\underset{w}{\text{optimize}} & & F(w)\\
            &\text{s. t.} & & Aw \geq B\\
            & & & \phi_{i}(w) \leq c_{i}\\
            \end{align}

        .. math::
            \begin{aligned}
            &\underset{w}{\max} & & R (w)\\
            &\text{s.t.} & & Aw \geq B\\
            & & &\phi_{i}(w) \leq c_{i} \; \forall \; i \; \in \; [1,13] \\
            & & & R (w) \geq \overline{\mu}
            \end{aligned}

        .. math::
            \begin{aligned}
            &\underset{w}{\min} & & \phi_{k}(w)\\
            &\text{s.t.} & & Aw \geq B\\
            & & &\phi_{i}(w) \leq c_{i} \; \forall \; i \; \in \; [1,13] \\
            & & & R (w) \geq \overline{\mu}
            \end{aligned}

        Parameters
        ----------
        target_variance: float | list | numpy.ndarray, optional
                         The targeted daily variance of the portfolio: the portfolio expected return is maximized
                         under this target constraint
        population_size: int, optional
                         Number of pareto optimal portfolio weights to compute along the efficient frontier
        l1_coef: float, default to None
                 L1 regularisation coefficient. Increasing this coef will reduce the number of non-zero weights.
                 It is similar to the L1 regularisation in Lasso.
                 If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation in Elastic-Net
        l1_coef: float, default to None
                 L2 regularisation coefficient. It is similar to the L2 regularisation in Ridge.
                 If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation in Elastic-Net


        Returns
        -------
        numpy.ndarray of shape (asset number,) if target is a scalar,
                otherwise numpy.ndarray of shape (population size, asset number) or (len(target_variance), asset number)
        the portfolio weights that are in the efficient frontier.
        """
        # self._validate_args(**{k: v for k, v in locals().items() if k != 'self'})

        if population_size is not None:
            min_variance, _ = self.minimum_variance(objective_values=True)
            max_variance_weight = self.optimize(risk_measure=RiskMeasure.VARIANCE,
                                                objective_function=ObjectiveFunction.MAX_RETURN)
            max_variance = max_variance_weight @ self.assets.expected_cov @ max_variance_weight.T
            target = np.logspace(np.log10(min_variance) + 0.01, np.log10(max_variance), num=population_size)
            return self.optimize(risk_measure=RiskMeasure.VARIANCE,
                                 objective_function=ObjectiveFunction.MAX_RETURN,
                                 l1_coef=l1_coef,
                                 l2_coef=l2_coef,
                                 max_variance=target,
                                 objective_values=objective_values)

        elif target_variance is not None:
            return self.optimize(risk_measure=RiskMeasure.VARIANCE,
                                 objective_function=ObjectiveFunction.MAX_RETURN,
                                 l1_coef=l1_coef,
                                 l2_coef=l2_coef,
                                 max_variance=target_variance,
                                 objective_values=objective_values)

        elif target_return is not None:
            return self.optimize(risk_measure=RiskMeasure.VARIANCE,
                                 objective_function=ObjectiveFunction.MIN_RISK,
                                 l1_coef=l1_coef,
                                 l2_coef=l2_coef,
                                 min_return=target_return,
                                 objective_values=objective_values)

        else:
            return self.minimum_variance(objective_values=objective_values)

    def mean_semivariance(self,
                          target_semivariance: Target = None,
                          target_return: Target = None,
                          population_size: int | None = None,
                          l1_coef: float | None = None,
                          l2_coef: float | None = None,
                          semi_variance_returns_target: Target = None,
                          objective_values: bool = False) -> Result:
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
        # self._validate_args(**{k: v for k, v in locals().items() if k != 'self'})

        if population_size is not None:
            min_semi_variance, _ = self.minimum_semivariance(objective_values=True)
            max_semi_variance_weight = self.optimize(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                                     semi_variance_returns_target=semi_variance_returns_target,
                                                     objective_function=ObjectiveFunction.MAX_RETURN)
            # TODO: max_semi_variance
            target = np.logspace(np.log10(min_semi_variance) + 0.01, np.log10(0.4 ** 2), num=population_size)
            return self.optimize(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                 objective_function=ObjectiveFunction.MAX_RETURN,
                                 l1_coef=l1_coef,
                                 l2_coef=l2_coef,
                                 max_semi_variance=target,
                                 semi_variance_returns_target=semi_variance_returns_target,
                                 objective_values=objective_values)

        elif target_semivariance is not None:
            return self.optimize(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                 objective_function=ObjectiveFunction.MAX_RETURN,
                                 l1_coef=l1_coef,
                                 l2_coef=l2_coef,
                                 max_semi_variance=target_semivariance,
                                 semi_variance_returns_target=semi_variance_returns_target,
                                 objective_values=objective_values)

        elif target_semivariance is not None:
            return self.optimize(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                 objective_function=ObjectiveFunction.MIN_RISK,
                                 l1_coef=l1_coef,
                                 l2_coef=l2_coef,
                                 min_return=target_return,
                                 semi_variance_returns_target=semi_variance_returns_target,
                                 objective_values=objective_values)

        else:
            return self.minimum_semivariance(objective_values=objective_values,
                                             semi_variance_returns_target=semi_variance_returns_target)

    def mean_cvar(self,
                  beta: float = 0.95,
                  target_cvar: float | list | np.ndarray | None = None,
                  population_size: int | None = None,
                  l1_coef: float | None = None,
                  l2_coef: float | None = None) -> Result:
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
        # noinspection PyTypeChecker
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

        weights = self._solve_problem(problem=problem,
                                      w=w,
                                      parameter=target_cvar_param,
                                      target=target_cvar)

        return weights

    def mean_cdar(self,
                  beta: float = 0.95,
                  target_cdar: float | list | np.ndarray | None = None,
                  population_size: int | None = None,
                  l1_coef: float | None = None,
                  l2_coef: float | None = None) -> Result:
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
        # noinspection PyTypeChecker
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

        weights = self._solve_problem(problem=problem,
                                      w=w,
                                      parameter=target_cdar_param,
                                      target=target_cdar)

        return weights
