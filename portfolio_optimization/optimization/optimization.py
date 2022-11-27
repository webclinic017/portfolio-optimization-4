import logging
import numpy as np
import cvxpy as cp
from cvxpy import SolverError, OPTIMAL
from cvxpy.constraints.constraint import Constraint
from scipy.sparse.linalg import ArpackNoConvergence
from enum import Enum
from numbers import Number
from portfolio_optimization.assets import Assets
from portfolio_optimization.exception import OptimizationError
from portfolio_optimization.utils.tools import args_names, rand_weights_dirichlet

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


WeightBounds = tuple[float | np.ndarray | None, float | np.ndarray | None] | None
Costs = float | np.ndarray
Duration = int | None
PrevWeight = np.ndarray | None
Rate = float
Budget = float | None
BudgetBounds = tuple[float | None, float | None]
Solvers = list[str] | None
solverParams = dict[str, dict] | None
Coef = float | None
Target = float | np.ndarray | None
ParametersValues = list[tuple[cp.Parameter, float | np.ndarray]] | None
PopulationSize = int | None
OptionalVariable = cp.Variable | None
Result = np.ndarray | tuple[float | np.ndarray, np.ndarray]


class Optimization:
    def __init__(self,
                 assets: Assets,
                 budget: Budget = 1,
                 budget_bounds: BudgetBounds = None,
                 weight_bounds: WeightBounds = (-1, 1),
                 transaction_costs: Costs = 0,
                 investment_duration: Duration = None,
                 previous_weights: PrevWeight = None,
                 risk_free_rate: Rate = 0,
                 logarithmic_returns: bool = False,
                 solvers: Solvers = None,
                 solver_params: solverParams = None,
                 solver_verbose: bool = False,
                 scale: float = 1):
        self.assets = assets
        r"""
        Convex portfolio optimization

        Parameters
        ----------
        assets: Assets
                The Assets object containing the assets market data and mean/covariance models

        budget: float | None, default 1
                The budget of the portfolio is the sum of long positions and short positions (sum of all weights).

                Examples:
                      * budget=1: fully invested portfolio 
                      * budget=0: market neutral portfolio
                      * budget=None: no constraints on the sum of weights
                      
                      
        budget_bounds: tuple[float | None, float | None] | None, default None
                       The budget bounds of the portfolio are the lower and upper levels of the sum of long positions 
                       and short positions (sum of all weights). If at least one of the budget bounds is not None, 
                       you have to set the budget to None because they contradict each other.

        weight_bounds: tuple[float | ndarray | None, float | ndarray | None] | None, default (-1, 1)
                       Minimum and maximum weight of each asset OR single min/max pair if they are all identical.
                       A value of None for the lower bound is equivalent to -Inf (no lower bound). And a value
                       of None for the upper bound is equivalent to +Inf (no upper bound).

                       Example:
                            * no short selling (long only portfolio): (0, None)
                            * short only: (None, 0)
                            * no bound (0, 0)
                            * no short selling and maximum weight of 200%: (0, 2)
                            * all weights between -100% and 100% of the total notional: (-1, 1)

        transaction_costs: float | ndarray, default 0
                           Transaction costs are fixed costs charged you buy or sell an asset.
                           When that value is different from 0, you also have to provide `investment_duration`.
                           They are used to compute the cost of rebalancing the portfolio which is:
                                * transaction_costs / investment_duration * |previous_weights - new_weights|
                           The costs is expressed in the same time unit as the returns series (see investment_duration
                           for more details).

                      Example:
                              * if your broker charges you 0.5% for asset A and 0.6% for Asset B on the notional amount
                              that you buy or sell, you have to input a transaction_costs of [0.005, 0.006]
                              * if your broker charges you 3.1 EUR per share of asset A and 4.8 EUR per share of
                              asset B, you have to convert them into percentage on the notional amount and input a
                              transaction_costs of [3.1 / price_asset_A, 4.8 / price_asset_B]

        investment_duration: float | None, default None
                             The expected investment duration expressed in the same period as the returns.
                             It needs to be provided when transaction_costs is different from 0.
                             If the returns are daily, then it is the expected investment duration in business days.

                             Examples:
                                * If you expect to keep your portfolio allocation static for one month and your returns
                                are daily, investment_duration = 21
                                * If you expect to keep your portfolio allocation static for one year and your returns
                                are daily, investment_duration = 255
                                * If you expect to keep your portfolio allocation static for one year and your returns
                                are weekly, investment_duration = 52

                             Raison:
                             When costs are provided, they need to be converted to an average cost per time unit over 
                             the expected investment duration.
                             This is because the optimization problem has no notion of investment duration.
                             For example, lets assume that asset A has an expected daily return of 0.01%
                             with a transaction cost of 1% and asset B has an expected daily return of 0.005%
                             with no transaction cost. Let's also assume that both have same volatility and
                             a correlation of 1. If the investment duration is only one month, we should allocate all
                             the weights to asset B whereas if the investment duration is one year, we should allocate
                             all the weights to asset A.
                             
                             Proof:
                             Duration = 1 months (21 business days):
                                    * expected return A = (1+0.01%)^21 - 1 - 1% ≈ 0.01% * 21 - 1% ≈ -0.8%
                                    * expected return B ≈ 0.005% * 21 - 0% ≈ 0.1%
                             Duration = 1 year (255 business days):
                                    * expected return A ≈ 0.01% * 255 - 1% ≈ 1.5%
                                    * expected return B ≈ 0.005% * 21 - 0% ≈ 1.3%
                             
                             In order to take that into account, the costs provided are divided by the expected
                             investment duration in the optimization problem.
                             
        previous_weights: ndarray | None, default None
                          The previous weights of the portfolio. They need to be of same size and same order as the
                          Assets. If transaction_cost is 0, it will have no impact on the portfolio allocation.
        
        risk_free_rate: float, default 0
                        The risk free interest rate to use in the portfolio optimization
        
        logarithmic_returns: bool, default False
                             If True, the optimization uses logarithmic returns instead of simple returns
        
        solvers: list[str] | None, default None
                 The list of cvxpy solver to try. If None, then the default list is ['ECOS', 'SCS', 'OSQP', 'CVXOPT'].
        
        solverParams: dict[str, dict] | None, default None
                      Dictionary of solver parameters with key being the solver name and value the dictionary of 
                      that solver parameter                       
        """
        self.budget = budget
        self.budget_bounds = budget_bounds
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
        self.solver_verbose = solver_verbose
        self.scale = scale
        self.loaded = True
        self._validation()

    def __setattr__(self, name, value):
        if name != 'loaded' and self.__dict__.get('loaded'):
            logger.warning(f'Attributes should be updated with the update() method to allow attribute validation')
        super().__setattr__(name, value)

    def _validation(self):
        r"""
        Validate the class attributes
        """
        if self.assets.asset_nb < 2:
            raise ValueError(f'assets should contains more than one asset')

        for name, bounds in [('weight_bounds', self.weight_bounds, 'budget_bounds', self.budget_bounds)]:
            if bounds is not None:
                if not isinstance(bounds, tuple or len(bounds) != 2):
                    raise ValueError(f'{name} should be a tuple of size 2: (lower_bound, upper_bound)')
                if name == 'weight_bounds':
                    for i in [0, 1]:
                        if isinstance(bounds[i], np.ndarray):
                            if len(bounds[i]) != self.assets.asset_nb:
                                raise ValueError(f'the weight_bounds arrays should be of size {self.assets.asset_nb}, '
                                                 f'but received {len(bounds[i])}')
                        elif bounds[i] is not None and not isinstance(bounds[i], Number):
                            raise TypeError(f'the elements of {name} should be of float or numpy array or None')
                else:
                    for i in [0, 1]:
                        if bounds[i] is not None and not isinstance(bounds[i], Number):
                            raise TypeError(f'the elements of {name} should be float or None')

        if (self.budget is not None
                and self.budget_bounds is not None
                and self.budget_bounds[0] is not None and self.budget_bounds[1] is not None):
            raise ValueError(f'if you provide budget_bounds, you need to set budget to None')

        lower_bounds, upper_bounds = self._get_weights_lower_and_upper_bounds(convert_none=True)
        if not np.all(lower_bounds <= upper_bounds):
            raise ValueError(f'the lower bound should be less or equal than the upper bound')

        if self.budget is not None:
            if sum(upper_bounds) < self.budget:
                raise ValueError(f'the sum of all upper bounds should be greater or equal to the budget: {self.budget}')
            if sum(lower_bounds) > self.budget:
                raise ValueError(f'the sum of all lower bounds should be less or equal to the budget: {self.budget}')
        if np.isscalar(self.transaction_costs):
            if self.transaction_costs < 0:
                raise ValueError(f'transaction_costs cannot be negative')
        else:
            if np.any(self.transaction_costs < 0):
                raise ValueError(f'transaction_costs cannot have negative values')

        if not np.isscalar(self.transaction_costs) or self.transaction_costs != 0:
            if self.investment_duration is None:
                raise ValueError(f'investment_duration cannot be missing when costs is different from 0')

        if self.previous_weights is not None:
            if not isinstance(self.previous_weights, np.ndarray):
                raise TypeError(f'previous_weights should be of type numpy.ndarray')
            if len(self.previous_weights) != self.assets.asset_nb:
                raise ValueError(
                    f'previous_weights should be of size {self.assets.asset_nb} '
                    f'but received {len(self.previous_weights)}')

    def _validate_args(self, **kwargs):
        r"""
        Validate method arguments
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

        lower_bounds, upper_bounds = self._get_weights_lower_and_upper_bounds(convert_none=True)
        for k, v in kwargs.items():
            if k.endswith('coef') and v is not None:
                if v < 0:
                    raise ValueError(f'{k} cannot be negative')
                elif v > 0 and np.all(lower_bounds >= 0):
                    logger.warning(f'Positive {k} will have no impact with positive or null lower bounds')

    def _portfolio_cost(self, w: cp.Variable) -> cp.Expression:
        r"""
        Portfolio cost.

        Returns
        -------
        CVXPY Expression of the portfolio cost
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

        return portfolio_cost

    def _portfolio_expected_return(self,
                                   w: cp.Variable,
                                   l1_coef: Coef = None,
                                   l2_coef: Coef = None,
                                   k: OptionalVariable = None,
                                   gr: OptionalVariable = None) -> cp.Expression:
        r"""
        Portfolio expected return with l1 and l2 regularization.

        Returns
        -------
        CVXPY Expression of the portfolio expected
        """

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

        portfolio_return = ret - self._portfolio_cost(w=w) - l1_regularization - l2_regularization
        return portfolio_return

    def _get_weights_lower_and_upper_bounds(self,
                                            convert_none: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
        r"""
        Convert the float lower and upper bounds into numpy arrays

        Returns
        -------
        Tuple of ndarray of lower and upper bounds
        """
        if self.weight_bounds is None:
            lower_bounds, upper_bounds = None, None
        else:
            lower_bounds, upper_bounds = self.weight_bounds

        if convert_none:
            if lower_bounds is None:
                lower_bounds = -np.Inf
            if upper_bounds is None:
                upper_bounds = np.Inf

        if np.isscalar(lower_bounds):
            lower_bounds = np.array([lower_bounds] * self.assets.asset_nb)
        if np.isscalar(upper_bounds):
            upper_bounds = np.array([upper_bounds] * self.assets.asset_nb)

        return lower_bounds, upper_bounds

    def _get_weight_constraints(self, w: cp.Variable, k: OptionalVariable = None) -> list[Constraint]:
        r"""
        Weight constraints

        Returns
        -------
        A list of weight constraints
        """
        weights_lower_bounds, weights_upper_bounds = self._get_weights_lower_and_upper_bounds()
        constraints = []
        if k is None:
            if weights_lower_bounds is not None:
                constraints.append(w >= weights_lower_bounds)
            if weights_upper_bounds is not None:
                constraints.append(w <= weights_upper_bounds)
        else:
            if weights_lower_bounds is not None:
                constraints.append(w >= weights_lower_bounds * k)
            if weights_upper_bounds is not None:
                constraints.append(w <= weights_upper_bounds * k)

        if self.budget is not None:
            if k is None:
                constraints.append(cp.sum(w) == self.budget)
            else:
                # noinspection PyTypeChecker
                constraints.append(cp.sum(w) == self.budget * k)

        if self.budget_bounds is not None:
            budget_lower_bounds, budget_upper_bounds = self.budget_bounds
            if k is None:
                if budget_lower_bounds is not None:
                    constraints.append(cp.sum(w) >= budget_lower_bounds)
                if budget_upper_bounds is not None:
                    constraints.append(cp.sum(w) <= budget_upper_bounds)
            else:
                if budget_lower_bounds is not None:
                    # noinspection PyTypeChecker
                    constraints.append(cp.sum(w) >= budget_lower_bounds * k)
                if budget_upper_bounds is not None:
                    # noinspection PyTypeChecker
                    constraints.append(cp.sum(w) <= budget_upper_bounds * k)

        return constraints

    def _solve_problem(self,
                       problem: cp.Problem,
                       w: cp.Variable,
                       parameters_values: ParametersValues = None,
                       k: OptionalVariable = None,
                       objective_values: bool = False) -> Result:
        """
        Solve CVXPY Problem

        Parameters
        ----------
        problem: cvxpy.Problem
                 CVXPY Problem.

        w: cvxpy.Variable
           Weights variable.

        parameters_values: list[tuple[cvxpy.Parameter, float | ndarray]] | None, default None
                           A list of tuple of CVXPY Parameter and there values.
                           If The values are ndarray instead of float, the optimization is solved for
                           each element of the array.

        k: cvxpy.Variable | None, default None
           CVXPY Variable used for Ratio optimization problems

        objective_values: bool, default False
                          If true, the optimization objective values are also returned with the weights.

        Returns
        -------
        If objective_values is True:
            tuple (objective values of the optimization problem, weights)
        else:
            weights

        if the parameters values are scalars, then the objective values and weights returned are also scalars otherwise
        they are numpy array of same length as the parameters values.
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
            for j, solver in enumerate(self.solvers):
                try:
                    problem.solve(solver=solver, verbose=self.solver_verbose, **self.solver_params[solver])
                    if w.value is None:
                        raise SolverError('No solution found')
                    if problem.status != OPTIMAL:
                        logger.warning(f'Solution is inaccurate. Try changing solver settings or try another solver or'
                                       f'change the scale')
                    break
                except (SolverError, ArpackNoConvergence) as e:
                    logger.warning(f'Solver {solver} failed with error: {e}')
                    try:
                        logger.info(f'Trying another solver: {self.solvers[j + 1]}')
                    except IndexError:
                        logger.error(f'All Solvers failed, try another list of solvers or '
                                     f'solve with solver_verbose=True for more information')

            if w.value is None:
                raise OptimizationError('No solution found')

            if k is None:
                weights.append(np.array(w.value, dtype=float))
            else:
                weights.append(np.array(w.value / k.value, dtype=float))

            results.append(problem.value / self.scale)

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
        r"""
        Update the class attributes then re-validate them
        """
        self.loaded = False
        valid_kwargs = args_names(self.__init__)
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError(f'Invalid keyword argument {k}. \n'
                                f'Valid arguments are {valid_kwargs}')
            setattr(self, k, v)
        self._validation()
        self.loaded = True

    def inverse_volatility(self) -> np.ndarray:
        r"""
        Inverse volatility portfolio

        Returns
        -------
        Assets weights of the inverse volatility portfolio, summing to 1
        """
        weights = 1 / self.assets.std
        weights = weights / sum(weights)
        return weights

    def equal_weighted(self) -> np.ndarray:
        """
        Equally weighted portfolio

        Returns
        -------
        Assets weights of the equally weighted portfolio, summing to 1
        """
        weights = np.ones(self.assets.asset_nb) / self.assets.asset_nb
        return weights

    def random(self) -> np.ndarray:
        """
        Randomly weighted portfolio

        Returns
        -------
        Random positive weights summing to 1 that respect the weight bounds constraints
        """
        # Produces n random weights that sum to 1 with uniform distribution over the simplex
        weights = rand_weights_dirichlet(n=self.assets.asset_nb)
        # Respecting bounds
        lower_bounds, upper_bounds = self._get_weights_lower_and_upper_bounds(convert_none=True)
        weights = np.minimum(np.maximum(weights, lower_bounds), upper_bounds)
        weights = weights / sum(weights)
        return weights

    def mean_risk_optimization(self,
                               objective_function: ObjectiveFunction,
                               risk_measure: RiskMeasure,
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

        r"""
        General function to solve Mean-Risk optimization problems.

        It takes one objective functions among:
            * minimize risk
            * maximize return
            * maximize the ratio: return/risk
            * maximize the utility: return - gamma x risk
        and one or more risk measures among:
            * variance
            * semi variance
            * CVaR
            * CDaR

        Some combination of objective function /  risk measure(s) may have no solutions.

        Parameters
        ----------
        risk_measure
        objective_function
        gamma
        l1_coef
        l2_coef
        min_return
        max_variance
        max_semi_variance
        semi_variance_returns_target
        max_cvar
        cvar_beta
        max_cdar
        cdar_beta
        objective_values

        Returns
        -------

        """

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
                for arg_name in args_names(risk_func):
                    if arg_name == 'w':
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
                objective = cp.Maximize(ret * self.scale)
            case ObjectiveFunction.MIN_RISK:
                objective = cp.Minimize(risk * self.scale)
            case ObjectiveFunction.UTILITY:
                objective = cp.Maximize(ret - gamma * risk)
            case ObjectiveFunction.RATIO:
                if self.logarithmic_returns:
                    constraints += [risk <= 1,
                                    cp.constraints.ExpCone(gr, np.ones((n, 1)) @ k, k + self.assets.returns @ w)]
                    # noinspection PyTypeChecker
                    objective = cp.Maximize(ret * self.scale)
                else:
                    # noinspection PyTypeChecker
                    constraints += [self._portfolio_expected_return(w=w, l1_coef=l1_coef, l2_coef=l2_coef)
                                    - self.risk_free_rate * k == 1]
                    objective = cp.Minimize(risk * self.scale)
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

        constraints = [self.assets.returns.T @ w - self._portfolio_cost(w=w) + alpha + v >= 0,
                       v >= 0]
        return risk, constraints

    def _cdar_risk(self, w: cp.Variable, cdar_beta: float):
        alpha = cp.Variable()
        v = cp.Variable(self.assets.date_nb + 1)
        z = cp.Variable(self.assets.date_nb)
        risk = alpha + 1.0 / (self.assets.date_nb * (1 - cdar_beta)) * cp.sum(z)
        # noinspection PyTypeChecker
        constraints = [z * self.scale >= v[1:] * self.scale - alpha * self.scale,
                       z * self.scale >= 0,
                       v[1:] * self.scale >= v[:-1] * self.scale - (
                               self.assets.returns.T @ w - self._portfolio_cost(w=w))
                       * self.scale,
                       v[1:] * self.scale >= 0,
                       v[0] * self.scale == 0]
        return risk, constraints

    def minimum_variance(self, objective_values: bool = False) -> Result:
        """
        Find the asset weights that minimize the portfolio semivariance (downside variance) and the value of the minimum
        semivariance
        :param returns_target: the target(s) to distinguish "downside" and "upside" returns
        :type returns_target: float or np.ndarray of shape(Number of Assets)
        :returns: the tuple (minimum semivariance, weights of the minimum semivariance portfolio)
        """
        return self.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
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
        return self.mean_risk_optimization(risk_measure=RiskMeasure.SEMI_VARIANCE,
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

        return self.mean_risk_optimization(risk_measure=RiskMeasure.CVAR,
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

        return self.mean_risk_optimization(risk_measure=RiskMeasure.CDAR,
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

        Parameters
        ----------
        target_variance: float | list | ndarray, optional
                         The targeted variance of the portfolio: the portfolio return is maximized
                         under this target constraint.

        target_return: float | list | ndarray, optional
                       The target return of the portfolio: the portfolio variance is minimized under
                       this target constraint.

        population_size: int, optional
                         Number of pareto optimal portfolio weights to compute along the mean-variance efficient
                         frontier.

        l1_coef: float, optional
                 L1 regularisation coefficient. Increasing this coef will reduce the number of non-zero weights.
                 It is similar to the L1 regularisation in Lasso.
                 If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation in Elastic-Net.

        l2_coef: float, optional
                 L2 regularisation coefficient. It is similar to the L2 regularisation in Ridge.
                 If both l1_coef and l2_coef are strictly positive, it is similar to the regularisation in Elastic-Net.

        objective_values: bool, default False
                          If true, the optimization objective values are also returned with the weights.

        Returns
        -------
        numpy.ndarray of shape (asset number,) if target is a scalar,
                otherwise numpy.ndarray of shape (population size, asset number) or (len(target_variance), asset number)
        the portfolio weights that are in the efficient frontier.
        """
        # self._validate_args(**{k: v for k, v in locals().items() if k != 'self'})

        kwargs = dict(risk_measure=RiskMeasure.VARIANCE,
                      l1_coef=l1_coef,
                      l2_coef=l2_coef)

        if population_size is not None:
            min_variance, _ = self.minimum_variance(objective_values=True)
            max_variance_weight = self.mean_risk_optimization(objective_function=ObjectiveFunction.MAX_RETURN,
                                                              **kwargs)
            max_variance = max_variance_weight @ self.assets.expected_cov @ max_variance_weight.T
            target = np.logspace(np.log10(min_variance) + 0.01, np.log10(max_variance), num=population_size)
            return self.mean_risk_optimization(objective_function=ObjectiveFunction.MAX_RETURN,
                                               max_variance=target,
                                               objective_values=objective_values,
                                               **kwargs)

        elif target_variance is not None:
            return self.mean_risk_optimization(objective_function=ObjectiveFunction.MAX_RETURN,
                                               max_variance=target_variance,
                                               objective_values=objective_values,
                                               **kwargs)

        elif target_return is not None:
            return self.mean_risk_optimization(objective_function=ObjectiveFunction.MIN_RISK,
                                               min_return=target_return,
                                               objective_values=objective_values,
                                               **kwargs)

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
            max_semi_variance_weight = self.mean_risk_optimization(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                                                   semi_variance_returns_target=semi_variance_returns_target,
                                                                   objective_function=ObjectiveFunction.MAX_RETURN)
            # TODO: max_semi_variance
            target = np.logspace(np.log10(min_semi_variance) + 0.01, np.log10(0.4 ** 2), num=population_size)
            return self.mean_risk_optimization(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                               objective_function=ObjectiveFunction.MAX_RETURN,
                                               l1_coef=l1_coef,
                                               l2_coef=l2_coef,
                                               max_semi_variance=target,
                                               semi_variance_returns_target=semi_variance_returns_target,
                                               objective_values=objective_values)

        elif target_semivariance is not None:
            return self.mean_risk_optimization(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                               objective_function=ObjectiveFunction.MAX_RETURN,
                                               l1_coef=l1_coef,
                                               l2_coef=l2_coef,
                                               max_semi_variance=target_semivariance,
                                               semi_variance_returns_target=semi_variance_returns_target,
                                               objective_values=objective_values)

        elif target_semivariance is not None:
            return self.mean_risk_optimization(risk_measure=RiskMeasure.SEMI_VARIANCE,
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

    pass

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

        pass

    def maximum_sharpe(self) -> np.ndarray:
        """
        Find the asset weights that maximize the portfolio sharpe ratio
        :returns: the asset weights that maximize the portfolio sharpe ratio.
        """

        if self.budget != 1:
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
