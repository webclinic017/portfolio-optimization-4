import logging
import numpy as np
import cvxpy as cp
from cvxpy import SolverError, OPTIMAL
from cvxpy.constraints.constraint import Constraint
from scipy.sparse.linalg import ArpackNoConvergence
from numbers import Number
from portfolio_optimization.meta import RiskMeasure, ObjectiveFunction
from portfolio_optimization.assets import Assets
from portfolio_optimization.exception import OptimizationError
from portfolio_optimization.utils.tools import args_names, rand_weights_dirichlet, clean
from portfolio_optimization.optimization.group_constraints import group_constraints_to_matrix

__all__ = ['Optimization']

logger = logging.getLogger('portfolio_optimization.optimization')

Weights = float | np.ndarray | None
AssetGroup = np.ndarray | None
Inequality = np.ndarray | None
GroupConstraints = list[str] | None
Costs = float | np.ndarray
Duration = int | None
Rate = float
Budget = float | None
Solvers = list[str] | None
solverParams = dict[str, dict] | None
Coef = float | None
Target = float | list | np.ndarray | None
ParametersValues = list[tuple[cp.Parameter, float | np.ndarray]] | None
PopulationSize = int | None
OptionalVariable = cp.Variable | None
Scale = float | None
Result = np.ndarray | tuple[float | tuple[float, float] | np.ndarray, np.ndarray]


class Optimization:
    def __init__(self,
                 assets: Assets,
                 min_weights: Weights | list = -1,
                 max_weights: Weights | list = 1,
                 budget: Budget = 1,
                 min_budget: Budget = None,
                 max_budget: Budget = None,
                 max_short: Budget = None,
                 max_long: Budget = None,
                 transaction_costs: Costs | list = 0,
                 investment_duration: Duration = None,
                 previous_weights: Weights = None,
                 asset_groups: AssetGroup = None,
                 group_constraints: GroupConstraints = None,
                 left_inequality: Inequality = None,
                 right_inequality: Inequality = None,
                 risk_free_rate: Rate = 0,
                 is_logarithmic_returns: bool = False,
                 solvers: Solvers = None,
                 solver_params: solverParams = None,
                 solver_verbose: bool = False,
                 scale: Scale = None):
        self.assets = assets
        r"""
        Convex portfolio optimization.

        Parameters
        ----------
        assets: Assets
                The Assets object containing the assets market data and mean/covariance models.
        
        min_weights: float | list | ndarray | None, default -1
                     The minimum weights (weights lower bounds). 
                     If a float is provided, that value will be applied to all the Assets.
                     If a list or array is provided, it has to be of same size and same order as the Assets.
                     A value of None is equivalent to -Inf (no lower bound).

                     Example:
                        * min_weights = 0: long only portfolio (no short selling).
                        * min_weights = None: no lower bound.
                        * min_weights = -2: all weights are above -200% of the total notional.
                        
        max_weights: float | list | ndarray | None, default 1
                     The maximum weights (weights upper bounds). 
                     If a float is provided, that value will be applied to the Assets.
                     If a list or array is provided, it has to be of same size and same order as the Assets.
                     A value of None is equivalent to +Inf (no upper bound).

                     Example:
                        * max_weights = 0: no long position (short only portfolio).
                        * max_weights = None: no upper bound.
                        * max_weights = 2: all weights are below 200% of the total notional.      

        budget: float | None, default 1
                The budget of the portfolio is the sum of long positions and short positions (sum of all weights).
                
                Examples:
                      * budget = 1: fully invested portfolio .
                      * budget = 0: market neutral portfolio.
                      * budget = None: no constraints on the sum of weights.
                      
        min_budget: float, optional
                    The minimum budget of the portfolio. It's the lower bound of the sum of long and short positions 
                    (sum of all weights). 
                    If provided, you have to set the budget to None because they contradict each other.
        
        max_budget: float, optional
                    The maximum budget of the portfolio. It's the upper bound of the sum of long and short positions 
                    (sum of all weights). 
                    If provided, you have to set the budget to None because they contradict each other.
                    
        max_short: float, optional
                   The maximum value of the sum of absolute values of negative weights (short positions).
                        
        max_long: float, optional
                  The maximum value of the sum of positive weights (long positions).

        transaction_costs: float | list | ndarray, default 0
                           Transaction costs are fixed costs charged for buying or selling an asset.
                           When that value is different from 0, you also have to provide `investment_duration`.
                           They are used to compute the cost of rebalancing the portfolio which is:
                                * transaction_costs / investment_duration * |previous_weights - new_weights|
                           The costs is expressed in the same time unit as the returns series (see investment_duration
                           for more details).

                      Example:
                              * transaction_costs = [0.005, 0.006]: the broker transaction costs are 0.5% for asset A 
                                    and 0.6% for Asset B expressed as % on the notional amount invested in the assets.
                              * transaction_costs = [3.1 / price_asset_A, 4.8 / price_asset_B]: the broker transaction 
                                    costs are 3.1 EUR for asset A and 4.8 EUR for asset B expressed in EUR per share. 
                                    Costs per shares have to be converted to percentage on the notional amount.
                              

        investment_duration: float, optional
                             The expected investment duration expressed in the same period as the returns.
                             It needs to be provided when transaction_costs is different from 0.
                             If the returns are daily, then it is the expected investment duration in business days.

                             Examples:
                                * If you expect to keep your portfolio allocation static for one month and your returns
                                    are daily, investment_duration = 21.
                                * If you expect to keep your portfolio allocation static for one year and your returns
                                    are daily, investment_duration = 255.
                                * If you expect to keep your portfolio allocation static for one year and your returns
                                    are weekly, investment_duration = 52.

                             Raison:
                                 When costs are provided, they need to be converted to an average cost per time unit 
                                 over the expected investment duration.
                                 This is because the optimization problem has no notion of investment duration.
                                 For example, lets assume that asset A has an expected daily return of 0.01%
                                 with a transaction cost of 1% and asset B has an expected daily return of 0.005%
                                 with no transaction cost. Let's assume that both have same volatility and
                                 a correlation of 1. If the investment duration is only one month, we should allocate 
                                 all the weights to asset B whereas if the investment duration is one year, we should 
                                 allocate all the weights to asset A.
                             
                             Proof:
                                 Duration = 1 months (21 business days):
                                        * expected return A = (1+0.01%)^21 - 1 - 1% ≈ 0.01% * 21 - 1% ≈ -0.8%
                                        * expected return B ≈ 0.005% * 21 - 0% ≈ 0.1%
                                 Duration = 1 year (255 business days):
                                        * expected return A ≈ 0.01% * 255 - 1% ≈ 1.5%
                                        * expected return B ≈ 0.005% * 21 - 0% ≈ 1.3%
                             
                             In order to take that duration into account, the costs provided are divided by the 
                             expected investment duration.
                             
        previous_weights: float | ndarray | list, optional
                          The previous weights of the portfolio. 
                          If a float is provided, that value will be applied to all the Assets.
                          If a list or array is provided, it has to be of same size and same order as the
                          Assets. If transaction_cost is 0, it will have no impact on the portfolio allocation.
                          
        asset_groups: list[list[str]] | ndarray
                      The list of asset groups. Each group should be a list of same size and order as the Assets. 

            Examples (for 4 assets):
                asset_groups = [['Equity', 'Equity', 'Bond', 'Fund'],
                                ['US', 'Europe', 'Japan', 'US']]

        group_constraints: list[str]
                 The list of constraints applied to the asset groups.
                 They need to respect the following patterns:
                    * 'group_1 <= factor * group_2'
                    * 'group_1 >= factor * group_2'
                    * 'group_1 <= factor'
                    * 'group_1 >= factor'

                factor can be a float or an int.
                group_1 and group_2 are the group names defined in asset_groups.
                The first expression means that the sum of all assets in group_1 should be less or equal to 'factor'
                times the sum of all assets in group_2.

                 Examples:
                    constraints = ['Equity <= 3 * Bond',
                                   'US >= 1.5',
                                   'Europe >= 0.5 * Fund',
                                   'Japan <= 1']
                                   
        left_inequality: ndarray, optional
                         A 2d numpy array of shape (number of constraints, number of assets) representing
                         the matrix :math:`A` of the linear constraint :math:`A \geq B`.
        
        right_inequality: ndarray, optional
                          A 1d numpy array of shape (number of constraints,) representing 
                          the matrix :math:`B` of the linear constraint :math:`A \geq B`.
                         
        risk_free_rate: float, default 0
                        The risk free interest rate to use in the portfolio optimization.
        
        is_logarithmic_returns: bool, default False
                                If True, the optimization uses logarithmic returns instead of simple returns.
        
        solvers: list[str], optional
                 The list of cvxpy solver to try. 
                 If None, then the default list is ['ECOS', 'SCS', 'OSQP', 'CVXOPT'].
        
        solverParams: dict[str, dict], optional
                      Dictionary of solver parameters with key being the solver name and value the dictionary of 
                      that solver parameter.
                      
        scale: float, default None
               The optimization data are scaled by this value. 
               It can be used to increase the optimization accuracies in specific cases.
               If None, 1000 is used for ObjectiveFunction.Ratio and 1 otherwise                  
        """
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.max_short = max_short
        self.max_long = max_long
        self.transaction_costs = transaction_costs
        self.previous_weights = previous_weights
        self.group_constraints = group_constraints
        self.asset_groups = asset_groups
        self.risk_free_rate = risk_free_rate
        self.is_logarithmic_returns = is_logarithmic_returns
        self.investment_duration = investment_duration
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.solvers = ['ECOS', 'SCS', 'OSQP', 'CVXOPT'] if solvers is None else solvers
        if solver_params is None:
            solver_params = {}
        self.solver_params = {k: solver_params.get(k, {}) for k in self.solvers}
        self.solver_verbose = solver_verbose
        self._scale = scale
        self._validation()
        self.loaded = True
        self._default_scale = 1

    @property
    def scale(self) -> float:
        if self._scale is None:
            return self._default_scale
        return self._scale

    @scale.setter
    def scale(self, value: Scale) -> None:
        self._scale = value

    @property
    def min_weights(self) -> Weights:
        return self._min_weights

    @min_weights.setter
    def min_weights(self, value: Weights | list) -> None:
        self._min_weights = clean(value, dtype=float)

    @property
    def max_weights(self) -> Weights:
        return self._max_weights

    @max_weights.setter
    def max_weights(self, value: Weights | list) -> None:
        self._max_weights = clean(value, dtype=float)

    @property
    def previous_weights(self) -> Weights:
        return self._previous_weights

    @previous_weights.setter
    def previous_weights(self, value: Weights | list) -> None:
        self._previous_weights = clean(value, dtype=float)

    @property
    def transaction_costs(self) -> Costs:
        return self._transaction_costs

    @transaction_costs.setter
    def transaction_costs(self, value: Costs | list) -> None:
        self._transaction_costs = clean(value, dtype=float)

    @property
    def asset_groups(self) -> AssetGroup:
        return self._asset_groups

    @asset_groups.setter
    def asset_groups(self, value: AssetGroup | list[list[str]]) -> None:
        if isinstance(value, list) and len(value) != 0:
            g1 = value[0]
            if not isinstance(g1, list):
                raise ValueError(f'asset_groups should be a 2d array or a list of list')
            for g in value:
                if len(g) != len(g1):
                    raise ValueError(f'All groups from asset_groups should be of same size')
        self._asset_groups = clean(value, dtype=str)

    def __setattr__(self, name, value):
        if name != 'loaded' and self.__dict__.get('loaded') and name[:8] != '_default':
            logger.warning(f'Attributes should be updated with the update() method to allow validation')
        super().__setattr__(name, value)

    def _convert_weights_bounds(self, convert_none: bool = False) -> tuple[np.ndarray | None, np.ndarray | None]:
        r"""
        Convert the float lower and upper bounds into numpy arrays
        Returns
        -------
        Tuple of ndarray of lower and upper bounds
        """

        min_weights, max_weights = self.min_weights, self.max_weights

        if convert_none:
            if min_weights is None:
                min_weights = -np.Inf
            if max_weights is None:
                max_weights = np.Inf

        if np.isscalar(min_weights):
            min_weights = np.array([min_weights] * self.assets.asset_nb)
        if np.isscalar(max_weights):
            max_weights = np.array([max_weights] * self.assets.asset_nb)

        return min_weights, max_weights

    def _validation(self):
        r"""
        Validate the class attributes
        """
        if self.assets.asset_nb < 2:
            raise ValueError(f'assets should contains more than one asset')

        for name in ['min_weights', 'max_weights']:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                if len(value) != self.assets.asset_nb:
                    raise ValueError(f'if {name} is an array, it should be of size {self.assets.asset_nb}, '
                                     f'but we received {len(value)}')
            elif value is not None and not isinstance(value, Number):
                raise TypeError(f'{name} should be of type float or list or numpy array or None')

        for name in ['min_budget', 'max_budget']:
            value = getattr(self, name)
            if value is not None and not isinstance(value, Number):
                raise TypeError(f'{name} should be of type float or int or None')

        for name in ['max_short', 'max_long']:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f'{name} must be strictly positif')

        min_weights, max_weights = self._convert_weights_bounds(convert_none=True)
        if not np.all(min_weights <= max_weights):
            raise ValueError(f'min_weights should be less or equal than max_weights')

        if self.budget is not None:
            if self.max_budget is not None:
                raise ValueError(f'if you provide max_budget, you need to set budget to None')
            if self.min_budget is not None:
                raise ValueError(f'if you provide min_budget, you need to set budget to None')
            if sum(max_weights) < self.budget:
                raise ValueError(f'the sum of all max_weights should be greater or equal to the budget: {self.budget}')
            if sum(min_weights) > self.budget:
                raise ValueError(f'the sum of all min_weights should be less or equal to the budget: {self.budget}')

        if self.transaction_costs is not None:
            if np.isscalar(self.transaction_costs):
                if self.transaction_costs < 0:
                    raise ValueError(f'transaction_costs cannot be negative')
            else:
                if self.transaction_costs.shape[0] != self.assets.asset_nb:
                    raise ValueError(f'transaction_costs should be of same size as the number of assets')
                if np.any(self.transaction_costs < 0):
                    raise ValueError(f'transaction_costs cannot have negative values')

            if not np.isscalar(self.transaction_costs) or self.transaction_costs != 0:
                if self.investment_duration is None:
                    raise ValueError(f'investment_duration cannot be missing when costs is different from 0')

        if self.previous_weights is not None:
            if not isinstance(self.previous_weights, (list, np.ndarray)):
                raise TypeError(f'previous_weights should a list or a numpy ndarray')
            if len(self.previous_weights) != self.assets.asset_nb:
                raise ValueError(
                    f'previous_weights should be of size {self.assets.asset_nb} '
                    f'but received {len(self.previous_weights)}')

        if self.asset_groups is not None:
            if len(self.asset_groups.shape) != 2:
                raise ValueError(f'asset_groups should be a 2d array or a list of list')
            if self.asset_groups.shape[1] != self.assets.asset_nb:
                raise ValueError(f'each list or array of asset_groups should have same size as the number of assets')
            if self.group_constraints is None:
                raise ValueError('if you provide asset_groups, you also need to provide group_constraints')

        if self.group_constraints is not None and self.asset_groups is None:
            raise ValueError('if you provide group_constraints, you also need to provide asset_groups')

        if self.left_inequality is not None:
            if not isinstance(self.left_inequality, np.ndarray):
                raise ValueError(f'left_inequality should be of type numpy array')
            if len(self.left_inequality.shape) != 2:
                raise ValueError(f'left_inequality should be a 2d array')
            if self.left_inequality.shape[1] != self.assets.asset_nb:
                raise ValueError(f'left_inequality should be a 2d array with number of columns equal to the number'
                                 f' of assets: {self.assets.asset_nb}')
            if self.right_inequality is None:
                raise ValueError(f'right_inequality should be provided when left_inequality is provided')

        if self.right_inequality is not None:
            if not isinstance(self.right_inequality, np.ndarray):
                raise ValueError(f'right_inequality should be of type numpy array')
            if len(self.right_inequality.shape) != 1:
                raise ValueError(f'right_inequality should be a 1d array')
            if self.left_inequality is None:
                raise ValueError(f'left_inequality should be provided when right_inequality is provided')
            if self.right_inequality.shape[0] != self.left_inequality.shape[0]:
                raise ValueError(f'right_inequality should be an 1d array with number of rows equal to the number'
                                 f'of columns of left_inequality')

    @staticmethod
    def _validate_args(**kwargs):
        r"""
        Validate method arguments
        """
        population_size = kwargs.get('population_size')
        targets_names = [k for k, v in kwargs.items() if k.startswith('target_')]
        not_none_targets_names = [k for k in targets_names if kwargs[k] is not None]
        if len(not_none_targets_names) > 1:
            raise ValueError(f'Only one target has to be provided but received {" AND ".join(not_none_targets_names)}')
        elif len(not_none_targets_names) == 1:
            target_name = not_none_targets_names[0]
        else:
            target_name = None

        if population_size is not None and target_name is not None:
            raise ValueError(f'You have to provide population_size OR {" OR ".join(targets_names)}')

        if population_size is not None and population_size <= 1:
            raise ValueError('f population_size should be strictly greater than one')

        if target_name is not None:
            target = kwargs[target_name]
            if not (np.isscalar(target) or isinstance(target, (list, np.ndarray))):
                raise ValueError(f'{target_name} should be a scalar, ndarray or list '
                                 f'but received {type(target)}')

    def _portfolio_cost(self, w: cp.Variable) -> cp.Expression | float:
        r"""
        Portfolio cost.

        Returns
        -------
        CVXPY Expression of the portfolio cost
        """
        if self.previous_weights is None:
            return 0

        if self.transaction_costs is None:
            return 0

        if np.isscalar(self.transaction_costs) and self.transaction_costs == 0:
            return 0

        if np.isscalar(self.previous_weights):
            prev_w = self.previous_weights * np.ones(self.assets.asset_nb)
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

        if self.is_logarithmic_returns:
            if k is not None:
                # noinspection PyTypeChecker
                ret = 1 / self.assets.date_nb * cp.sum(gr) - self.risk_free_rate * k
            else:
                ret = 1 / self.assets.date_nb * cp.sum(cp.log(1 + self.assets.returns @ w))
        else:
            ret = self.assets.expected_returns @ w

        portfolio_return = ret - self._portfolio_cost(w=w) - l1_regularization - l2_regularization
        return portfolio_return

    def _get_weight_constraints(self, w: cp.Variable, k: OptionalVariable = None) -> list[Constraint]:
        r"""
        Weight constraints

        Returns
        -------
        A list of weight constraints
        """
        min_weights, max_weights = self._convert_weights_bounds()

        if k is not None:
            factor = k
        else:
            factor = 1

        constraints = []
        if min_weights is not None:
            constraints.append(w >= min_weights * factor)
        if max_weights is not None:
            constraints.append(w <= max_weights * factor)
        if self.budget is not None:
            constraints.append(cp.sum(w) == self.budget * factor)
        if self.min_budget is not None:
            constraints.append(cp.sum(w) >= self.min_budget * factor)
        if self.max_budget is not None:
            constraints.append(cp.sum(w) <= self.max_budget * factor)
        if self.max_long is not None:
            constraints.append(cp.sum(cp.pos(w)) * self.scale <= self.max_long * factor * self.scale)
        if self.max_short is not None:
            constraints.append(cp.sum(cp.neg(w)) * self.scale <= self.max_short * factor * self.scale)
        if self.asset_groups is not None and self.group_constraints is not None:
            a, b = group_constraints_to_matrix(groups=self.asset_groups, constraints=self.group_constraints)
            constraints.append(a @ w * self.scale - b * factor * self.scale <= 0)
        if self.left_inequality is not None and self.right_inequality is not None:
            constraints.append(self.left_inequality @ w * self.scale - self.right_inequality * factor * self.scale <= 0)

        return constraints

    def _solve_problem(self,
                       risk_measure: RiskMeasure,
                       problem: cp.Problem,
                       w: cp.Variable,
                       parameters_values: ParametersValues = None,
                       k: OptionalVariable = None,
                       objective_values: bool = False) -> Result:
        """
        Solve CVXPY Problem

        Parameters
        ----------
        risk_measure: RiskMeasure
                      The problem risk measure

        problem: cvxpy.Problem
                 CVXPY Problem.

        w: cvxpy.Variable
           Weights variable.

        parameters_values: list[tuple[cvxpy.Parameter, float | ndarray]], optional
                           A list of tuple of CVXPY Parameter and there values.
                           If The values are ndarray instead of float, the optimization is solved for
                           each element of the array.

        k: cvxpy.Variable, optional
           CVXPY Variable used for Ratio optimization problems

        objective_values: bool, default False
                          If true, the optimization objective values are also returned with the weights.

        Returns
        -------
        If objective_values is True:
            tuple (objective values of the optimization problem, weights)
            if k is not None (Ratio): the objective values are the tuple (mean, risk)
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
                        logger.warning(f'Solution is inaccurate. Try changing the solver settings, try another solver '
                                       f', change the problem scale or or solve with solver_verbose=True')
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
                results.append(problem.value / self.scale)

            else:
                weights.append(np.array(w.value / k.value, dtype=float))
                mean = 1 / k.value
                if risk_measure in [RiskMeasure.CVAR, RiskMeasure.CDAR]:
                    risk = problem.value / k.value / self.scale
                else:
                    risk = problem.value / k.value ** 2 / self.scale
                results.append((mean, risk))

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
        min_weights, max_weights = self._convert_weights_bounds(convert_none=True)
        weights = np.minimum(np.maximum(weights, min_weights), max_weights)
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
                               min_acceptable_returns: Target = None,
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
        Optimizing the ratio ret(w)/risk(w) (which is not convex), belong to the class of fractional programming (FP).
        To solve FP problems, we introduce additional variables. However, ret(w) cannot include costs or L1 or L2
        normalization because they make it non-linear.

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
        min_acceptable_returns
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

        if objective_function == ObjectiveFunction.RATIO:
            self._default_scale = 1000
        else:
            self._default_scale = 1

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
                # noinspection PyTypeChecker
                objective = cp.Maximize(ret * self.scale - gamma * risk * self.scale)
            case ObjectiveFunction.RATIO:
                # noinspection PyTypeChecker
                if self.is_logarithmic_returns:
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
        return self._solve_problem(risk_measure=risk_measure,
                                   problem=problem,
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
                            min_acceptable_returns: Target = None):
        if min_acceptable_returns is None:
            min_acceptable_returns = self.assets.expected_returns

            # Additional matrix
        if (not np.isscalar(min_acceptable_returns)
                and min_acceptable_returns.shape != (len(min_acceptable_returns), 1)):
            semi_variance_returns_target = min_acceptable_returns[:, np.newaxis]

        v = cp.Variable(self.assets.date_nb)
        risk = cp.sum_squares(v) / (self.assets.date_nb - 1)
        # noinspection PyTypeChecker
        constraints = [(self.assets.returns - min_acceptable_returns).T @ w >= -v,
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
        r"""
        Minimum Variance Portfolio

        Parameters
        ----------
        objective_values: bool, default False
                          If true, the minimum variance is also returned with the weights.

        Returns
        -------
        If objective_values is True, the tuple (minimum variance, weights) of the optimal portfolio is returned,
        otherwise only the weights are returned.
        """
        return self.mean_risk_optimization(risk_measure=RiskMeasure.VARIANCE,
                                           objective_function=ObjectiveFunction.MIN_RISK,
                                           objective_values=objective_values)

    def minimum_semivariance(self,
                             min_acceptable_returns: Target = None,
                             objective_values: bool = False) -> Result:
        r"""
        Minimum Semivariance Portfolio

        Parameters
        ----------
        min_acceptable_returns: float | list | ndarray, optional
                                Minimum acceptable returns, in the same periodicity as the returns to distinguish
                                "downside" and "upside" returns.
                                If an array is provided, it needs to be of same size as the number of assets.

        objective_values: bool, default False
                          If true, the minimum semi_variance is also returned with the weights.

        Returns
        -------
        If objective_values is True, the tuple (minimum semi_variance, weights) of the optimal portfolio is returned,
        otherwise only the weights are returned.
        """
        return self.mean_risk_optimization(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                           objective_function=ObjectiveFunction.MIN_RISK,
                                           min_acceptable_returns=min_acceptable_returns,
                                           objective_values=objective_values)

    def minimum_cvar(self, beta: float = 0.95, objective_values: bool = False) -> Result:
        r"""
        Minimum CVaR (Conditional Value-at-Risk or Expected Shortfall) Portfolio.
        The CVaR is the average of the “extreme” losses beyond the VaR threshold

        Parameters
        ----------
        beta: float, default 0.95
              The VaR (Value At Risk) confidence level (expected VaR on the worst (1-beta)% days)

        objective_values: bool, default False
                          If true, the minimum CVaR is also returned with the weights.

        Returns
        -------
        If objective_values is True, the tuple (minimum CVaR, weights) of the optimal portfolio is returned,
        otherwise only the weights are returned.
        """

        return self.mean_risk_optimization(risk_measure=RiskMeasure.CVAR,
                                           objective_function=ObjectiveFunction.MIN_RISK,
                                           cvar_beta=beta,
                                           objective_values=objective_values)

    def minimum_cdar(self, beta: float = 0.95, objective_values: bool = False) -> Result:
        r"""
        Minimum CDaR (Conditional Drawdown-at-Risk) Portfolio.
        The CDaR is the average drawdown for all the days that drawdown exceeds a threshold

        Parameters
        ----------
        beta: float, default 0.95
               he DaR (Drawdown at Risk) confidence level (DaR on the worst (1-beta)% days)

        objective_values: bool, default False
                        If true, the minimum CDaR is also returned with the weights.

        Returns
        -------
        If objective_values is True, the tuple (minimum CDaR, weights) of the optimal portfolio is returned,
        otherwise only the weights are returned.
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
                         under this lower constraint.

        target_return: float | list | ndarray, optional
                       The target return of the portfolio: the portfolio variance is minimized under
                       this upper constraint.

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
        If objective_values is True:
            tuple (objective values of the optimization problem, weights)
        else:
            weights
        """
        self._validate_args(**locals())

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
                          population_size: PopulationSize = None,
                          min_acceptable_returns: Target = None,
                          l1_coef: Coef = None,
                          l2_coef: Coef = None,
                          objective_values: bool = False) -> Result:
        """
        Optimization along the mean-semivariance frontier

        Parameters
        ----------
        target_semivariance: float | list | ndarray, optional
                         The targeted semivariance (downside variance) of the portfolio: the portfolio return is
                         maximized under this lower constraint.

        target_return: float | list | ndarray, optional
                       The target return of the portfolio: the portfolio variance is minimized under
                       this upper constraint.

        population_size: int, optional
                         Number of pareto optimal portfolio weights to compute along the mean-variance efficient
                         frontier.


        min_acceptable_returns: float | list | ndarray, optional
                                Minimum acceptable returns, in the same periodicity as the returns to distinguish
                                "downside" and "upside" returns.
                                If an array is provided, it needs to be of same size as the number of assets.

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
        If objective_values is True:
            tuple (objective values of the optimization problem, weights)
        else:
            weights
        """
        self._validate_args(**locals())

        kwargs = dict(risk_measure=RiskMeasure.SEMI_VARIANCE,
                      min_acceptable_returns=min_acceptable_returns,
                      l1_coef=l1_coef,
                      l2_coef=l2_coef)

        if population_size is not None:
            min_semi_variance, _ = self.minimum_semivariance(objective_values=True)
            max_semi_variance_weight = self.mean_risk_optimization(objective_function=ObjectiveFunction.MAX_RETURN,
                                                                   **kwargs)
            # TODO: max_semi_variance
            target = np.logspace(np.log10(min_semi_variance) + 0.01, np.log10(0.4 ** 2), num=population_size)
            return self.mean_risk_optimization(objective_function=ObjectiveFunction.MAX_RETURN,
                                               max_semi_variance=target,
                                               objective_values=objective_values,
                                               **kwargs)

        elif target_semivariance is not None:
            return self.mean_risk_optimization(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                               objective_function=ObjectiveFunction.MAX_RETURN,
                                               l1_coef=l1_coef,
                                               l2_coef=l2_coef,
                                               max_semi_variance=target_semivariance,
                                               min_acceptable_returns=min_acceptable_returns,
                                               objective_values=objective_values)

        elif target_semivariance is not None:
            return self.mean_risk_optimization(risk_measure=RiskMeasure.SEMI_VARIANCE,
                                               objective_function=ObjectiveFunction.MIN_RISK,
                                               l1_coef=l1_coef,
                                               l2_coef=l2_coef,
                                               min_return=target_return,
                                               min_acceptable_returns=min_acceptable_returns,
                                               objective_values=objective_values)

        else:
            return self.minimum_semivariance(objective_values=objective_values,
                                             min_acceptable_returns=min_acceptable_returns)

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
