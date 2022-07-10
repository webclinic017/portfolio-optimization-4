import logging
import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence

from portfolio_optimization.exception import *

__all__ = ['get_optimization_weights']

logger = logging.getLogger('portfolio_optimization.utils.solver')


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
