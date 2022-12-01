import re
import numpy as np

from portfolio_optimization.exception import GroupConstraintsError

__all__ = ['group_constraints_to_matrix']


def group_to_array(groups: np.ndarray, group: str) -> np.ndarray:
    """
    Convert a 2d array of groups into a vector of 1 or 0 for the given group.
    """
    arr = np.any(np.where(groups == group, True, False), axis=0)
    if not np.any(arr):
        raise GroupConstraintsError(f"unable to find the group named '{group}' in the groups provided")
    return 1 * arr


def operator_sign(operator: str) -> int:
    """
    Convert the operators '>=' and '<=' into +1 or -1
    """
    if operator == '<=':
        return 1
    if operator == '>=':
        return -1
    raise GroupConstraintsError(f"operator '{operator}' is not valid. It should be be '<=' or '>='")


def factor_to_float(factor: str) -> float:
    """
    Convert the factor string into a float
    """
    try:
        return float(factor)
    except ValueError:
        raise GroupConstraintsError(f'Unable to convert {factor} into float from the constraint provided')


def group_constraints_to_matrix(groups: np.ndarray,
                                constraints: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of constraints into the left and right matrix of the weight constraint inequality A <= B.

    Parameters
    ----------
    groups: np.ndarray
            The list of assets group.

            Examples:
                groups = np.array([['Equity', 'Equity', 'Bond', 'Fund'],
                                  ['US', 'Europe', 'Japan', 'US']])

    constraints: list[str]
                 The list of constraints.
                 They need to respect the following patterns:
                    * 'group_1 <= factor * group_2'
                    * 'group_1 >= factor * group_2'
                    * 'group_1 <= factor'
                    * 'group_1 >= factor'

                factor can be a float or an int.
                group_1 and group_2 are the group names defined in groups.
                The first expression means that the sum of all assets in group_1 should be less or equal to 'factor'
                times the sum of all assets in group_2.

                 Examples:
                    constraints = ['Equity' <= 3 * 'Bond',
                                   'US' >= 1.5,
                                   'Europe' >= 0.5 * Fund',
                                   'Japan' <= 1]

    Returns
    -------
    The left and right matrix of the constraint inequality A <= B
    """
    if len(groups.shape) != 2:
        raise GroupConstraintsError('groups should be a 2d array')

    r1 = re.compile(r'^\s*(\w+)\s*(<=|>=)\s*(\S+)\s*(\*+)\s*(\w*)\s*$')
    r2 = re.compile(r'^\s*(\w+)\s*(<=|>=)\s*(\S+)\s*$')

    a, b = [], []
    for constraint_string in constraints:
        result = r1.match(constraint_string)
        if result is not None:
            g1, operator, factor, _, g2 = result.groups()
            factor = factor_to_float(factor)
            v1 = group_to_array(groups=groups, group=g1)
            v2 = group_to_array(groups=groups, group=g2)
            s = operator_sign(operator)
            a.append(s * (v1 - factor * v2))
            b.append(0)
        else:
            result = r2.match(constraint_string)
            if result is not None:
                g1, operator, factor = result.groups()
                factor = factor_to_float(factor)
                v1 = group_to_array(groups=groups, group=g1)
                s = operator_sign(operator)
                a.append(s * v1)
                b.append(s * factor)
            else:
                raise GroupConstraintsError(f"The constraint '{constraint_string}' doesn't match any of the patterns. "
                                            f"Please check the documentation.")

    a = np.stack(a, axis=0)
    b = np.array(b, dtype=float)
    return a, b
