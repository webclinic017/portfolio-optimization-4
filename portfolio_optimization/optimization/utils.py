import re

import numpy as np


class GroupConstraintsError(Exception):
    """ Error while processing group constraints """


groups = [['a', 'a', 'a', 'a', 'b', 'b'],
          ['c', 'c', 'd', 'e', 'e', 'e']]

constraints = ['a <= 2 * b',
               'a <= 1.2']

c = 'a <= 2 * b'

def group_constraints_to_matrix(groups:list[list[str]],
                                constraints:list[str]) -> tuple[np.ndarray, np.ndarray]:
    r1 = re.compile(r'^\s*(\w+)\s*(<=|==|>=)\s*(\w+)\s*(\*+)\s*(\w*)\s*$')
    r2 = re.compile(r'^\s*(\w+)\s*(<=|==|>=)\s*(\w+)\s$')

    for constraint_string in constraints:
        for r  in [r1, r2]:
            result = r.match(constraint_string)
            if result is not None:
                g1, operator, factor, _,  g2 = result.groups()
                try:
                    factor = float(factor)
                except ValueError:
                    raise GroupConstraintsError(f'Unable to convert {factor} into float in {constraint_string}')

