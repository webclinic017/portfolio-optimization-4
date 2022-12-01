import numpy as np

from portfolio_optimization.optimization.group_constraints import group_constraints_to_matrix
from portfolio_optimization import GroupConstraintsError


def test_group_constraints_to_matrix():
    groups = np.array([['a', 'a', 'b', 'b'],
              ['c', 'a', 'c', 'a'],
              ['d', 'e', 'd', 'e']])

    constraints = ['a <= 2 * b ',
                   'a <= 1.2',
                   'd >= 3 ',
                   ' e >=  .5*d']
    a, b = group_constraints_to_matrix(groups=groups, constraints=constraints)
    np.testing.assert_array_almost_equal(a, np.array([[1., 1., -2., -1.],
                                                      [1., 1., 0., 1.],
                                                      [-1., 0., -1., 0.],
                                                      [0.5, -1., 0.5, -1.]]))

    np.testing.assert_array_almost_equal(b,
                                         np.array([0., 1.2, -3., 0.]))

    for c in [['a == 2'],
              ['a <= 2*bb'],
              ['a <= 2*b*c'],
              ['a 2']]:
        try:
            group_constraints_to_matrix(groups=groups, constraints=c)
            raise
        except GroupConstraintsError:
            pass
