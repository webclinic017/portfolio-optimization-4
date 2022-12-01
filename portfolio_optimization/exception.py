__all__ = ['OptimizationError',
           'GroupConstraintsError']


class OptimizationError(Exception):
    """ Optimization Did not converge """


class GroupConstraintsError(Exception):
    """ Error while processing group constraints """
