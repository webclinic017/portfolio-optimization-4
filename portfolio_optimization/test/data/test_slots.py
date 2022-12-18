import numpy as np
import portfolio_optimization.utils.metrics as mt
from portfolio_optimization.utils.tools import args_names, clean_locals
from portfolio_optimization.meta import Ratio, RiskMeasure

GLOBAL_ARGS_NAMES = {'returns',
                     'min_acceptable_return',
                     'compounded'}


class MyDescriptor:
    def __set_name__(self, owner, name):
        self.private_name = f'_{name}'

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        # Clear the property cache of the associated risk measure
        print(obj.__slots__)
        setattr(obj, self.private_name, value)


class MyClass:
    __slots__ = ('returns',
                 'min_acceptable_return',
                 '_b',
                 'c',
                 'semi_variance')

    def __init__(self):
        self._b = 3
        self.returns = np.array([1, 2, 3, 4])
        self.min_acceptable_return = None

    @property
    def b(self):
        return self._b

    def __setattr__(self, name, value):
        print('   __setattr__({}, {}) called'.format(name, value))
        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name != 'shape':
                print(f'compute function {name}')
            metric = RiskMeasure(name)
            func = getattr(mt, metric.value)
            args = {arg_name: getattr(self, arg_name) if arg_name not in GLOBAL_ARGS_NAMES else getattr(self, arg_name)
                    for arg_name in args_names(func)}
            value = func(**args)
            setattr(self, name, value)
            return value

    def reset(self):
        delattr(self, 'semi_variance')


class MyClass2(MyClass):
    __slots__ = ('x', 'y')

    def __init__(self):
        self.x = 10
        super().__init__()


def test_slots():
    cl = MyClass()
    print(cl.returns)
    cl.min_acceptable_return = 0
    print(cl.semi_variance)
    print(cl.semi_variance)
    cl.reset()
    print(cl.semi_variance)
    print(cl.__slots__)
    print(cl.b)
    cl.c = 5
    cl.b = 3
    hasattr(cl, 'semi_variance')
    print(cl.)
