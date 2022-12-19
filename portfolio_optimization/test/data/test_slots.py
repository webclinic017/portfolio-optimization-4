import numpy as np
import portfolio_optimization.utils.metrics as mt
from portfolio_optimization.utils.tools import args_names, clean_locals
from portfolio_optimization.meta import Ratio, RiskMeasure
from functools import wraps

GLOBAL_ARGS_NAMES = {'returns',
                     'min_acceptable_return',
                     'compounded'}


def cached_property2(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        # method_output = method(self, *args, **kwargs)
        return method

    return wrapped


class cached_property:
    def __init__(self, func):
        self.func = func
        self.private_name = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = f'_{name}'

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.private_name is None:
            raise TypeError('Cannot use cached_property instance without calling __set_name__ on it.')
        try:
            value = getattr(instance, self.private_name)
        except AttributeError:
            value = self.func(instance)
            setattr(instance, self.private_name, value)
        return value

    def __set__(self,instance, owner=None):
        raise AttributeError(f"'{type(instance).__name__}' object attribute '{self.public_name}' is read-only")

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
                 '_z',
                 'c',
                 'semi_variance')

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @cached_property
    def b(self):
        x=3*5
        return x

    def __setattr__(self, name, value):
        print('   __setattr__({}, {}) called'.format(name, value))
        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if name != 'shape':
                print(f'compute function {name}')
            try:
                metric = RiskMeasure(name)
                func = getattr(mt, metric.value)
                args = {arg_name: getattr(self, arg_name) if arg_name not in GLOBAL_ARGS_NAMES else getattr(self, arg_name)
                        for arg_name in args_names(func)}
                value = func(**args)
                setattr(self, name, value)
                return value
            except ValueError:
                raise AttributeError(e)

    def reset(self):
        delattr(self, 'semi_variance')


class MyClass2(MyClass):
    __slots__ = ('x', 'y')

    def __init__(self):
        self.x = 10
        super().__init__()


def test_slots():
    cl = MyClass(returns=np.array([1, 2, 3, 4]),
                 min_acceptable_return=0)
    cl.returns
    cl.b=3
    cl._z=2

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

class property2(object):
    """
    Property attribute.

      fget
        function to be used for getting an attribute value
      fset
        function to be used for setting an attribute value
      fdel
        function to be used for del'ing an attribute
      doc
        docstring

    Typical use is to define a managed attribute x:

    class C(object):
        def getx(self): return self._x
        def setx(self, value): self._x = value
        def delx(self): del self._x
        x = property(getx, setx, delx, "I'm the 'x' property.")

    Decorators make defining new properties or modifying existing ones easy:

    class C(object):
        @property
        def x(self):
            "I am the 'x' property."
            return self._x
        @x.setter
        def x(self, value):
            self._x = value
        @x.deleter
        def x(self):
            del self._x
    """
    def deleter(self, *args, **kwargs): # real signature unknown
        """ Descriptor to obtain a copy of the property with a different deleter. """
        pass

    def getter(self, *args, **kwargs): # real signature unknown
        """ Descriptor to obtain a copy of the property with a different getter. """
        pass

    def setter(self, *args, **kwargs): # real signature unknown
        """ Descriptor to obtain a copy of the property with a different setter. """
        pass

    def __delete__(self, *args, **kwargs): # real signature unknown
        """ Delete an attribute of instance. """
        pass

    def __getattribute__(self, *args, **kwargs): # real signature unknown
        """ Return getattr(self, name). """
        pass

    def __get__(self, *args, **kwargs): # real signature unknown
        """ Return an attribute of instance, which is of type owner. """
        pass

    def __init__(self, fget=None, fset=None, fdel=None, doc=None): # known special case of property.__init__
        """
        Property attribute.

          fget
            function to be used for getting an attribute value
          fset
            function to be used for setting an attribute value
          fdel
            function to be used for del'ing an attribute
          doc
            docstring

        Typical use is to define a managed attribute x:

        class C(object):
            def getx(self): return self._x
            def setx(self, value): self._x = value
            def delx(self): del self._x
            x = property(getx, setx, delx, "I'm the 'x' property.")

        Decorators make defining new properties or modifying existing ones easy:

        class C(object):
            @property
            def x(self):
                "I am the 'x' property."
                return self._x
            @x.setter
            def x(self, value):
                self._x = value
            @x.deleter
            def x(self):
                del self._x
        # (copied from class doc)
        """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __set_name__(self, *args, **kwargs): # real signature unknown
        """ Method to set name of a property. """
        pass

    def __set__(self, *args, **kwargs): # real signature unknown
        """ Set an attribute of instance to value. """
        pass

    fdel = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    fget = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    fset = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    __isabstractmethod__ = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default





class C:
    __slots__={'_efg'}

    def __init__(self):
        pass


    @property2
    def abc(self):
        return 2

    @cached_property
    def efg(self):
        return 2

class D(C):
    __slots__={'jk'}

    def __init__(self, a):
        self._efg=a
        super().__init__()

    @property2
    def abc(self):
        return 2

    @cached_property
    def efg(self):
        return 2


d=D()
print(d.abc)
print(d.adcs)



