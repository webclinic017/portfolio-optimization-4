import numpy as np
from numba import jit
import time

@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

@jit(nopython=True)
def t1(): # Function is compiled and runs in machine code
    s=0
    for i in range(1000000):
        s=s+1
    return s

@jit(nopython=True)
def dominate(fitness_1: np.array, fitness_2: np.array) -> bool:
    """
    Return true if each objective of the current portfolio's fitness is not strictly worse than
    the corresponding objective of the other portfolio's fitness and at least one objective is
    strictly better.
    """
    not_equal = False
    for self_value, other_value in zip(fitness_1, fitness_2):
        if self_value > other_value:
            not_equal = True
        elif self_value < other_value:
            return False
    return not_equal

@jit(nopython=True)
def dominate_loop():
    f1 = np.array([1,2,3,4,4])
    f2 = np.array([1,2,5,4,4])
    for i in range(10000):
        dominate(f1,f2)



s=time.time()
dominate_loop()
e=time.time()
print((e-s)*100)


