import numpy as np
import numpy.linalg as linalg
from scipy.stats import entropy

def JSD(P, Q):
    _P = P / linalg.norm(P, ord=1)
    _Q = Q / linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def ent(array):
    return -np.sum(array * np.log(array),axis=1)
