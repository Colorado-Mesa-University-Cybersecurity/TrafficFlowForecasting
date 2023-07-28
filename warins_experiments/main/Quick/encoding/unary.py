'''

'''

import numpy as np

from collections import ChainMap


def clamp_to_0_1(x):
    return max(min(x, 1), 0)

def p_2_pot(x):
    ''' 
        A pot shape in the interval [0,1] using a second degree polynomial
        0 and 1 are mapped to 1 and .5 is mapped to 0
    '''
    return np.power(2, 2*(x-.5))

def p_4_pot(x):
    ''' 
        A pot shape in the interval [0,1] using a fourth degree polynomial
        0 and 1 are mapped to 1 and .5 is mapped to 0
    '''
    return np.power(4, 2*(x-.5))

def p_6_pot(x):
    ''' 
        A pot shape in the interval [0,1] using a sixth degree polynomial
        0 and 1 are mapped to 1 and .5 is mapped to 0
    '''
    return np.power(6, 2*(x-.5))

def gaussian_spike(x):
    return np.exp(-np.power(x, 2))

def double_gaussian_spike(x):
    ''' 
        A double gaussian spike shape in the interval [0,1]
            gives larger changes in gradient to spread around input 
            in the interval [0,1]
    '''

    first_spike  = (-0.5) * gaussian_spike((53*x) - 2.2)
    second_spike = gaussian_spike((2.7*x) - 2.2)

    return first_spike + second_spike

def sharp_zero_spike(x):
    return 1 - np.exp(-np.power(np.power((200*x), 2)+0.0001, -1))

def curve_over_x_1(x):
    '''
        A curve on the interval [0,1] that maps 0 to 0 and 1 to 1 but stays above
            the line f(x) = x
    '''

    return x*np.exp(1-x)

def curve_over_x_2(x):
    return -(x - 1)*(x - 1) + 1

# closures allowing customizable functions

def mult_by_n(n):
    def f(x):
        return n * x
    return f


def p_n_pot(n):
    ''' 
        A pot shape in the interval [0,1] using an nth-degree polynomial
        0 and 1 are mapped to 1 and .5 is mapped to 0
    '''
    def f(x):
        return np.power(n, 2*(x-.5))
    return f


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'numpy': f'\t\t{np.__version__}'
    })

    return versions

