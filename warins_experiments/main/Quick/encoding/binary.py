
import numpy as np

from collections import ChainMap


def _dist_l1(x, y):
    return np.abs(x - y)

def _dist_l2(x, y):
    val = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return val 

def _avg(x, y):
    return (x + y) / 2

def _mult(x, y):
    return x * y

def _add(x, y):
    return x + y

def _rbf_l1(x, y):
    '''radial basis function'''
    return np.exp(-_dist_l1(x, y)**2)

def _rbf_l2(x, y):
    '''radial basis function'''
    return np.exp(-_dist_l2(x, y)**2)


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'numpy': f'\t\t{np.__version__}'
    })

    return versions

