'''

'''

import numpy as np

from collections import (
    ChainMap,
    namedtuple
)

from typing import NamedTuple


PCA_data = namedtuple(
    'pca_data',
    [
        'title', 
        'X_train', 
        'y_train', 
        'Xy', 
        'components', 
        'n_components', 
        'classes', 
        'target_label'
    ]
)

class PCAData(NamedTuple):
    title: str
    X_train: np.ndarray
    y_train: np.ndarray
    Xy: np.ndarray
    components: np.ndarray
    n_components: int
    classes: list
    target_label: str

    def __str__(self):
        return f"""PCAData(
            title={self.title}, 
            X_train={self.X_train}, 
            y_train={self.y_train}, 
            Xy={self.Xy}, 
            components={self.components}, 
            n_components={self.n_components}, 
            classes={self.classes}, 
            target_label={self.target_label}
        )"""


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'numpy': f'\t\t{np.__version__}',
    })

    return versions

