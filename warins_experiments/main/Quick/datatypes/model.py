'''

'''

import numpy as np
import fastai
import torch

from fastai.tabular.data import TabularPandas
from fastai.data.core import DataLoaders
from fastcore.transform import Transform
from torch.utils.data import DataLoader

from collections import (
    ChainMap,
    namedtuple
)

from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Union
)

DataTransform = List[Transform]

Model_data = namedtuple(
    'model_data', 
    [
        'name', 
        'model', 
        'classes', 
        'X_train', 
        'y_train', 
        'X_test', 
        'y_test', 
        'to',         # TabularPandas Object
        'dls',        # DataLoaders Object
        'model_type'
    ]
)

class ModelData(NamedTuple):
    name: str
    model: Any
    classes: list
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    to: TabularPandas
    dls: Union[DataLoader, DataLoaders]
    model_type: str
    
    def __str__(self):
        return f"""ModelData(
            name={self.name}, 
            model={self.model}, 
            classes={self.classes}, 
            X_train={self.X_train}, 
            y_train={self.y_train}, 
            X_test={self.X_test}, 
            y_test={self.y_test}, 
            to={self.to}, 
            dls={self.dls}, 
            model_type={self.model_type}
        )"""
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def __hash__(self):
        return hash(self.__dict__)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, *args, **kwargs):
        return self.__dict__(*args, **kwargs)
    
    def __getitem__(self, key):
        return self.__dict__[key]


Model_datum = namedtuple(
    'Model_datum', 
    [
        'models', # A list of Model_data objects
        'results', # A dictionary of results by metric
        'avg_results' # A dictionary of average results by metric
    ]
)

class ModelDatum(NamedTuple):
    models: List[ModelData]
    results: dict

    def __str__(self):
        return f"""ModelDatum(
            models={self.models}, 
            results={self.results}
        )"""
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def __hash__(self):
        return hash(self.__dict__)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, *args, **kwargs):
        return self.__dict__(*args, **kwargs)
    
    def __getitem__(self, key):
        return self.__dict__[key]



def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'numpy': f'\t\t{np.__version__}',
        'torch': f'\t\t{torch.__version__}'
    })

    return versions

