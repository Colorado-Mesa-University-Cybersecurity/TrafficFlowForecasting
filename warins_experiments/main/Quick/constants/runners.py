
from fastai.callback.all import (
    ShowGraphCallback,
    minimum,
    slide,
    steep,
    valley
)

from fastai.metrics import (
    BalancedAccuracy,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
    RocAuc
)

from fastai.tabular.all import (
    Categorify,
    FillMissing,
    Normalize,
    accuracy
)

from ..datatypes.model import DataTransform


FIT: str = 'fit'
ONE_CYCLE: str = 'one_cycle'
FLAT_COS: str = 'flat_cos'

VALLEY: str = 'valley'
SLIDE: str = 'slide'
STEEP: str = 'steep'
MINIMUM: str = 'minimum'

LEARNING_RATE_OPTIONS: dict = {
    VALLEY: 0,
    SLIDE: 1, 
    STEEP: 2, 
    MINIMUM: 3
}

DEFAULT_CALLBACKS: list = [
    ShowGraphCallback
]

DEFAULT_PROCS: DataTransform = [
    FillMissing, 
    Categorify, 
    Normalize
]

DEFAULT_LR_FUNCS: list = [
    valley, 
    slide, 
    steep, 
    minimum
]

DEFAULT_METRICS: list = [
    accuracy, 
    BalancedAccuracy(), 
    RocAuc(), 
    MatthewsCorrCoef(), 
    F1Score(average='macro'), 
    Precision(average='macro'), 
    Recall(average='macro')
]