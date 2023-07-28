
import pandas as pd

from collections import ChainMap

from ..datatypes.model import (
    Model_data,
    ModelData
)


def decode_classes_and_create_Xy_df(model_data: Model_data or ModelData, target_label: str):
    """
        Function takes a Model_data namedtuple and returns a dataframe with the X and decoded y data
    """
    X = model_data.X_train
    y = model_data.y_train
    classes = model_data.classes

    Xy_df = pd.DataFrame(X)
    y_s: list = []
    for x in y:
        y_s.append(classes[x])
    Xy_df[target_label] = y_s

    return Xy_df


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions

