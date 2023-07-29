'''

'''

import pathlib

import fastai
import pandas as pd

from collections import ChainMap

from fastai.tabular.all import tabular_learner

from ..constants.runners import DEFAULT_PROCS

from .utils import (
    create_dataloaders,
    create_feature_sets,
    create_splits_from_tabular_object,
    get_classes_from_dls,
    get_target_type
)

from ..datatypes.model import (
    Model_data,
    ModelData
)


def transform_and_split_data(
    df: pd.DataFrame,
    file_name: str,
    target_label: str, 
    split: float = 0,
    name: str or None = None, 
    batch_size: int = 64,
    categorical : list = ['Protocol'], 
    procs = DEFAULT_PROCS,
    leave_out: list = []
) -> Model_data or ModelData:
    '''
        Transform and split the data into a train and test set
        A utility function that emulates the packaging of other experiment runners
        returns the 10-tuple with the following indicies:
        results: tuple = (name, model, classes, X_train, y_train, X_test, y_test, to, dls, model_type)
    '''

    print(f"Shape of Input Data: {df.shape}")

    if name is None:
        name = f'Splitting Data'
    
    categorical_features, continuous_features = create_feature_sets(
        df, 
        target_label, 
        leave_out = leave_out, 
        categorical = categorical
    )

    dls = create_dataloaders(
        df,
        target_label,
        categorical_features,
        continuous_features,
        procs,
        batch_size,
        split
    )
    
    to = dls.tabular_object

    model = tabular_learner(dls)
    X_train, X_test, y_train, y_test = create_splits_from_tabular_object(to)

    # we add a target_type_ attribute to our model so yellowbrick knows how to make the visualizations
    classes: list = get_classes_from_dls(dls)

    model.target_type_ = get_target_type(classes, allow_single=True)
    model._target_labels = target_label

    p = pathlib.Path(file_name)
    file_name: str = str(p.parts[-1])

    model_data: Model_data = Model_data(
        file_name, 
        model, 
        classes, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        to, 
        dls, 
        name
    )

    return model_data

    
def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions