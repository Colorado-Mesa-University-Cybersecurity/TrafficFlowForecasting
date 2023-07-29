'''

'''

import pathlib

import fastai
import pandas as pd
import sklearn

from collections import ChainMap

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, 
    classification_report
)

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


def run_sk_experiment(
    df: pd.DataFrame, 
    file_name: str, 
    target_label: str, 
    split: float = 0.2,
    batch_size: int = 64, 
    categorical : list = ['Protocol'], 
    procs = DEFAULT_PROCS, 
    name: str or None = 'K-Nearest Neighbors',
    leave_out: list = [], 
    model = KNeighborsClassifier()
) -> Model_data or ModelData:
    '''
        Run binary classification using K-Nearest Neighbors
        returns the 10-tuple Model_data
    '''

    print(f"Shape of Input Data: {df.shape}")

    if name is None:
        name = f'SKLearn Classifier: {name}'
 
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

    X_train, X_test, y_train, y_test = create_splits_from_tabular_object(to)

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    report = classification_report(y_test, prediction)
    print(report)
    print(f'\tAccuracy: {accuracy_score(y_test, prediction)}\n')

    # we add a target_type_ attribute to our model so yellowbrick knows how to make the visualizations
    classes = get_classes_from_dls(dls)
    model.target_type_ = get_target_type(classes)

    # extract the name from the path
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
        'pandas': f'\t\t{pd.__version__}',
        'sklearn': f'\t\t{sklearn.__version__}',
    })

    return versions

