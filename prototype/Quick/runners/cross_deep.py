'''

'''

import pathlib

import fastai
import numpy as np
import pandas as pd

from collections import ChainMap

from fastai.tabular.all import (
    tabular_learner
)

from sklearn.model_selection import StratifiedShuffleSplit

from ..constants.random import SEED

from .wrappers import SklearnWrapper

from .utils import (
    create_cv_dataloaders,
    create_feature_sets,
    create_splits_from_tabular_object,
    get_classes_from_dls,
    get_target_type,
    run_model
)

from ..constants.runners import (
    DEFAULT_CALLBACKS,
    DEFAULT_METRICS,
    DEFAULT_PROCS,
    ONE_CYCLE,
    VALLEY
)

from ..datatypes.model import (
    Model_data,
    Model_datum,
    ModelData,
    ModelDatum
)


def run_cross_validated_deep_nn_experiment(
    df: pd.DataFrame, 
    file_name: str, 
    target_label: str, 
    shape: tuple, 
    k_folds: int = 10,
    categorical = ['Protocol'],
    callbacks: list = DEFAULT_CALLBACKS,
    metrics: list = DEFAULT_METRICS,
    procs: list = DEFAULT_PROCS,
    leave_out: list = [],
    experiment_type: str or None = None,
    epochs: int = 10,
    batch_size: int = 64,
    name: str or None = None,
    lr_choice: str = VALLEY,
    fit_choice: str = ONE_CYCLE,
    no_bar: bool = False
) -> Model_datum or ModelDatum:
    '''
        Function will fit a deep neural network to the given dataset using cross-validation
    '''

    print('Shape of input dataframe:', df.shape)
    print(f"Running {k_folds}-fold cross-validation")

    if name is None:
        width: int = shape[0]
        for x in shape:
            width = x if (x > width) else width
        name = f'Deep_NN_{len(shape)}x{width}_cross_validated'

    metrics_ = metrics

    if(experiment_type is None):
        experiment_type = f'Deep_NN_{shape[0]}x{shape[1]}'

    p = pathlib.Path(file_name)
    file_name: str = str(p.parts[-1])

    categorical_features, continuous_features = create_feature_sets(
        df, 
        target_label, 
        leave_out = leave_out, 
        categorical = categorical
    )



    ss = StratifiedShuffleSplit(n_splits=k_folds, random_state=SEED, test_size=1/k_folds)


    model_data_list: list = [0]*k_folds

    if metrics is DEFAULT_METRICS:
        fold_results: dict = {'loss': [], 'accuracy': [], 'BalancedAccuracy': [], 'roc_auc': [], 'MCC': [], 'f1': [], 'precision': [], 'recall': [], 'all': []}
    else:
        fold_results: dict = {'all': []}


    for i, (train_index, test_index) in enumerate(ss.split(df.copy().drop(target_label, axis=1), df[target_label])):

        print(i)
        print(f'Train Index: {train_index.shape}')
        print(f'Test Index: {test_index.shape}')
        
        fold_name = f'{name}_fold_{i+1}'

        dls = create_cv_dataloaders(
            df,
            target_label,
            categorical_features,
            continuous_features,
            procs,
            batch_size,
            test_index
        )

        to = dls.tabular_object

        model = tabular_learner(
            dls, 
            layers=list(shape), 
            metrics=metrics, 
            cbs=callbacks,
        )

        model, model_results = run_model(
            fold_name,
            model,
            epochs,
            no_bar,
            lr_choice,
            fit_choice
        )

        if(metrics_ is DEFAULT_METRICS):
            fold_results['loss'].append(model_results[0])
            fold_results['accuracy'].append(model_results[1])
            fold_results['BalancedAccuracy'].append(model_results[2])
            fold_results['roc_auc'].append(model_results[3])
            fold_results['MCC'].append(model_results[4])
            fold_results['f1'].append(model_results[5])
            fold_results['precision'].append(model_results[6])
            fold_results['recall'].append(model_results[7])
            fold_results['all'].append(model_results)
            print(f'loss: {model_results[0]}, accuracy: {model_results[1]*100: .2f}%')
        else:
            fold_results['all'].append(model_results)
            print(f'loss: {model_results[0]}')


        X_train, X_test, y_train, y_test = create_splits_from_tabular_object(to)

        classes = get_classes_from_dls(dls)
        wrapped_model = SklearnWrapper(model)

        wrapped_model.target_type = get_target_type(classes)
        wrapped_model._target_labels = target_label

        model_data_list[i]: Model_data = Model_data(
            fold_name, 
            wrapped_model, 
            classes, 
            X_train, 
            y_train, 
            X_test, 
            y_test, 
            to, 
            dls, 
            experiment_type
        )

    results: dict = {}
    avg_results: dict = {}

    for key, item in fold_results.items():
        results[key] = tuple(item)
        avg_results[key] = np.mean(item)

    model_datum: Model_datum = Model_datum(
        tuple(model_data_list), 
        fold_results,
        avg_results
    )

    return model_datum


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'numpy': f'\t\t{np.__version__}',
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions

