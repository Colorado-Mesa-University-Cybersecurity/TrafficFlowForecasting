'''

'''

import contextlib
import fastai
import pandas as pd

from collections import ChainMap

from fastai.data.all import DataLoaders

from fastai.tabular.all import (
    ClassificationInterpretation, 
    IndexSplitter,
    RandomSplitter,
    TabularPandas,
    range_of,
    tabular_learner
)

from ..constants.random import SEED

from ..constants.runners import (
    DEFAULT_LR_FUNCS, 
    FIT, 
    FLAT_COS, 
    LEARNING_RATE_OPTIONS, 
    ONE_CYCLE
)


def create_feature_sets(
    df: pd.DataFrame, 
    target_label: str, 
    leave_out: list = [],
    categorical: list = [],
) -> tuple:
    '''
        Function will create the categorical and continuous feature sets for the dataframe
    '''

    categorical_features: list = []
    untouched_features  : list = []

    for x in leave_out:
        if x in df.columns:
            untouched_features.append(x)

    for x in categorical:
        if x in df.columns:
            categorical_features.append(x)

    continuous_features = list(
        set(df) - 
        set(categorical_features) - 
        set([target_label]) - 
        set(untouched_features)
    )

    return categorical_features, continuous_features


def create_dataloaders(
    df: pd.DataFrame,
    target_label: str,
    categorical_features: list,
    continuous_features: list,
    procs: list,
    batch_size: int,
    split: float
) -> tuple:
    '''
        Function will create the data loaders for the experiment
    '''

    splits = RandomSplitter(valid_pct = split, seed=SEED)(range_of(df))

    to = TabularPandas(
        df, 
        procs=procs, 
        cat_names=categorical_features, 
        cont_names=continuous_features, 
        y_names=target_label, 
        splits=splits
    )

    try:
        dls = to.dataloaders(bs=batch_size)
    except:
        dls = to

    dls.tabular_object = to

    return dls


def create_cv_dataloaders(
    df: pd.DataFrame,
    target_label: str,
    categorical_features: list,
    continuous_features: list,
    procs: list,
    batch_size: int,
    test_splits
) -> tuple:
    '''
        Function will create the data loaders for the experiment
    '''

    splits = IndexSplitter(test_splits)(df)

    to = TabularPandas(
        df, 
        procs=procs, 
        cat_names=categorical_features, 
        cont_names=continuous_features, 
        y_names=target_label, 
        splits=splits
    )

    try:
        dls = to.dataloaders(bs=batch_size)
    except:
        dls = to

    dls.tabular_object = to

    return dls


def create_splits_from_tabular_object(to: TabularPandas) -> tuple:
    '''
        Function will create the splits from the tabular object
    '''

    # We extract the training and test datasets from the dataframe
    X_train = to.train.xs.reset_index(drop=True)
    X_test = to.valid.xs.reset_index(drop=True)
    y_train = to.train.ys.values.ravel()
    y_test = to.valid.ys.values.ravel()

    return X_train, X_test, y_train, y_test


def get_classes_from_dls(dls: DataLoaders) -> list:
    '''
        Function will return the classes from the dataloaders
    '''

    temp_model = tabular_learner(dls)

    return list(temp_model.dls.vocab)


def get_target_type(classes: list, allow_single = False) -> str:
    '''
        Function will return the type of classification problem
    '''

    if len(classes) == 2:
        target_type_ = 'binary'
    elif len(classes) > 2:  
        target_type_ = 'multiclass'
    else:
        if allow_single:
            target_type_ = 'single'
        else:
            print('Must be more than one class to perform classification')
            raise ValueError('Wrong number of classes')

    return target_type_


def run_model(
    name: str,
    learner,
    epochs: int,
    no_bar: bool,
    lr_choice: str,
    fit_choice: str,
    lr_funcs: list = DEFAULT_LR_FUNCS
):
    '''
        Function will run the model
    '''

    lr_choice: int = LEARNING_RATE_OPTIONS[lr_choice]
    

    with learner.no_bar() if no_bar else contextlib.ExitStack() as gs:

        if lr_choice != 'None':
            lr = learner.lr_find(suggest_funcs=lr_funcs)

                # fitting functions, they give different results, some networks perform better with different learning schedule during fitting
            if(fit_choice == FIT):
                learner.fit(epochs, lr[lr_choice])
            elif(fit_choice == FLAT_COS):
                learner.fit_flat_cos(epochs, lr[lr_choice])
            elif(fit_choice == ONE_CYCLE):
                learner.fit_one_cycle(epochs, lr_max=lr[lr_choice])
            else:
                assert False, f'{fit_choice} is not a valid fit_choice'

        else:

            if(fit_choice == FIT):
                learner.fit(epochs)
            elif(fit_choice == FLAT_COS):
                learner.fit_flat_cos(epochs)
            elif(fit_choice == ONE_CYCLE):
                learner.fit_one_cycle(epochs)
            else:
                assert False, f'{fit_choice} is not a valid fit_choice'


        learner.recorder.plot_sched() 
        results = learner.validate()
        interp = ClassificationInterpretation.from_learner(learner)
        interp.plot_confusion_matrix()

    learner.save(f'{name}.model')

    return learner, results


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'pandas': f'\t\t{pd.__version__}',
    })

    return versions

