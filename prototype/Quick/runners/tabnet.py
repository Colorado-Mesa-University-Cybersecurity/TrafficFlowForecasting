'''

'''

import pathlib

import fastai
import fast_tabnet
import pandas as pd

from collections import ChainMap

from fastai.optimizer import ranger

from fastai.metrics import (
    BalancedAccuracy,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
    RocAuc
)

from fastai.tabular.all import (
    CrossEntropyLossFlat,
    Learner,
    accuracy,
    get_emb_sz
)

from fast_tabnet.core import (
    TabNetModel as TabNet
)

from .utils import (
    create_dataloaders,
    create_feature_sets,
    create_splits_from_tabular_object,
    get_classes_from_dls,
    get_target_type,
    run_model
)

from .wrappers import SklearnWrapper

from ..constants.runners import (
    DEFAULT_CALLBACKS,
    DEFAULT_METRICS,
    DEFAULT_PROCS,
    FLAT_COS,
    VALLEY
)

from ..datatypes.model import (
    Model_data,
    ModelData
)


def run_tabnet_experiment(
    df: pd.DataFrame, 
    file_name: str, 
    target_label: str, 
    split: float = 0.2, 
    name: str or None = None,
    categorical: list = ['Protocol'],
    procs = DEFAULT_PROCS, 
    leave_out: list = [],
    epochs: int = 10,
    steps: int = 1,
    batch_size: int = 64,
    metrics: list = DEFAULT_METRICS,
    attention_size: int = 16,
    attention_width: int = 16,
    callbacks: list = DEFAULT_CALLBACKS,
    lr_choice: str = VALLEY,
    fit_choice: str = FLAT_COS,
    no_bar: bool = False
) -> Model_data or ModelData:
    '''
    Function trains a TabNet model on the dataframe and returns a model data named tuple
        Based on TabNet: Attentive Interpretable Tabular Learning by Sercan Arik and Tomas Pfister from Google Cloud AI (2016)
            where a DNN selects features from the input features based on an attention layer. Each step of the model selects 
            different features and uses the input from the previous step to ultimately make predictions
    
        Combines aspects of a transformer, decision trees, and deep neural networks to learn tabular data, and has achieved state
            of the art results on some datasets.

        Capable of self-supervised learning, however it is not implemented here yet.

    (https://arxiv.org/pdf/1908.07442.pdf)

    Parameters:
        df: pandas dataframe containing the data
        file_name: name of the file the dataset came from
        target_label: the label to predict
        name: name of the experiment, if none a default is given
        split: the percentage of the data to use for testing
        categorical: list of the categorical columns
        procs: list of preprocessing functions to apply in the dataloaders pipeline
                additional options are: 
                    PCA_tabular (generate n principal components) 
                    Normal (features are scaled to the interval [0,1])
        leave_out: list of columns to leave out of the experiment
        epochs: number of epochs to train for  
        batch_size: number of samples processed in one forward and backward pass of the model
        metrics: list of metrics to calculate and display during training
        attention size: determines the number of rows and columns in the attention layers
        attention width: determines the width of the decision layer
        callbacks: list of callbacks to apply during training
        lr_choice: where the learning rate sampling function should find the optimal learning rate
                    choices are: 'valley', 'steep', 'slide', and 'minimum'
        fit_choice: choice of function that controls the learning schedule choices are:
                    'fit': a standard learning schedule 
                    'flat_cos': a learning schedule that starts high before annealing to a low value
                    'one_cyle': a learning schedule that warms up for the first epochs, continues at a high
                                    learning rate, and then cools down for the last epochs
                    the default is 'flat_cos'

    
    returns a model data named tuple
        model_data: tuple = (file_name, model, classes, X_train, y_train, X_test, y_test, model_type)
    '''

    print(f"Shape of Input Data: {df.shape}")

    if name is None:
        name = f"TabNet_steps_{steps}_width_{attention_width}_attention_{attention_size}"

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
    
    emb_szs = get_emb_sz(to)

    net = TabNet(emb_szs, len(to.cont_names), dls.c, n_d=attention_width, n_a=attention_size, n_steps=steps) 
    tab_model = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=metrics, opt_func=ranger, cbs=callbacks)

    # extract the file_name from the path
    p = pathlib.Path(file_name)
    file_name: str = str(p.parts[-1])

    tab_model, results = run_model(
        name,
        tab_model,
        epochs,
        no_bar,
        lr_choice,
        fit_choice
    )

    print(f'loss: {results[0]}, accuracy: {results[1]*100: .2f}%')

    # we add a target_type_ attribute to our model so yellowbrick knows how to make the visualizations
    classes = get_classes_from_dls(dls)
    wrapped_model = SklearnWrapper(tab_model)

    wrapped_model.target_type_ = get_target_type(classes)
    wrapped_model._target_labels = target_label
    
    model_data: Model_data = Model_data(
        file_name, 
        wrapped_model, 
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
        'fast_tabnet': f'\t{fast_tabnet.__version__}',
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions

