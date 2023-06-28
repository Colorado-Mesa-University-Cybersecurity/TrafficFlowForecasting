'''

'''

import pathlib

import fastai
import pandas as pd

from collections import ChainMap

from fastai.tabular.all import (
    accuracy,
    tabular_learner
)

from .wrappers import SklearnWrapper

from .utils import (
    create_dataloaders,
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
    ModelData
)


def run_deep_nn_experiment(
    df: pd.DataFrame, 
    file_name: str, 
    target_label: str, 
    shape: tuple, 
    split: float = 0.2, 
    categorical: list = ['Protocol'],
    procs = DEFAULT_PROCS, 
    leave_out: list = [],
    epochs: int = 10,
    batch_size: int = 64,
    metrics: list = DEFAULT_METRICS,
    callbacks: list = DEFAULT_CALLBACKS,
    lr_choice: str = VALLEY,
    name: str or None = None,
    fit_choice: str = ONE_CYCLE,
    no_bar: bool = False,
) -> Model_data or ModelData:
    '''
        Function trains a deep neural network model on the given data. 

        Parameters:
            df: pandas dataframe containing the data
            file_name: name of the file the dataset came from
            target_label: the label to predict
            shape: the shape of the neural network, the i-th value in the tuple represents the number of nodes in the i+1 layer
                    and the number of entries in the tuple represent the number of layers
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
            callbacks: list of callbacks to apply during training
            lr_choice: where the learning rate sampling function should find the optimal learning rate
                        choices are: 'valley', 'steep', 'slide', and 'minimum'
            fit_choice: choice of function that controls the learning schedule choices are:
                                'fit': a standard learning schedule 
                                'flat_cos': a learning schedule that starts high before annealing to a low value
                                'one_cyle': a learning schedule that warms up for the first epochs, continues at a high
                                                learning rate, and then cools down for the last epochs
                                the default is 'one_cycle'
         
        
        returns a model data named tuple
            model_data: tuple = (file_name, model, classes, X_train, y_train, X_test, y_test, model_type)
    '''
    
    print(f"Shape of Input Data: {df.shape}")
    shape = tuple(shape)

    if name is None:
        width: int = shape[0]
        for x in shape:
            width = x if (x > width) else width
        name = f'Deep_NN_{len(shape)}x{width}'

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

    learner = tabular_learner(
        dls, 
        layers=list(shape), 
        metrics = metrics,
        cbs=callbacks,
    )
    
    # extract the file_name from the path
    p = pathlib.Path(file_name)
    file_name: str = str(p.parts[-1])

    learner, results = run_model(
        name,
        learner,
        epochs,
        no_bar,
        lr_choice,
        fit_choice
    )

    
    print(f'loss: {results[0]}, accuracy: {results[1]*100: .2f}%')

    # we add a target_type_ attribute to our model so yellowbrick knows how to make the visualizations
    classes = get_classes_from_dls(dls)
    wrapped_model = SklearnWrapper(learner)

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
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions

