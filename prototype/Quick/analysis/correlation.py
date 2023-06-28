
import pandas as pd

from collections import ChainMap

from ..datatypes.model import (
    Model_data,
    ModelData
)

def calculate_correlations(model_data: Model_data or ModelData, target_label: str):
    '''
        Function merges together the encoded and standardized model data and labels to calculate pearson correlation
    '''

    encoded_data = model_data.X_train.copy()
    encoded_data[target_label] = model_data.y_train

    return encoded_data.corr()


def extract_correlations(correlations: pd.DataFrame, feature: str, leave_out: list = ['Traffic Type', 'Application Type']) -> list:
    '''
        Function takes a correlation dataframe and extracts a list of features correlated with the input feature. 
            Anything in leave_out is not included in the list.
    '''

    correlation_order: list = list(correlations.sort_values(by=feature, ascending=False).index)

    for x in leave_out:
        if x in correlation_order:
            correlation_order.remove(x)

    return correlation_order

    
def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions

