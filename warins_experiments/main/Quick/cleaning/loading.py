'''

'''

from typing import Callable
import pandas as pd

from collections import ChainMap

from .utils import (
    clean_data, 
    features_with_bad_values, 
    load_data 
)

from Quick.constants.cleaning import (
    FILE,
    DATASET,
    STATS
)


def examine_dataset(job_id: int, files: str, datasets: str) -> ChainMap:
    '''
        Function will return a chainmap dictionary containing dataframe of the job_id 
            passed in as well as that dataframe's feature stats, data composition, 
            and file name.

        This chainmap is expected as the input for all of the other helper functions
    '''
    # if files is None and 'file_set' in tuple(globals()):
    #     files = file_set

    # if datasets is None and 'data_set' in tuple(globals()):
    #     datasets = data_set


    job_id = job_id - 1  # adjusts for indexing while enumerating jobs from 1
    print(f'Dataset {job_id+1}/{len(datasets)}: We now look at {files[job_id]}\n\n')

    # Load the dataset
    df: pd.DataFrame = load_data(files[job_id])
 

    # print the data composition
    print(f'''
        File:\t\t\t\t{files[job_id]}  
        Job Number:\t\t\t{job_id+1}
        Shape:\t\t\t\t{df.shape}
        Samples:\t\t\t{df.shape[0]} 
        Features:\t\t\t{df.shape[1]}
    ''')
    

    # return the dataframe and the feature stats in a chainmap. This is a dictionary that can be
    #     updated by other functions, grouping together dictionaries and only showing the keys
    #     that appear first in the chainmap. This is useful for transforming the dataset and 
    #     performing experiments, so we can keep track of the history and do not need to reload
    #     the dataset.
    data_summary: ChainMap =  ChainMap({
        FILE:             files[job_id],
        DATASET:          df,
        STATS:    features_with_bad_values(df, files[job_id]), 
    })
    
    return data_summary


def dataset_loader(files: str, datasets: str) -> Callable[[int], ChainMap]:
    '''
        Function will return a function that will load the dataset and return a dictionary containing the dataframe and
        the feature stats.
    '''

    def load_dataset(job_id: int) -> ChainMap:
        return examine_dataset(job_id, files, datasets)

    return load_dataset


def package_data_for_inspection(df: pd.DataFrame) -> ChainMap:
    '''
        Function will return a dictionary containing dataframe passed in as well as that dataframe's feature stats.
    '''

    # print the data composition
    print(f'''
    Packaging Data:

    Dataset statistics:
        Shape:\t\t\t\t{df.shape}
        Samples:\t\t\t{df.shape[0]} 
        Features:\t\t\t{df.shape[1]}
    ''')

    # return the dataframe and the feature stats
    data_summary: ChainMap =  ChainMap({
        'File':             '',
        'Dataset':          df,
        'Feature_stats':    features_with_bad_values(df, ''), 
    })
    
    return data_summary


def package_data_for_inspection_with_label(df: pd.DataFrame, label: str) -> ChainMap:
    '''
        Function will return a dictionary containing dataframe passed in as well as that dataframe's feature stats.
    '''

    # print the data composition
    print(f'''
    Packaging Data {label}:

    Dataset statistics:
        Shape:\t\t\t\t{df.shape}
        Samples:\t\t\t{df.shape[0]} 
        Features:\t\t\t{df.shape[1]}
    ''')
    

    # return the dataframe and the feature stats
    data_summary: ChainMap =  ChainMap({
        'File':             f'{label}',
        'Dataset':          df,
        'Feature_stats':    features_with_bad_values(df, f'{label}'),
    })
    
    return data_summary


def remove_infs_and_nans(data_summary: ChainMap) -> pd.DataFrame:
    '''
        Function will return the dataset with all inf and nan values removed.
    '''

    df: pd.DataFrame = data_summary['Dataset'].copy()
    df = clean_data(df, [])

    return df


def rename_columns(data_summary: ChainMap, columns: list, new_names: list) -> ChainMap:
    '''
        Function will return the data_summary dict with the names of the columns in the dataframe changed
    '''

    df: pd.DataFrame = data_summary['Dataset'].copy()
    for x, i in enumerate(columns):
        df.rename(columns={i: new_names[x]}, inplace=True)

    out: dict = {'Dataset': df}

    return data_summary.new_child(out)


def rename_values_in_column(data_summary: ChainMap, replace: list) -> pd.DataFrame:
    '''
        Function will return a dataframe with the names of the columns changed

        replace: [('column', {'old_name': 'new_name', ...}), ...]
    '''
    length: int = len(replace)

    df: pd.DataFrame = data_summary['Dataset'].copy()
    for i in range(length):
        df[replace[i][0]].replace(replace[i][1], inplace=True)


    return df


def rename_values_in_column_df(df: pd.DataFrame, replace: list) -> pd.DataFrame:
    '''
        Function will return a dataframe with the names of the columns changed

        replace: [('column', {'old_name': 'new_name', ...}), ...]
    '''
    length: int = len(replace)

    df1: pd.DataFrame = df.copy()
    for i in range(length):
        df1[replace[i][0]].replace(replace[i][1], inplace=True)


    return df1



def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions

