
import pandas as pd

from collections import ChainMap

from ..constants.runners import DEFAULT_PROCS

from .decode import decode_classes_and_create_Xy_df

from ..runners.package import transform_and_split_data

from ..runners.transforms import Normal

def normalize_bilabel_dataset_between_0_1(
    df: pd.DataFrame,
    labels = ['Traffic Type', 'Application Type'],
    categorical = ['Protocol'],
    leave_out = []
) -> pd.DataFrame:
    '''
        Function takes a dataframe and merges its labels before normalizing data. 
        The labels are then split back into their original form, but the merged label is kept for verification purposes.

        returns a trilabel dataframe with the new label under the 'Merged Type' column
    '''
    inter_df = df.copy()

    d1_y = inter_df[labels[0]]
    d2_y = inter_df[labels[1]]

    for label in labels:
        if label not in leave_out:
            leave_out.append(label)

    merged_y = pd.concat([d1_y, d2_y], axis=1)

    merged: list = []
    for x in zip(d1_y, d2_y):
        merged.append(f'{x[0]}_{x[1]}')

    inter_df['Merged Type'] = merged

    procs = DEFAULT_PROCS.copy()
    procs.append(
        Normal
    )

    normalized_model_data = transform_and_split_data(
        inter_df,
        '',
        'Merged Type',
        split = 0,
        name = 'Normalized Data',
        categorical = categorical,
        leave_out = labels,
        procs = procs
    )

    merged_df = decode_classes_and_create_Xy_df(normalized_model_data, 'Merged Type')

    merged_y = merged_df['Merged Type']
    new_labels: list = []

    for l in labels:
        new_labels.append([])

    for x in merged_y:
        split_label = x.split('_')

        for i in range(len(split_label)):
            new_labels[i].append(split_label[i])

    for i, x in enumerate(labels):
        merged_df[x] = new_labels[i]

    total_df = merged_df.copy()
    total_df['Merged Type'] = merged_y


    return total_df

    
    
def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions

