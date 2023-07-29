
import pandas as pd
import numpy as np

from collections import ChainMap

from .unary import clamp_to_0_1

def produce_single_channel_image_matrix(
    df_sample: pd.DataFrame, 
    row_ordering: list, 
    column_ordering: list, 
    padding: tuple, 
    identity: float = 1, 
    operation: callable = lambda x, y: x*y,
    activation: callable = lambda x: x,
) -> np.ndarray:
    '''
    Function will create a len(row_ordering) x len(column_ordering) matrix
        representing the input sample.
    
    This uses the Identity-oriented Tabular Image Transformation algorithm to produce an image from tabular data:
            From the sample, two vectors reordered by the specified row and column ordering are padded with identity values
                before the operation is then applied to the generalised outer product of the vectors.

            Since we have an identity (or constant value if an identity does not exist), the top right corner with a size dictated 
                by the padding parameter will have a monochromatic square the allows classifiers to understand the orientation 
                of the image.

            The padding[0]xlen(row_ordering) and len(column_ordering)xpadding[1] submatrices will contain feature bands that 
                are meant to communicate the true values of the input features to the classifier

            The remaining interior submatrix will contain the nested feature geometry imposed by the operation on the 
                outer product of the feature vectors ordered by the input row_ordering and column_ordering vectors.

            An activation function is called on the output of the operation in order to tune the output before being clamped 
                to the interval [0,1]. 

            The produced image is called a Symmetric Identity-oriented Tabular Image if and only if these conditions are met
                1) the row_ordering and column_ordering vectors are the same or are in reversed order
                2) padding[0] = padding[1]
                3) the operation (binary function) being performed is commutative, i.e. f(a, b) = f(b, a)

            The produced image is called an Asymmetric Identity-oriented Tabular Image if it does not fulfill one of the above conditions.
    '''

    row_vector    = [identity] * padding[0] + list(df_sample.reindex(columns=row_ordering   ).values[0])
    column_vector = [identity] * padding[1] + list(df_sample.reindex(columns=column_ordering).values[0])

    product: callable = lambda x, y: clamp_to_0_1(activation(operation(x, y)))
    composed_outer = np.frompyfunc(product, 2, 1)

    pic = composed_outer.outer(row_vector, column_vector).astype(np.float64)

    return pic
    


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'pandas': f'\t\t{pd.__version__}',
        'numpy': f'\t\t{np.__version__}'
    })

    return versions

