import sys
import numpy as np

from collections import (
    ChainMap,
    namedtuple
)

from typing import (
    Any,
    List,
    NamedTuple,
    Union,
    # get_args,
    # get_origin
)

from ..constants.datatypes import BASE_TYPE_SCHEMA

Base_Data = namedtuple('base_data', BASE_TYPE_SCHEMA)

class BaseData(NamedTuple):
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    classes: List[str]
    target_label: str

    def validate(self, verbose: bool = False) -> bool:

        valid = True

        for field in self._fields:
            field_type = self.__annotations__[field]


            if (
                field_type != type(self.__getattribute__(field))
                or 
                (
                    hasattr(field_type, '__origin__')
                    and 
                    field_type.__origin__ == type(self.__getattribute__(field))
                )
            ):
                if field_type.__origin__ != type(self.__getattribute__(field)):
                    if verbose:
                        print(f'Field {field} is not of type {field_type}', file=sys.stderr)

                    valid = False
        
        return valid