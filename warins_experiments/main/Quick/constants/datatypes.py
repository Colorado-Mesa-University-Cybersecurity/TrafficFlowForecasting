

BASE_TYPE_SCHEMA = [
    'name',
    'X_train',
    'y_train',
    'classes',
    'target_label'
]

MODEL_TYPE_SCHEMA = [
    *BASE_TYPE_SCHEMA,
    'model',
    'model_type',
    'X_test',
    'y_test',
    'to',
    'dls'   
]

COMPONENT_BASE_TYPE_SCHEMA = [
    'Xy',
    'Components',
    'n_components'
]


COMPONENT_TYPE_SCHEMA = [
    *BASE_TYPE_SCHEMA,
    *COMPONENT_BASE_TYPE_SCHEMA
]

CLOUD_TYPE_SCHEMA = [
    *BASE_TYPE_SCHEMA,
    'clouds', 
    'clouds_y', 
    'clouds_y_decoded', 
]

TOPOLOGICAL_TYPE_SCHEMA = [
    *BASE_TYPE_SCHEMA,
    'persistence', 
    'features', 
    'Xy'
    'fig', 
]
