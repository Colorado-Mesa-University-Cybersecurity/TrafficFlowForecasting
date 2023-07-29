'''

'''

import fastai

from collections import ChainMap

from fastai.tabular.all import (
    Learner,
    TabularLearner,
    delegates,
    get_c,
    get_emb_sz,
    tabular_config,
)

from .torch import Torch_Model_Loader
from .residual import ResidualTabularModel

@delegates(Learner.__init__)
def residual_tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, cardinality=None, ps=None, **kwargs):
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params."
    if config is None: config = tabular_config()
    if layers is None: layers = [200,100]
    to = dls.train_ds
    emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = ResidualTabularModel(emb_szs, len(dls.cont_names), n_out, layers, y_range=y_range, cardinality=cardinality, ps=ps, **config)
    return TabularLearner(dls, model, **kwargs)


@delegates(Learner.__init__)
def torch_learner(dls, model=None, emb_szs=None, n_out=None, config={'layers': [100 for _ in range(20)], 'cardinality':[1 for _ in range(20)]}, y_range=None, **kwargs):
    '''
        Creates a Fastai Learner using a dataloader and passed in model 
            model parameters are passed in through config
            Tabular learner parameters are passed in as additional kwargs
    
        returns a Fastai TabularModel created using the remaining params
    '''
    emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
    if n_out is None: 
        n_out = get_c(dls)
    
    # option to scale outputs to a certain range, uses a sigmoid function
    # if y_range is None and 'y_range' in config: 
    #     y_range = config.pop('y_range')

    md = Torch_Model_Loader(emb_szs, len(dls.cont_names), n_out, config=config, y_range=y_range, model=model)

    return TabularLearner(dls, md, **kwargs)


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
    })

    return versions

