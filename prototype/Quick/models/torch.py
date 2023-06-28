'''

'''

import fastai
import torch
import torch.nn as nn

from collections import ChainMap

from fastai.tabular.all import (
    Embedding,
    LinBnDrop,
    Module,
    SigmoidRange,
    is_listy,
    ifnone,
)

from .blocks import BottleneckResidualBlock
from .residual import ResidualTabularModel

class Torch_Model_Loader(Module):
    "Loads a torch model and allows use of the fastai scheduler and optimizer"
    def __init__(self, emb_szs, n_cont, out_sz, model=None, embed_p=0.0, y_range=None, bn_cont=True, config={'layers': [100 for _ in range(20)], 'cardinality':[1 for _ in range(20)]}):
    
        # setup for the model using Fastai's system of embedding categorical features
    
        self.embeds   = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont  = nn.BatchNorm1d(n_cont) if bn_cont else None
        self.n_emb    = sum(e.embedding_dim for e in self.embeds)
        self.in_sz    = self.n_emb + n_cont
        self.n_cont   = n_cont
        
        
        if(model == None):
            model = ResidualTabularModel
        self.model = model(input_features=self.in_sz, output_features=out_sz, config=config)


        _layers: list = []
        _layers.append(self.model)


        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.loader = nn.Sequential(*_layers)



    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.loader(x)



def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'torch': f'\t\t{torch.__version__}',
    })

    return versions

