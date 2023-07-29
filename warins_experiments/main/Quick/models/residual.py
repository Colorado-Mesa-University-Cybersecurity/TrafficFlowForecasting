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

class ResidualTabularModel(Module):
    "Residual model for tabular data."
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True),
                 lin_first=True, cardinality: list or None = None):
        ps = ifnone(ps, [0]*len(layers))
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]

        # print(f'sizes', sizes)
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        
        _layers: list = []
        num_residuals = 0
        residual_locations = []
        enum_length = len(list(enumerate(zip(ps+[0.],actns))))
        for i, (p, a) in enumerate(zip(ps+[0.],actns)):
            if(i==0 or i == enum_length-1):
                _layers.append(LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first))
            else:
                if(cardinality == None):
                    modules = [ LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first), ]
                else:
                    modules = [ LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first) for _ in range(cardinality[i]) ]
                num_residuals += 1 
                residual_locations.append(i)
                _layers.append( BottleneckResidualBlock(modules, i, sizes[i], sizes[i+1]) )

        print(f'Layer sizes: {sizes}, length: {len(sizes)}')
        print(f'Number of residual blocks: {num_residuals}')
        print('Residual locations: ', residual_locations)

        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)



def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'torch': f'\t\t{torch.__version__}',
    })

    return versions

