'''

'''

import fastai


from collections import ChainMap

from fastai.tabular.all import (
    LinBnDrop,
    Module
)


class ResidualBlock(Module):
    '''
        A simple residule block that creates a skip connection around any input module
        Output size of the input module must match the module's input size
    '''
    def __init__(self, module, layer):
        self.module = module
        self.layer = layer

    def forward(self, inputs):
        fx = self.module(inputs)
        if(inputs.shape != fx.shape):
            print('mismatch at layer:', self.layer ,inputs.shape, fx.shape)
            assert False
        return fx + inputs


class CardinalResidualBlock(Module):
    '''
        A residule block that creates a skip connection around a set of n branches
            where the number of branches is determined by the number of input modules
            in the branches list parameter.

        The output of the branches is summed together along with the input
        Output size of the input module must match the module's input size
    '''
    def __init__(self, branches: list, layer: int):
        self.branches = branches
        self.layer = layer

    def forward(self, inputs):
        fx = self.branches[0](inputs)
        if(inputs.shape != fx.shape):
            print('mismatch at layer:', self.layer ,inputs.shape, fx.shape)
            assert False
        if(len(self.branches) > 1):
            for i in range(len(self.branches) - 1):
                fx += self.branches[i + 1](inputs)

        return fx + inputs

# currently need to create a bottlenecking block that can be used to reduce the number of inputs
#   being passed by the residual connection 

class BottleneckResidualBlock(Module):
    '''
        A residule block that creates a skip connection around a set of n branches
            where the number of branches is determined by the number of input modules
            in the branches list parameter.

            the residual connection is put through a linear batchnormed layer if the
            input size is different from the output size
            Then, the output of the branches is summed together along with the possibly transformed input
    '''
    def __init__(self, branches: list, layer: int, in_size: int, out_size: int):
        self.branches = branches
        self.layer = layer

        self.in_size = in_size
        self.out_size = out_size

        # self.linear = nn.Linear(in_size, out_size)
        self.linear = LinBnDrop(in_size, out_size)

    def forward(self, inputs):

        fx = self.branches[0](inputs)
        for i in range(len(self.branches) - 1):
            fx += self.branches[i + 1](inputs)

        if(inputs.shape != fx.shape):
            inputs = self.linear(inputs)
        return fx + inputs



def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
    })

    return versions

