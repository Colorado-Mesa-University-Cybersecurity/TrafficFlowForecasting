'''
Utility functions and values for plotting
'''

import colorsys

from collections import ChainMap


def get_n_color_list(n: int, opacity=.8, luminosity=.5, saturation=.5) -> list:
    '''
        Function creates a list of n distinct colors, formatted for use in a plotly graph
    '''

    colors = []

    for i in range(n):
        color = colorsys.hls_to_rgb(
                i / n,
                luminosity,
                saturation,
        )
        
        colors.append(f'rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, {opacity})')

    return colors


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({

    })

    return versions

