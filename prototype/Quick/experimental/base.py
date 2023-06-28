'''
Here we are going to define object oriented machine learning experiment runners.

There will be the following parts:
    - Data input
    - Experiment objects
    - Experiment Runner
'''

import os


class Base(object):
    def __init__(self):
        pass

    def run(self):
        pass

    def process(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    