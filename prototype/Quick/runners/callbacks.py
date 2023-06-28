'''

'''

import pathlib

import fastai
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from collections import ChainMap

from fastai.callback.all import Callback
from fastai.tabular.all import range_of
from fastai.basics import Tensor

from fastcore.all import L



class DFLogger(Callback):
    '''
        Class defines a callback that is passed to the fastai learner that
            will save the recorded metrics for each epoch to a dataframe
    '''

    order=60


    def __init__(self, fname='history.csv', append=False):
        self.fname,self.append = pathlib.Path(fname),append
        self.df = pd.DataFrame()
        self.flag = True


    def to_csv(self, file: str or None = None):
        if file is None:
            self.df.to_csv(self.path/self.fname, index=False)
        else:
            self.df.to_csv(file, index=False)


    def before_fit(self):
        "Prepare file with metric names."
        if hasattr(self, "gather_preds"): return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.old_logger,self.learn.logger = self.logger,self.add_row


    def add_row(self, log):
        "Write a line with `log` and call the old logger."
        if(self.flag):
            self.df = pd.DataFrame([log], columns=self.recorder.metric_names)
            self.flag = False
        else:
            if (len(log) == len(self.df.columns)):
                self.new_row = pd.DataFrame([log], columns=self.recorder.metric_names)
                self.df = pd.concat([self.df, self.new_row], ignore_index=True)


    def after_fit(self):
        "Close the file and clean up."
        if hasattr(self, "gather_preds"): return
        self.learn.logger = self.old_logger


class LazyGraphCallback(Callback):
    '''
        Class defines a callback that is passed to the fastai learner that
            saves the validation and training loss metrics to graph when
            calling the .plot_graph() method
        
        This allows us to display the train/validation loss graph after the    
            experiment is run, even if it is run in no_bar mode
    '''

    order: int = 65
    run_valid: bool = False
    

    def __init__(self):
        self.graph_params: list = []
        self.graphs: list = []


    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")
        if not self.run: return
        self.nb_batches: list = []


    def after_train(self): self.nb_batches.append(self.train_iter)


    def after_epoch(self):
        "Plot validation loss in the pbar graph"
        if not self.nb_batches: return
        rec = self.learn.recorder
        iters = range_of(rec.losses)
        val_losses = [v[1] for v in rec.values]
        x_bounds = (0, (self.n_epoch - len(self.nb_batches)) * self.nb_batches[0] + len(rec.losses))
        y_bounds = (0, max((max(Tensor(rec.losses)), max(Tensor(val_losses)))))
        self.graph_params.append(([(iters, rec.losses), (self.nb_batches, val_losses)], x_bounds, y_bounds))


    def plot_graph(self, ax: plt.Axes or None = None, title: str = 'Loss'):

        params = self.graph_params[-1]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.plot(params[0][0][0], params[0][0][1], label='Training')
        ax.plot(params[0][1][0], params[0][1][1], label='Validation')
        ax.legend()


class ModelStatsCallback(Callback):
    '''
        Collect all batches, along with `pred` and `loss`, into `self.datum`.
        This allows the data to be used after the model is used.
    '''

    order=60 # mysterious param we need to investigate


    def __init__(self):
        self.datum: list = []
        self.records: list = []
        self.c_idx: int  = -1


    def before_fit(self):
        self.datum.append(L())
        self.records.append({'loss': [], 'predictions': []})
        self.c_idx += 1


    def after_batch(self):
        vals = self.learn.to_detach((self.xb,self.yb,self.pred,self.loss))
        self.datum[self.c_idx].append(vals)
        self.records[self.c_idx]['predictions'].append(vals[2])
        self.records[self.c_idx]['loss'].append(vals[3])
        self.records[self.c_idx]['iters'] = range_of(self.recorder.losses)



def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'matplotlib': f'\t{mpl.__version__}',
        'pandas': f'\t\t{pd.__version__}'
    })

    return versions

