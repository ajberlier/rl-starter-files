import numpy as np


class EarlyStoppingOnPlateau():

    def __init__(self, mode='min', patience=10, threshold=1e-3, termination_callback=None):

        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.terminated = False
        self.termination_callback = termination_callback
        self.best = np.nan
        self.metric_values = list()
        self.is_equiv_to_best_epoch = False

    def is_done(self):
        return self.terminated

    def step(self, metrics):
        if self.mode not in ['min','max']:
            raise RuntimeError("Invalid mode: {}".format(self.mode))
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if np.isnan(self.best):
            self.best = current

        if np.isnan(self.best):
            self.best = current
        if self.mode == 'min' and current < self.best:
            self.best = current
        if self.mode == 'max' and current > self.best:
            self.best = current

        self.metric_values.append(current)

        error_from_best = np.abs(np.asarray(self.metric_values) - self.best)
        error_from_best[error_from_best < np.abs(self.threshold)] = 0
        if np.all(np.isnan(error_from_best)):
            return
        # unpack numpy array, select first time since that value has happened
        idx = error_from_best == 0
        if np.any(idx):
            best_metric_epoch = np.where(idx)[0][0]
        else:
            best_metric_epoch = 0

        # update the number of "bad" epochs. The (epoch-1) handles 0 based indexing vs natural counting of epochs
        num_bad_epochs = (len(self.metric_values) - 1) - best_metric_epoch
        # if this epoch is equivalent in loss to the best
        self.is_equiv_to_best_epoch = error_from_best[-1] == 0

        if num_bad_epochs >= self.patience:
            self.terminated = True
            if self.termination_callback is not None:
                self.termination_callback()
