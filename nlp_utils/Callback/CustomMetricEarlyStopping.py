from typing import Optional, Tuple, Callable
import numpy as np
import logging

from tensorflow.keras.callbacks import Callback


class CustomMetricEarlyStoppingCallback(Callback):
    def __init__(
            self, data: Tuple[np.array], training_data: Optional[Tuple[np.array]] = None,
            score_func: Callable = None, min_delta: float = 0, patience: int = 0, verbose: int = 0, mode: str = 'auto',
            baseline: float = None, restore_best_weights: bool = False):

        super().__init__()

        self.x_train: Optional[np.arary] = None
        self.y_train: Optional[np.arary] = None
        if training_data is not None:
            self.x_train, self.y_train = training_data

        self.x_valid, self.y_valid = data
        self.score_func = score_func

        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logging.warning(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = self.model.get_weights()
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.score_func(self.y_valid, self.model.predict(self.x_valid))
        if self.y_train is not None and self.x_train is not None:
            current_train = self.score_func(self.y_train, self.model.predict(self.x_train))
            diff = current_train - current
            print(
                f'\nEarlyStopping Metric: {current:.3f}, training: {current_train:.3f}, fitting diff: {diff:.3f}\n')
        else:
            print(
                f'\nEarlyStopping Metric: {current:.3f}, best: {self.best:.3f}\n')

        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = None
        return monitor_value
