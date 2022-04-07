"""
Modified from ReduceLROnPlateau in pytorch that can be matched with PGDOptimizer.
"""
import math

from pipeline.modules.pgd_optimizer import PGDOptimizer

inf = math.inf


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self._factor = factor

        # Attach optimizer
        if not isinstance(optimizer, PGDOptimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self._min_lrs = min_lr
        self._patience = patience
        self._verbose = verbose
        self._cooldown = cooldown
        self._cooldown_counter = 0
        self._mode = mode
        self._threshold = threshold
        self._threshold_mode = threshold_mode
        self._best = None
        self._num_bad_epochs = None
        self._mode_worse = None  # the worse value for the chosen mode
        self._eps = eps
        self._last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

        self._last_lr = 0

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self._best = self._mode_worse
        self._cooldown_counter = 0
        self._num_bad_epochs = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self._last_epoch += 1

        if self._is_better(current, self._best):
            self._best = current
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1

        if self.in_cooldown:
            self._cooldown_counter -= 1
            self._num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self._num_bad_epochs > self._patience:
            self._reduce_lr(self._last_epoch)
            self._cooldown_counter = self._cooldown
            self._num_bad_epochs = 0

        self._last_lr = self.optimizer.super_params.get("alpha")

    def _reduce_lr(self, epoch):
        old_lr = float(self.optimizer.super_params.get("alpha"))
        new_lr = max(old_lr * self._factor, self._min_lrs)
        if old_lr - new_lr > self._eps:
            self.optimizer.super_params['alpha'] = new_lr
            if self._verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                      ' to {:.4e}.'.format(epoch_str, new_lr))

    @property
    def in_cooldown(self):
        return self._cooldown_counter > 0

    def _is_better(self, a, best):
        if self._mode == 'min' and self._threshold_mode == 'rel':
            rel_epsilon = 1. - self._threshold
            return a < best * rel_epsilon

        elif self._mode == 'min' and self._threshold_mode == 'abs':
            return a < best - self._threshold

        elif self._mode == 'max' and self._threshold_mode == 'rel':
            rel_epsilon = self._threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self._threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self._mode_worse = inf
        else:  # mode == 'max':
            self._mode_worse = -inf

        self._mode = mode
        self._threshold = threshold
        self._threshold_mode = threshold_mode
