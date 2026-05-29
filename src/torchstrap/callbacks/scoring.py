""" Callbacks for calculating scores."""

from contextlib import contextmanager
from contextlib import suppress
from functools import partial
import warnings

import math

import torch

from torchstrap.utils import to_numpy
from torchstrap.utils import to_device

from .callbacks import Callback

__all__ = ["PassThroughScoring"]

class PassThroughScoring(Callback):
    """Creates scores on epoch level based on batch level scores
    """
    def __init__(
            self,
            name,
            lower_is_better=True,
            on_train=False,
    ):
        self.name = name
        self.lower_is_better = lower_is_better
        self.on_train = on_train

    def initialize(self):
        self.best_score_ = torch.tensor(float("inf")) if self.lower_is_better else torch.tensor(float("-inf"))
        return self

    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score.lt(self.best_score_)
        return current_score.gt(self.best_score_)

    def get_avg_score(self, history):
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights = history['batches'][bs_key][-1]
        scores = history['batches'][self.name][-1]

        stacked_scores = torch.stack(scores)
        stacked_weights = torch.stack(weights).unsqueeze_(-1)


        w_sum = stacked_weights.sum()
        
        wxs_sum = stacked_scores.mul_(stacked_weights).sum(0)

        return wxs_sum.div_(w_sum)

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(self, state, history, **kwargs):
        if len(history['batches'].get(self.name, [[]])[-1]) == 0:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        self.best_score_ = torch.where(is_best, score_avg, self.best_score_)
        history.append(self.name, score_avg)
        if is_best is not None:
            history.append(self.name + '_best', is_best)

