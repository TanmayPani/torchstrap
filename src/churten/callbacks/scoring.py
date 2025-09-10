""" Callbacks for calculating scores."""

from contextlib import contextmanager
from contextlib import suppress
from functools import partial
import warnings

import numpy as np
import sklearn
from sklearn.metrics import make_scorer, check_scoring
from sklearn.metrics._scorer import _BaseScorer

from sklearn_wrappers.torch.callbacks import Callback
from sklearn_wrappers.torch.utils import to_numpy
from sklearn_wrappers.torch.utils import to_device

__all__ = ['BatchScoring', 'EpochScoring', 'PassThroughScoring']

@contextmanager
def _cache_net_forward_iter(net, use_caching, y_preds):
    """Caching context for ``NeuralNet`` instance.
    """
    if net.use_caching != 'auto':
        use_caching = net.use_caching

    if not use_caching:
        yield net
        return
    y_preds = iter(y_preds)

    # pylint: disable=unused-argument
    def cached_forward_iter(*args, device=net.device, **kwargs):
        for yp in y_preds:
            yield to_device(yp, device=device)

    net.forward_iter = cached_forward_iter
    try:
        yield net
    finally:
        # By setting net.forward_iter we define an attribute
        # `forward_iter` that precedes the bound method
        # `forward_iter`. By deleting the entry from the attribute
        # dict we undo this.
        del net.__dict__['forward_iter']


def convert_sklearn_metric_function(scoring):
    """If ``scoring`` is a sklearn metric function, convert it to a
    sklearn scorer and return it. Otherwise, return ``scoring`` unchanged."""
    if callable(scoring):
        module = getattr(scoring, '__module__', None)

        if module is None:
            return

        # those are scoring objects returned by make_scorer starting
        # from sklearn 0.22
        scorer_names = ('_PredictScorer', '_ProbaScorer', '_ThresholdScorer', '_Scorer')
        if (
                hasattr(module, 'startswith') and
                module.startswith('sklearn.metrics.') and
                not module.startswith('sklearn.metrics.scorer') and
                not module.startswith('sklearn.metrics.tests.') and
                not scoring.__class__.__name__ in scorer_names
        ):
            return make_scorer(scoring)
    return scoring


class ScoringBase(Callback):
    """Base class for scoring.

    Subclass and implement an ``on_*`` method before using.
    """
    def __init__(
            self,
            scoring,
            lower_is_better=True,
            on_train=False,
            name=None,
            target_extractor=to_numpy,
            use_caching=True,
    ):
        self.scoring = scoring
        self.lower_is_better = lower_is_better
        self.on_train = on_train
        self.name = name
        self.target_extractor = target_extractor
        self.use_caching = use_caching

    # pylint: disable=protected-access
    def _get_name(self):
        """Find name of scoring function."""
        if self.name is not None:
            return self.name
        if self.scoring_ is None:
            return 'score'
        if isinstance(self.scoring_, str):
            return self.scoring_
        if isinstance(self.scoring_, partial):
            return self.scoring_.func.__name__
        if isinstance(self.scoring_, _BaseScorer):
            if hasattr(self.scoring_._score_func, '__name__'):
                # sklearn < 0.22
                return self.scoring_._score_func.__name__
            # sklearn >= 0.22
            return self.scoring_._score_func._score_func.__name__
        if isinstance(self.scoring_, dict):
            raise ValueError("Dict not supported as scorer for multi-metric scoring."
                             " Register multiple scoring callbacks instead.")
        return self.scoring_.__name__

    def initialize(self):
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        self.scoring_ = convert_sklearn_metric_function(self.scoring)
        self.name_ = self._get_name()
        return self

    # pylint: disable=attribute-defined-outside-init,arguments-differ
    def on_train_begin(self, net, **kwargs):
        #self.X_indexing_ = check_indexing(X)
        #self.y_indexing_ = check_indexing(y)

        # Looks for the right most index where `*_best` is True
        # That index is used to get the best score in `net.history`
        with suppress(ValueError, IndexError, KeyError):
            best_name_history = net.history[:, '{}_best'.format(self.name_)]
            idx_best_reverse = best_name_history[::-1].index(True)
            idx_best = len(best_name_history) - idx_best_reverse - 1
            self.best_score_ = net.history[idx_best, self.name_]

    def _scoring(self, net, X_test, y_test):
        """Resolve scoring and apply it to data. Use cached prediction
        instead of running inference again, if available."""
        scorer = check_scoring(net, self.scoring_)
        if scorer is None:
            return None
        return scorer(net, X_test, y_test)

    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score < self.best_score_
        return current_score > self.best_score_


class BatchScoring(ScoringBase):
    """Callback that performs generic scoring on batches.
    """
    # pylint: disable=unused-argument,arguments-differ

    def on_batch_end(self, net, batch = None, training = False, **kwargs):
        if training != self.on_train:
            return

        if batch is None:
            return

        X, y = batch.unpack_data()
        y_preds = [kwargs['y_pred']]
        with _cache_net_forward_iter(net, self.use_caching, y_preds) as cached_net:
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y=None.
            y = None if y is None else self.target_extractor(y)
            try:
                score = self._scoring(cached_net, X, y)
                cached_net.history.record_batch(self.name_, score)
            except KeyError:
                pass

    def get_avg_score(self, history):
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights, scores = list(zip(
            *history[-1, 'batches', :, [bs_key, self.name_]]))
        score_avg = np.average(scores, weights=weights)
        return score_avg

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        history = net.history
        try:  # don't raise if there is no valid data
            history[-1, 'batches', :, self.name_]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name_, score_avg)
        if is_best is not None:
            history.record(self.name_ + '_best', bool(is_best))


class EpochScoring(ScoringBase):
    """Callback that performs generic scoring on predictions.
    """
    def _initialize_cache(self):
        self.y_trues_ = []
        self.y_preds_ = []

    def initialize(self):
        super().initialize()
        self._initialize_cache()
        return self

    # pylint: disable=arguments-differ,unused-argument
    def on_epoch_begin(self, net, **kwargs):
        self._initialize_cache()

    # pylint: disable=arguments-differ
    def on_batch_end(
            self, net, batch = None, y_pred = None, training=False, **kwargs):
        use_caching = self.use_caching
        if net.use_caching !=  'auto':
            use_caching = net.use_caching

        if (not use_caching) or (training != self.on_train):
            return

        if batch is None:
            return
        _X, y, _, _ = batch.unpack_data()
        if y is not None:
            self.y_trues_.append(y)
        self.y_preds_.append(y_pred)

    def get_test_data(self, dataset_train, dataset_valid, use_caching):
        """Return data needed to perform scoring.
        """
        dataset = dataset_train if self.on_train else dataset_valid

        if use_caching:
            X_test = dataset
            y_pred = self.y_preds_
            y_test = [self.target_extractor(y) for y in self.y_trues_]
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y_test=None.
            y_test = np.concatenate(y_test) if y_test else None
            return X_test, y_test, y_pred

        X_test, y_test = dataset, None

        if y_test is not None:
            # We allow y_test to be None but the scoring function has
            # to be able to deal with it (i.e. called without y_test).
            y_test = self.target_extractor(y_test)
        return X_test, y_test, []

    def _record_score(self, history, current_score):
        """Record the current store and, if applicable, if it's the best score
        yet.

        """
        history.record(self.name_, current_score)

        is_best = self._is_best_score(current_score)
        if is_best is None:
            return

        history.record(self.name_ + '_best', bool(is_best))
        if is_best:
            self.best_score_ = current_score

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(
            self,
            net,
            dataset_train = None,
            dataset_valid = None,
            **kwargs):
        use_caching = self.use_caching
        if net.use_caching !=  'auto':
            use_caching = net.use_caching

        X_test, y_test, y_pred = self.get_test_data(
            dataset_train,
            dataset_valid,
            use_caching=use_caching,
        )
        if X_test is None:
            return

        with _cache_net_forward_iter(net, self.use_caching, y_pred) as cached_net:
            current_score = self._scoring(cached_net, X_test, y_test)

        self._record_score(net.history, current_score)

    def on_train_end(self, *args, **kwargs):
        self._initialize_cache()


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
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        return self

    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score < self.best_score_
        return current_score > self.best_score_

    def get_avg_score(self, history):
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights, scores = list(zip(
            *history[-1, 'batches', :, [bs_key, self.name]]))
        score_avg = np.average(scores, weights=weights)
        return score_avg

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(self, net, **kwargs):
        history = net.history
        try:  # don't raise if there is no valid data
            history[-1, 'batches', :, self.name]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name, score_avg)
        if is_best is not None:
            history.record(self.name + '_best', bool(is_best))
