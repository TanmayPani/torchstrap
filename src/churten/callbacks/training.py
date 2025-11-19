import os
import pickle
import warnings
from contextlib import suppress
from fnmatch import fnmatch
from functools import partial
from itertools import product
from copy import deepcopy

from beartype.typing import Callable

import torch

from .callbacks import Callback
from churten.utils import _check_f_arguments
from churten.utils import noop
from churten.utils import open_file_like

class Checkpoint(Callback):
    """Save the model during training if the given metric improved.
    """
    def __init__(
            self,
            monitor='valid_loss_best',
            f_params='params.pt',
            f_optimizer='optimizer.pt',
            f_criterion='criterion.pt',
            f_history='history.json',
            f_pickle=None,
            fn_prefix='',
            dirname='',
            event_name='event_cp',
            sink=noop,
            load_best=False,
            verbose=True,
            **kwargs
    ):
        self.monitor = monitor
        self.f_params = f_params
        self.f_optimizer = f_optimizer
        self.f_criterion = f_criterion
        self.f_history = f_history
        self.f_pickle = f_pickle
        self.fn_prefix = fn_prefix
        self.dirname = dirname
        self.event_name = event_name
        self.sink = sink
        self.load_best = load_best
        self.verbose = verbose
        self._check_kwargs(self, kwargs)
        vars(self).update(**kwargs)
        self._validate_filenames()

    @classmethod
    def _check_kwargs(cls, instance, kwargs):
        for key in kwargs:
            if not key.startswith('f_'):
                raise TypeError(
                    "{cls_name} got an unexpected argument '{key}', did you mean "
                    "'f_{key}'?".format(cls_name=instance.__class__.__name__, key=key))

    def initialize(self):
        self._validate_filenames()
        if self.dirname and not os.path.exists(self.dirname):
            os.makedirs(self.dirname, exist_ok=True)
        return self

    def on_train_end(self, state, history, **kwargs):
        if not self.load_best or self.monitor is None:
            return
        self._sink(
            "Loading best checkpoint after training.", 
            verbose = self.verbose,
        )

    def on_epoch_end(self, state, history, **kwargs):
        if "{}_best".format(self.monitor) in history:
            warnings.warn(
                "Checkpoint monitor parameter is set to '{0}' and the history "
                "contains '{0}_best'. Perhaps you meant to set the parameter "
                "to '{0}_best'".format(self.monitor), UserWarning)

        if self.monitor is None:
            do_checkpoint = True
        elif callable(self.monitor):
            do_checkpoint = self.monitor(state, history, **kwargs)
        else:
            try:
                do_checkpoint = history[self.monitor][-1]
            except KeyError as e:
                msg = (
                    f"{e.args[0]} Make sure you have validation data if you use "
                    "validation scores for checkpointing.")
                raise Exception(msg)

        if self.event_name is not None:
            history.append(self.event_name, bool(do_checkpoint))

        if do_checkpoint:
            self.save_model(state, history)
            self._sink("A checkpoint was triggered in epoch {}.".format(
                history.num_epochs()
            ), self.verbose)

    def _f_kwargs(self):
        return {key: getattr(self, key) for key in dir(self)
                if key.startswith('f_') and (key != 'f_history_')}

    def save_model(self, state, history):
        """Save the model.

        This function saves some or all of the following:

          - model parameters;
          - optimizer state;
          - criterion state;
          - training history;
          - custom modules;
          - entire model object.

        """
        kwargs_module, kwargs_other = _check_f_arguments(
            self.__class__.__name__, **self._f_kwargs())

        for key, val in kwargs_module.items():
            if val is None:
                continue

            f = self._format_target(state, history, val, -1)
            key = key[:-1]  # remove trailing '_'
            self._save_params(f, state, history, 'f_' + key, key + " state")

        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            f = self.f_history_
            self._save_params(f, state, history, "f_history", "history")

        f_pickle = kwargs_other.get('f_pickle')
        if f_pickle:
            f_pickle = self._format_target(state, history, f_pickle, -1)
            with open_file_like(f_pickle, 'wb') as f:
                pickle.dump(state, history, f)

    @property
    def f_history_(self):
        # This is a property and not in initialize to allow ``NeuralNet``
        # to call ``load_params`` without needing the checkpoint to
        # by initialized.
        if self.f_history is None:
            return None
        return os.path.join(
            self.dirname, self.fn_prefix + self.f_history)

    def get_formatted_files(self, state, history):
        """Returns a dictionary of formatted filenames"""
        idx = -1
        if (
                self.event_name is not None and history
        ):
            for i, v in enumerate(history[self.event_name]):
                if v:
                    idx = i

        return {key: self._format_target(state, history, val, idx) for key, val
                in self._f_kwargs().items()}

    def _save_params(self, f, state, history, f_name, log_name):
        try:
            state.save_params(**{f_name: f})
        except Exception as e:  # pylint: disable=broad-except
            self._sink(
                "Unable to save {} to {}, {}: {}".format(
                    log_name, f, type(e).__name__, e), self.verbose)

    def _format_target(self, state, history, f, idx):
        """Apply formatting to the target filename template."""
        if f is None:
            return None
        if isinstance(f, str):
            f = self.fn_prefix + f.format(
                state=state,
                last_epoch=history[:][idx] ,  # TODO
                last_batch=history['batches'][:][idx][-1],
            )
            return os.path.join(self.dirname, f)
        return f

    def _validate_filenames(self):
        """Checks if passed filenames are valid.

        Specifically, f_* parameter should not be passed in
        conjunction with dirname.

        """
        _check_f_arguments(self.__class__.__name__, **self._f_kwargs())

        if not self.dirname:
            return

        def _is_truthy_and_not_str(f):
            return f and not isinstance(f, str)

        if any(_is_truthy_and_not_str(val) for val in self._f_kwargs().values()):
            raise Exception(
                'dirname can only be used when f_* are strings')

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)

class LoadInitState(Callback):
    """Loads the model, optimizer, and history from a checkpoint into a
    :class:`.NeuralNet` when training begins.
    """
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def initialize(self):
        self.did_load_ = False
        return self

    def on_train_begin(self, state, history,
                       X=None, y=None, **kwargs):
        if not self.did_load_:
            self.did_load_ = True
            with suppress(FileNotFoundError):
                if isinstance(self.checkpoint, TrainEndCheckpoint):
                    state.load_params(
                        checkpoint=self.checkpoint.checkpoint_,
                    )
                else:
                    state.load_params(
                        checkpoint=self.checkpoint,
                    )

class TrainEndCheckpoint(Callback):
    """Saves the model parameters, optimizer state, and history at the end of
    training. The default ``fn_prefix`` is ``'train_end_'``.
    """
    def __init__(
            self,
            f_params='params.pt',
            f_optimizer='optimizer.pt',
            f_criterion='criterion.pt',
            f_history='history.json',
            f_pickle=None,
            fn_prefix='train_end_',
            dirname='',
            sink=noop,
            verbose=True,
            **kwargs
    ):
        self.f_params = f_params
        self.f_optimizer = f_optimizer
        self.f_criterion = f_criterion
        self.f_history = f_history
        self.f_pickle = f_pickle
        self.fn_prefix = fn_prefix
        self.dirname = dirname
        self.verbose = verbose
        self.sink = sink
        Checkpoint._check_kwargs(self, kwargs)
        vars(self).update(**kwargs)

    def _f_kwargs(self):
        return {name: getattr(self, name) for name in dir(self)
                if name.startswith('f_')}

    def initialize(self):
        self.checkpoint_ = Checkpoint(
            monitor=None,
            fn_prefix=self.fn_prefix,
            dirname=self.dirname,
            event_name=None,
            sink=self.sink,
            **self._f_kwargs()
        )
        self.checkpoint_.initialize()
        return self

    def on_train_end(self, state, history, **kwargs):
        self.checkpoint_.save_model(state, history)
        self.checkpoint_._sink("Final checkpoint triggered", self.verbose)
        return self

class EarlyStopping(Callback):
    """Callback for stopping training when scores don't improve.
    """
    def __init__(
            self,
            checkpoint,
            monitor='valid_loss',
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
            sink=print,
            load_best=False,
            verbose=True,
    ):
        self.checkpoint = checkpoint
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.misses_ = 0
        self.dynamic_threshold_ = None
        self.sink : Callable = sink
        self.load_best = load_best
        self.verbose = verbose

    def __getstate__(self):
        # Avoids to save the module_ weights twice when pickling
        state = self.__dict__.copy()
        state['best_model_weights_'] = None
        return state

    # pylint: disable=arguments-differ
    def on_train_begin(self, state, history, **kwargs):
        if self.threshold_mode not in ['rel', 'abs']:
            raise ValueError(
                "Invalid threshold mode: '{}'".format(self.threshold_mode)
            )
        self.misses_ = torch.zeros(state.num_replicas, dtype=torch.int) 
        self.dynamic_threshold_ = float("inf") if self.lower_is_better else float("-inf")
        self.best_model_weights_ = None
        self.best_epoch_ = torch.zeros(state.num_replicas, dtype=torch.int)

    def on_epoch_end(self, state, history, **kwargs):
        current_score = history[self.monitor][-1]
        is_score_improved = self._is_score_improved(current_score)
        self.misses_ = torch.where(is_score_improved, 0, self.misses_+1)
        
        self.dynamic_threshold_ = self._calc_new_threshold(current_score, is_score_improved)
        self.best_epoch_[is_score_improved] = history["iepoch"][-1]
        if self.load_best:
            self.best_model_weights_ = deepcopy(state.state_dict()) # TODO

        models_to_keep = []
        for i, misses in enumerate(self.misses_.tolist()):
            if misses == self.patience:
                models_to_keep.append(state.active_models[i]) 
                if self.verbose:
                    self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=self.verbose)

    def on_train_end(self, state, history, **kwargs):
        if (
            self.load_best and (self.best_epoch_ != history["epoch"][-1])
            and (self.best_model_weights_ is not None)
        ):
            state.load_state_dict(self.best_model_weights_) #TODO
            self._sink("Restoring best model from epoch {}.".format(
                self.best_epoch_
            ), verbose=self.verbose)

    def _is_score_improved(self, score : torch.Tensor) -> torch.Tensor:
        if self.lower_is_better:
            return score.lt(self.dynamic_threshold_)
        return score.gt(self.dynamic_threshold_)

    def _calc_new_threshold(self, score : torch.Tensor, is_improved : torch.Tensor) -> torch.Tensor:
        """Determine threshold based on score."""
        if self.threshold_mode == 'rel':
            abs_threshold_change = torch.where(is_improved, self.threshold * score, 0) 
        else:
            abs_threshold_change = torch.where(is_improved, self.threshold, 0)
        if self.lower_is_better:
            new_threshold = score - abs_threshold_change
        else:
            new_threshold = score + abs_threshold_change
        return new_threshold

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)

