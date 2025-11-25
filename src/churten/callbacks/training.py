import os
import pickle
import warnings
from contextlib import suppress
from copy import deepcopy

from beartype.typing import Callable

import torch

from churten.callbacks import Callback
from churten.utils import noop, _check_f_arguments, open_file_like
from churten.utils.typing import BoolVector

class Checkpoint(Callback):
    """Save the model during training if the given metric improved.
    """
    def __init__(
            self,
            monitor='valid_loss_best',
            f_state='state_dict.pt',
            f_history='history.json',
            fn_prefix='',
            root_dir="checkpoint",
            event_name='event_cp',
            sink=noop,
            load_best=False,
            verbose=True,
            **kwargs,
    ):
        self.monitor = monitor
        self.f_state = f_state
        self.f_history = f_history
        self.fn_prefix = fn_prefix
        self.root_dir = root_dir
        self.event_name = event_name
        self.sink = sink
        self.load_best = load_best
        self.verbose = verbose

    def initialize(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)
        return self

    def on_train_end(self, state, history, **kwargs):
        if not self.load_best or self.monitor is None:
            return
        self._sink(
            "Loading best checkpoint after training.", 
            self.verbose,
        )

        state.from_file(self.root_dir, self.fn_prefix + self.f_state)
        history.from_file(os.path.join(self.root_dir, self.fn_prefix + self.f_history))


    def on_epoch_end(self, state, history, **kwargs):
        if f"{self.monitor}_best" in history:
            warnings.warn(
                f"Checkpoint monitor parameter is set to '{self.monitor}' and the history "
                f"contains '{self.monitor}_best'. Perhaps you meant to set the parameter "
                f"to '{self.monitor}_best'", UserWarning,
            )

        if self.monitor is None:
            do_checkpoint_list : list[bool] = state.model_status_list
        elif callable(self.monitor):
            do_checkpoint_list : list[bool] = self.monitor(state, history, **kwargs)
        else:
            try:
                do_checkpoint : BoolVector = history[self.monitor][-1]
                do_checkpoint_list : list[bool] = do_checkpoint.tolist()
                do_checkpoint_list[:] = [
                    do_chk and is_active 
                    for do_chk, is_active in zip(
                        do_checkpoint_list, state.model_status_list
                    )
                ]
            except KeyError as e:
                msg = (
                    f"{e.args[0]} Make sure you have validation data if you use "
                    "validation scores for checkpointing."
                )
                raise Exception(msg)

        if self.event_name is not None:
            history.append(
                self.event_name, torch.any(do_checkpoint).item()
            )

        history.to_file(
            os.path.join(self.root_dir, self.fn_prefix+self.f_history)
        )

        for imodel, do_chk in enumerate(do_checkpoint_list):
            if do_chk:
                state[imodel].to_file(
                    os.path.join(self.root_dir, f"replica_{imodel}"), 
                    self.fn_prefix+self.f_state,
                )
                self._sink(
                    f"A checkpoint was triggered in epoch {history.num_epochs}.", 
                    self.verbose,
                )

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

    def on_train_begin(
        self, state, history, **kwargs,
    ):
        if not self.did_load_:
            self.did_load_ = True
            with suppress(FileNotFoundError):
                if isinstance(self.checkpoint, TrainEndCheckpoint):
                    pass # TODO : implement!
                else:
                    pass 
                    
class TrainEndCheckpoint(Callback):
    """Saves the model parameters, optimizer state, and history at the end of
    training. The default ``fn_prefix`` is ``'train_end_'``.
    """
    def __init__(
            self,
            f_state='state_dict.pt',
            f_history='history.json',
            fn_prefix='train_end_',
            root_dir='',
            sink=noop,
            verbose=True,
            **kwargs
    ):
        self.f_state = f_state 
        self.f_history = f_history
        self.fn_prefix = fn_prefix
        self.root_dir = root_dir
        self.verbose = verbose
        self.sink = sink

    def initialize(self):
        self.checkpoint_ = Checkpoint(
            monitor=None,
            f_state = self.f_state,
            f_history = self.f_history,
            fn_prefix=self.fn_prefix,
            root_dir=self.root_dir,
            event_name=None,
            sink=self.sink,
        )
        self.checkpoint_.initialize()
        return self

    def on_train_end(self, state, history, **kwargs):
        self.checkpoint_.on_train_end(state, history, **kwargs)
        self.checkpoint_._sink("Final checkpoint triggered", self.verbose)
        return self

class EarlyStopping(Callback):
    """Callback for stopping training when scores don't improve.
    """
    def __init__(
        self,
        monitor='valid_loss',
        patience=5,
        threshold=1e-4,
        threshold_mode='rel',
        lower_is_better=True,
        sink=print,
        load_best=False,
        verbose=True,
    ):
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
        state['best_model_weights_'].clear()
        return state

    # pylint: disable=arguments-differ
    def on_train_begin(self, state, history, **kwargs):
        if self.threshold_mode not in ['rel', 'abs']:
            raise ValueError(
                f"Invalid threshold mode: '{self.threshold_mode}'"
            )
        self.misses_ = torch.zeros(state.num_replicas, dtype=torch.int) 
        self.dynamic_threshold_ = float("inf") if self.lower_is_better else float("-inf")
        self.best_model_weights_ = [None]*state.num_replicas
        self.best_epoch_ = torch.zeros(state.num_replicas, dtype=torch.int)

    def on_epoch_end(self, state, history, **kwargs):
        current_score = history[self.monitor][-1]
        is_score_improved = self._is_score_improved(current_score)
        self.misses_ = torch.where(is_score_improved, 0, self.misses_+1)
        
        self.dynamic_threshold_ = self._calc_new_threshold(current_score, is_score_improved)
        self.best_epoch_[is_score_improved] = history["iepoch"][-1]
        if self.load_best:
            is_best_epoch = is_score_improved.tolist()
            for imodel, is_best in enumerate(is_best_epoch):
                self.best_model_weights_[imodel] = deepcopy(state[imodel].state_dict())

        for i, misses in enumerate(self.misses_.tolist()):
            if misses == self.patience:
                state.mask_model(i) 
                if self.verbose:
                    self._sink(
                    f"Stopping module {i} since {self.monitor} has not "
                        f"improved in the last {self.patience} epochs.", 
                        verbose=self.verbose
                    )


    def on_train_end(self, state, history, **kwargs):
        if self.load_best:
            for imodel in range(state.num_replicas):
                if (
                    (self.best_epoch_[imodel] != history["epoch"][-1]) and \
                    (self.best_model_weights_[imodel] is not None)
                ):
                    state[imodel].load_state_dict(
                        self.best_model_weights_[imodel]
                    ) 
                    self._sink(
                        f"Restoring best model from epoch {self.best_epoch_[imodel]}.", 
                        verbose=self.verbose
                    )

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

