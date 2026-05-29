import os
from itertools import chain
from functools import partial
from copy import deepcopy
from typing import Callable, Optional, Any, Self, Literal
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field, InitVar

import torch
from torch import Tensor
from torch.func import functional_call, stack_module_state
from torch.func import vmap, grad_and_value
from torch.nn import Module, ModuleList, Identity
from torch.nn.utils import skip_init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from optree import PyTree, PyTreeTypeVar, PyTreeSpec
from optree import tree_map, tree_map_, tree_transpose_map, tree_iter
from optree import tree_flatten, tree_unflatten, tree_flatten_one_level
from optree import tree_structure, treespec_accessors
from optree import tree_transpose, treespec_one_level
from optree import treespec_list, treespec_dict, treespec_leaf

from torchstrap.optimizer import GradientTransformation
from torchstrap.callbacks import Callback, PrintLog, PassThroughScoring, EpochTimer
from torchstrap.state import State
from torchstrap.history import History
from torchstrap.utils.typing import LoaderT


class StatelessModule(Module):
    _compiled_eval: Optional[Callable] = None
    _compiled_eval_wgrad: Optional[Callable] = None

    def __init__(self, base_model: Module):
        super().__init__()
        self._base_model = base_model.to(device="meta")

    @classmethod
    def init(
        cls,
        model: Module | Callable[..., Module],
        optimizer: GradientTransformation,
        /,
        *,
        num_replicas: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        init_randomness: Literal["same", "different"] = "same",
        model_init_args: Optional[tuple] = None,
        model_init_kwargs: Optional[dict[str, Any]] = None,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        if model_init_args is None:
            model_init_args = ()
        if model_init_kwargs is None:
            model_init_kwargs = {}
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if isinstance(model, Module):
            base_model = model
            models = ModuleList(
                deepcopy(base_model)
                .to(dtype=dtype, device=device)
                .requires_grad_(requires_grad)
                for _ in range(num_replicas)
            )
        else:
            match init_randomness:
                case "same":
                    base_model = model(
                        *model_init_args,
                        **model_init_kwargs,
                    )
                    models = ModuleList(
                        deepcopy(base_model)
                        .to(dtype=dtype, device=device)
                        .requires_grad_(requires_grad)
                        for _ in range(num_replicas)
                    )
                case "different":
                    models = ModuleList(
                        model(
                            *model_init_args,
                            dtype=dtype,
                            device=device,
                            **model_init_kwargs,
                        ).requires_grad_(False)
                        for _ in range(num_replicas)
                    )
                    base_model = models[0]

                case _:
                    raise ValueError(
                        "``model`` positional argument given Callable that is NOT torch.nn.Module."
                        f"But ``model_init_randomness`` keyword argument got {init_randomness} "
                        "(value other than [``same``, ``different``]). Either provide an initialized "
                        "torch.nn.Module or pass ``model_init_randomness`` kwarg properly."
                    )

        batch_size = (num_replicas,) if num_replicas > 1 else ()
        params_and_buffers_dicts = stack_module_state(models)
        optim, state = optimizer(
            params_and_buffers_dicts, batch_size=batch_size, **optimizer_kwargs
        )

        return cls(base_model), optim, state

    @property
    def _default_callbacks(self):
        return [
            ("epoch_timer", EpochTimer()),
            (
                "train_loss",
                PassThroughScoring(
                    name="train_loss",
                    on_train=True,
                ),
            ),
            (
                "valid_loss",
                PassThroughScoring(
                    name="valid_loss",
                ),
            ),
            ("print_log", PrintLog()),
        ]

    def get_default_callbacks(self):
        return self._default_callbacks

    def initialize_callbacks(
        self,
        callbacks: list[tuple[str, Callback]],
    ):
        _print_cbs = []
        _callbacks = []
        _callback_iter = chain(self._default_callbacks, callbacks)
        for i_cb, (cb_name, cb) in enumerate(_callback_iter):
            if isinstance(cb, PrintLog):
                _print_cbs.append((cb_name, cb))
                continue
            _callbacks.append((cb_name, cb))

        _callbacks.extend(_print_cbs)

        for cb_name, cb in _callbacks:
            cb.initialize()

        return _callbacks

    def notify(
        self,
        method_name,
        state,
        history,
        callbacks: Optional[list[tuple[str, Callback]]] = None,
        **kwargs,
    ):
        getattr(self, method_name)(state, history, **kwargs)
        if callbacks is not None:
            for cb_name, cb in callbacks:
                getattr(cb, method_name)(state, history, **kwargs)

    # pylint: disable=unused-argument
    def on_train_begin(self, state, history, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_train_end(self, state, history, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_epoch_begin(self, state, history, **kwargs):
        history.new_epoch()

    # pylint: disable=unused-argument
    def on_epoch_end(self, state, history, **kwargs):
        pass

    # pylint: disable=unused-argument
    def on_batch_begin(self, state, history, batch=None, training=False, **kwargs):
        history.new_batch()

    def on_batch_end(self, state, history, batch=None, training=False, **kwargs):
        pass

    def on_grad_computed(
        self, state, history, named_parameters, batch=None, training=False, **kwargs
    ):
        pass

    @torch.inference_mode()
    def predict(
        self,
        state: State,
        iterator: LoaderT,
        *,
        in_dims: int | tuple = 0,
        out_dims: int | tuple = 0,
        randomness: str = "error",
        chunk_size: Optional[int] = None,
        concat_dim: int = 1,
        is_training: bool = False,
        **kwargs,
    ):
        # Switch _base_model to eval so Dropout/BatchNorm submodules become
        # deterministic no-ops; otherwise dropout fires inside vmap and trips
        # randomness="error".
        self.train(is_training)
        return torch.cat(
            [
                vmap(
                    self.__call__,
                    in_dims=in_dims,
                    out_dims=out_dims,
                    randomness=randomness,
                    chunk_size=chunk_size,
                )(
                    state.param_dict,
                    state.buffer_dict,
                    input.to(
                        device=state.device,
                        dtype=state.dtype,
                    ),
                    **kwargs,
                )
                .detach()
                .cpu()
                for input, _, _ in iterator
            ],
            dim=concat_dim,
        )

    @torch.inference_mode()
    def infer(
        self,
        state: State,
        *args: Any,
        in_dims: int | tuple = 0,
        out_dims: int | tuple = 0,
        randomness: str = "error",
        chunk_size: Optional[int] = None,
        is_training: bool = False,
        **kwargs,
    ):
        self.train(is_training)
        return (
            vmap(
                self.__call__,
                in_dims=in_dims,
                out_dims=out_dims,
                randomness=randomness,
                chunk_size=chunk_size,
            )(
                state.param_dict,
                state.buffer_dict,
                *args,
                **kwargs,
            )
            .detach()
            .cpu()
        )

    @torch.no_grad()
    def fit(
        self,
        optimizer,
        criterion: Callable,
        state: State,
        train_iterator: LoaderT,
        *,
        valid_iterator: Optional[LoaderT] = None,
        num_epochs: int = 1,
        callbacks: Optional[list[tuple[str, Callback]]] = None,
        history: Optional[History] = None,
        **kwargs,
    ):
        callbacks = [] if callbacks is None else callbacks
        callbacks = self.initialize_callbacks(callbacks)

        if history is None:
            history = History()

        elif len(history) != state.num_replicas:
            raise ValueError(
                "lengths of history and state collections must be the same"
                f" but got {state.num_replicas} for state"
                f" and {len(history)} for history"
            )

        self.notify(
            "on_train_begin",
            state,
            history,
            callbacks,
        )

        try:
            self.fit_loop(
                optimizer,
                criterion,
                state,
                train_iterator,
                valid_iterator,
                history,
                callbacks,
                num_epochs=num_epochs,
                **kwargs,
            )
        except KeyboardInterrupt:
            pass

        self.notify(
            "on_train_end",
            state,
            history,
            callbacks,
        )
        return history

    def fit_loop(
        self,
        optimizer,
        criterion: Callable,
        state: State,
        train_iterator: LoaderT,
        valid_iterator: Optional[LoaderT],
        history: History,
        callbacks: Optional[list[tuple[str, Callback]]],
        *,
        num_epochs,
        **kwargs,
    ):

        for iepoch in range(num_epochs):
            self.notify(
                "on_epoch_begin",
                state,
                history,
                callbacks,
            )

            self.train(True)
            self.run_single_epoch(
                optimizer,
                criterion,
                state,
                train_iterator,
                history,
                callbacks,
                **kwargs,
            )

            if valid_iterator is not None:
                self.train(False)
                with torch.no_grad():
                    self.run_single_epoch(
                        optimizer,
                        criterion,
                        state,
                        valid_iterator,
                        history,
                        callbacks,
                        **kwargs,
                    )

            self.notify("on_epoch_end", state, history, callbacks)

    def run_single_epoch(
        self,
        optimizer,
        criterion: Callable,
        state: State,
        iterator: LoaderT,
        history: History,
        callbacks: Optional[list[tuple[str, Callback]]],
        **kwargs,
    ):
        for input, target, sample_weight in iterator:
            self.notify(
                "on_batch_begin",
                state,
                history,
                callbacks,
            )

            batch_loss = self.step(
                optimizer,
                criterion,
                state,
                input.to(
                    device=state.device,
                    dtype=state.dtype,
                ),
                target.to(
                    device=state.device,
                    dtype=state.dtype,
                ),
                sample_weight.to(
                    device=state.device,
                    dtype=state.dtype,
                )
                if sample_weight is not None
                else None,
                **kwargs,
            )

            prefix = "train" if self.training else "valid"
            history.append_batch(
                f"{prefix}_loss",
                batch_loss.detach().cpu(),
            )
            history.append_batch(
                f"{prefix}_batch_size",
                torch.tensor(input.shape[0]),
            )

            self.notify(
                "on_batch_end",
                state,
                history,
                callbacks,
            )

    def step(
        self,
        optimizer,
        criterion,
        state,
        *args,
        **kwargs,
    ):
        if self.training:
            grads, loss = self.evaluate_and_gradient(
                criterion,
                state.param_dict,
                state.buffer_dict,
                *args,
                in_dims=state.vmap_in_dims(args),
                **kwargs,
            )

            state.set_gradients(grads)
            optimizer.apply_gradient(state)
            return loss
        else:
            loss = self.evaluate(
                criterion,
                state.param_dict,
                state.buffer_dict,
                *args,
                in_dims=state.vmap_in_dims(args),
                **kwargs,
            )
            return loss

    def evaluate_and_gradient(self, *args, **kwargs):
        if self._compiled_eval_wgrad is not None:
            return self._compiled_eval_wgrad(*args, **kwargs)
        else:
            return self.vmapped_forward_eval_wgrad(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if self._compiled_eval is not None:
            return self._compiled_eval(*args, **kwargs)
        else:
            return self.vmapped_forward_eval(*args, **kwargs)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        super(Module, self).__setattr__(
            "_compiled_eval",
            torch.compile(self.vmapped_forward_eval, *args, **kwargs),
        )

        super(Module, self).__setattr__(
            "_compiled_eval_wgrad",
            torch.compile(self.vmapped_forward_eval_wgrad, *args, **kwargs),
        )

    def vmapped_forward_eval_wgrad(
        self,
        criterion,
        *args,
        in_dims=0,
        out_dims=0,
        randomness="error",
        chunk_size=None,
        argnums=0,
        has_aux=False,
        **kwargs,
    ):
        return vmap(
            grad_and_value(
                partial(self.forward_eval, criterion),
                argnums=argnums,
                has_aux=has_aux,
            ),
            in_dims=in_dims,
            out_dims=out_dims,
            randomness=randomness,
            chunk_size=chunk_size,
        )(*args, **kwargs)

    def vmapped_forward_eval(
        self,
        criterion,
        *args,
        in_dims=0,
        out_dims=0,
        randomness="error",
        chunk_size=None,
        **kwargs,
    ):
        return vmap(
            partial(self.forward_eval, criterion),
            in_dims=in_dims,
            out_dims=out_dims,
            randomness=randomness,
            chunk_size=chunk_size,
        )(*args, **kwargs)

    def forward_eval_wgrad(
        self,
        criterion: Callable,
        *args,
        argnums=0,
        has_aux=False,
        **kwargs,
    ):
        return grad_and_value(
            partial(self.forward_eval, criterion),
            argnums=argnums,
            has_aux=has_aux,
        )(*args, **kwargs)

    def forward_eval(
        self,
        criterion: Callable,
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        X: Tensor,
        y: Tensor,
        sample_weight: Optional[Tensor],
        *,  # *args,
        model_call_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        prediction = self._call_impl(
            params,
            buffers,
            X,
            **model_call_kwargs,
        )
        return criterion(prediction, y, sample_weight, **kwargs)

    # becomes the _call_impl method
    def forward(
        self,
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        *args,
        non_linearity: Callable = Identity(),
        **kwargs,
    ):
        return non_linearity(
            functional_call(
                self._base_model,
                (params, buffers),
                args,
                kwargs,
            )
        )
