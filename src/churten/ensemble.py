import os
from copy import deepcopy
from typing import Callable, Optional, Any
from collections import OrderedDict
from collections.abc import Iterator

import torch
from torch import Tensor

from torch.func import functional_call, stack_module_state
from torch.func import vmap, grad_and_value

from torch.nn import Module, ModuleList, Identity
from torch.nn.utils import skip_init
import torch.nn.functional as F

from tensordict import TensorDict

from churten.optimizer import GradientTransformations
from churten.utils import params_for
from churten.nn.archs import TensorCallable
from churten.callbacks import PrintLog

from churten.history import History

def replicate_module(
    module : Module,
    num_replicas : int = 1,
    *,
    device : str | torch.device = "cpu", 
    dtype : torch.dtype = torch.float32,
    requires_grad : bool = False,
):
    return ModuleList(deepcopy(module) \
                .to(dtype=dtype, device=device) \
                .requires_grad_(requires_grad) 
            for _ in range(num_replicas)
        )

class Ensemble(Module):
    def __init__(
        self,
        modeller : Module | Callable[..., Module],
        criterion : TensorCallable,
        num_replicas : int = 1,
        *,
        model_init_args : tuple = (),
        model_init_kwargs : dict = {},
        model_init_randomness : str = "same",
        device : str | torch.device = "cpu",
        dtype : torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = dtype
        self.num_replicas = num_replicas

        if isinstance(modeller, Module):
            _single_model = modeller
            _models = replicate_module(
                _single_model, self.num_replicas, device=self.device, dtype=self.dtype,
            )
        elif model_init_randomness == "same":
            _single_model = modeller(*model_init_args, **model_init_kwargs)
            _models = replicate_module(
                _single_model, self.num_replicas, device=self.device, dtype=self.dtype,
            )
        else:
            _models = ModuleList(
                modeller(
                    *model_init_args, 
                    dtype=self.dtype, 
                    device=self.device, 
                    **model_init_kwargs
                ).requires_grad_(False) 
                for _ in range(self.num_replicas)
            )
            _single_model = _models[0]


        self._params_dict, self._buffers_dict = stack_module_state(_models)
        self._base_model = _single_model.to(device="meta")

        self.criterion = criterion
        
        
        self._compiled_eval = None
        self._compiled_eval_wgrad = None

        self.num_iterations = 0
        self.last_checkpoint = ""

    @property
    def params_and_buffers_dicts(self):
        return (self._params_dict, self._buffers_dict)

    def save_checkpoint(
        self, 
        path : str,
    ):
        os.makedirs(path, exist_ok=True)

        self.last_checkpoint = path
        TensorDict(self._params_dict, device=self.device).memmap_(f"{path}/params")
        TensorDict(self._buffers_dict, device=self.device).memmap_(f"{path}/buffers")

    def load_checkpoint(
        self, 
        path : Optional[str] = None, 
        **kwargs,
    ):
        path = path or self.last_checkpoint
        if path == "":
            raise ValueError("No checkpoints have been saved so please pass a path")                                                                                     
        
        self._params_dict = TensorDict.load_memmap(
            f"{path}/params", device=self.device,
        ).to_dict()

        self._buffers_dict = TensorDict.load_memmap(
            f"{path}/buffers", device=self.device,
        ).to_dict()


    def predict(
        self,
        iterator : Iterator[Any],
        concat_dim = 1,
        **kwargs,
    ):
        #self.to_device(self.device)
        with torch.inference_mode():
            return torch.cat([
                vmap(self.__call__)(
                    self._params_dict,
                    self._buffers_dict,
                    input.to(
                        device = self.device,
                        dtype = self.dtype,
                    ),
                    **kwargs,
                ).detach().cpu() for input, _, _ in iterator
            ], dim=concat_dim)

    def infer(
        self,
        *args,
        **kwargs,
    ):
        with torch.inference_mode():
            return vmap(self.__call__)(
                self._params_dict,
                self._buffers_dict,
                *args,
                **kwargs,
            ).detach().cpu()

    def fit(
        self,
        optimizer : GradientTransformations,
        train_iterator : Iterator,
        valid_iterator : Iterator,
        num_epochs : int = 1,
        prefix : str = "model",
        **kwargs,
    ):
        if self.num_iterations > 0:
            self.load_checkpoint()
            
        dir = f"{prefix}/iteration_{self.num_iterations}"

        history = History(
            num_replicas=self.num_replicas, 
            path = f"{dir}/history",
        )
        print_log = PrintLog().initialize()

        optimizer.init(self._params_dict)

        for iepoch in range(num_epochs):
            self.train(True)
            train_losses = self.fit_step(optimizer, train_iterator, **kwargs)
            
            self.train(False)
            with torch.inference_mode():
                valid_losses = self.fit_step(optimizer, valid_iterator, **kwargs)

            history.append(
                train_loss = train_losses.cpu(),
                valid_loss = valid_losses.cpu(),
            )

            del train_losses
            del valid_losses
            
            print_log.on_epoch_end(history)

        self.save_checkpoint(dir)
        self.num_iterations += 1

        history.flush()

        return history

    def to_device(self, device):
        self._params_dict = TensorDict(self._params_dict, device=device).to_dict()
        self._buffers_dict = TensorDict(self._buffers_dict, device=device).to_dict()

    def fit_step(
        self,
        optimizer : GradientTransformations,
        iterator : Iterator[Any],
        **kwargs,
    ):
        batch_losses = []
        for input, target, sample_weight in iterator:
            batch_loss, batch_grads = self.step(
                optimizer,
                input.to(device=self.device),
                target.to(device=self.device),
                weight = sample_weight.to(device=self.device) if sample_weight is not None else None,
                **kwargs,
            )

            batch_losses.append(batch_loss)

        return torch.stack(batch_losses, dim=1)


    def step(
        self,
        optimizer : GradientTransformations,
        *args, **kwargs,
    ):
        if self.training:
            grads, loss = self.evaluate_and_gradient(
                self._params_dict, 
                self._buffers_dict, 
                *args, 
                **kwargs
            )
            optimizer.apply_gradients(grads)
            return loss, grads
        else:
            loss = self.evaluate(
                self._params_dict, 
                self._buffers_dict, 
                *args, 
                **kwargs
            )
            return loss, None

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
        self._compiled_eval = torch.compile(
            self.vmapped_forward_eval, *args, **kwargs
        )
        self._compiled_eval_wgrad = torch.compile(
            self.vmapped_forward_eval_wgrad, *args, **kwargs
        )

    def vmapped_forward_eval_wgrad(
        self, 
        *args, 
        argnums = 0, 
        has_aux = False, 
        **kwargs,
    ):
        return vmap( 
            grad_and_value(
                self.forward_eval, 
                argnums = argnums, 
                has_aux = has_aux,
            )
        )(*args, **kwargs)

    def vmapped_forward_eval(
        self, *args, **kwargs,
    ):
        return vmap(self.forward_eval)(*args, **kwargs)


    def forward_eval_wgrad(
        self, *args, argnums = 0, has_aux = False, **kwargs,
    ):
        return grad_and_value(
            self.forward_eval, 
            argnums = argnums, 
            has_aux = has_aux,
        )(*args, **kwargs)

    def forward_eval(
        self,
        params : dict[str, Tensor],
        buffers : dict[str, Tensor],
        *args,
        model_call_kwargs : dict[str, Any] = {},
        **kwargs,
    ):
        prediction = self._call_impl(
            params, 
            buffers, 
            *args[:-1], 
            **model_call_kwargs,
        )
        return self.criterion(prediction, args[-1], **kwargs)

    #becomes the _call_impl method
    def forward(
        self, 
        params : dict[str, Tensor], 
        buffers : dict[str, Tensor], 
        *args, 
        non_linearity : TensorCallable = Identity(),
        **kwargs,
    ):
        return non_linearity(
            functional_call(
                self._base_model, 
                (params, buffers), 
                args, kwargs,
            )
        )
