from copy import deepcopy
from typing import Callable, Union, Protocol, Optional, Any
from typing import runtime_checkable
from collections.abc import Iterator, Mapping, Sequence

import torch

from torch import Tensor, device, no_grad

from torch.utils import _pytree

from torch.func import functional_call
from torch.func import vmap, grad_and_value

from torch.nn import Identity, Parameter
from torch.nn import ParameterDict
from torch.nn import Module
from torch.nn.utils import skip_init

import torch.nn.functional as F

from churten.optimizer import GradientTransformations
from churten.utils import params_for
from churten.nn.archs import TensorCallable 

class BaseModel(Module):
    def __init__(
        self,
        model : Module,
        criterion : TensorCallable,
        optimizer : GradientTransformations,
        device : str | torch.device = "cpu",
        dtype : torch.dtype = torch.float32,
        params : Optional[dict[str, Tensor]] = None,
        buffers : Optional[dict[str, Tensor]] = None,
    ):
        super().__init__()

        self.device = device
        
        self._params_dict = {
            key : param.detach().to(device=self.device, dtype=dtype).requires_grad_(False) for key, param in model.named_parameters()
        } if params is None else params

        self._buffers_dict = {
            key : buffer.detach().to(device=self.device, dtype=dtype).requires_grad_(False) for key, buffer in model.named_buffers()
        } if buffers is None else buffers
        
        self.model = model.to(device="meta").requires_grad_(False)
        self.criterion = criterion

        self.optimizer = optimizer
        optimizer.state.init(self._params_dict)

        self._compiled_eval = None
        self._compiled_eval_wgrad = None

    @property
    def params_and_buffers_dicts(self):
        return (self._params_dict, self._buffers_dict)

    #becomes the _call_impl method
    def forward(
        self, 
        params : dict[str, Tensor], 
        buffers : dict[str, Tensor], 
        *args, 
        non_linearity : TensorCallable = torch.nn.Identity(),
        **kwargs,
    ):
        return non_linearity(
            functional_call(
                self.model, 
                (params, buffers), 
                args, kwargs,
            )
        )

    
class Model(BaseModel):
    def __init__(
        self,
        model_cls : type[Module] | Module,
        criterion : TensorCallable,
        optimizer : GradientTransformations,
        *args,
        device : str | torch.device = "cpu",
        dtype : torch.dtype = torch.float32,
        model_init_fn : Optional[Callable[[Module], None]] = None,
        **kwargs,
    ):
        if isinstance(model_cls, Module):
            model = model_cls
        elif model_init_fn is not None:
            model = skip_init(model_cls, *args, **kwargs)
            model.apply(model_init_fn)
        else:
            model = model_cls(*args, **kwargs)

        super().__init__(
            model,
            criterion,
            optimizer,
            device=device,
            dtype=dtype,
        )


    def infer(
        self,
        iterator : Iterator[Any],
        concat_dim = 0,
        #non_linearity = Identity(),
        **kwargs,
    ):
        return torch.cat([
            self.__call__(
                self._params_dict,
                self._buffers_dict,
                input,
                **kwargs,
            ) for input, _, _ in iterator
        ], dim=concat_dim)

    def fit(
        self,
        train_iterator,
        valid_iterator,
        num_epochs = 1,
        **kwargs,
    ):
        for iepoch in range(num_epochs):
            self.train(True)
            train_losses = self.epoch_step(train_iterator, **kwargs)
            
            self.train(False)
            with torch.no_grad():
                valid_losses = self.epoch_step(valid_iterator, **kwargs)

            print("Train loss: ", train_losses.mean(), ", Valid_losses: ", valid_losses)


    def epoch_step(
        self,
        iterator : Iterator[Any],
        **kwargs,
    ):
        batch_losses = []
        for input, target, sample_weight in iterator:
            batch_loss, batch_grads = self.step(
                input.to(device=self.device),
                target.to(device=self.device),
                weight = sample_weight.to(device=self.device),
                **kwargs,
            )

            print(batch_loss, batch_grads)

            batch_losses.append(batch_loss)

        return torch.stack(batch_losses)


    def step(self, *args, **kwargs):
        if self.training:
            grads, loss = self.evaluate_and_gradient(
                self._params_dict, self._buffers_dict, *args, **kwargs
            )
            self.optimizer.apply_gradients(grads)
            return loss, grads
        else:
            loss = self.evaluate(self._params_dict, self._buffers_dict, *args, **kwargs)
            return loss, None

    def evaluate_and_gradient(self, *args, **kwargs):
        if self._compiled_eval_wgrad is not None:
            return self._compiled_eval_wgrad(*args, **kwargs)
        else:
            return self.forward_eval_wgrad(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if self._compiled_eval is not None:
            return self._compiled_eval(*args, **kwargs)
        else:
            return self.forward_eval(*args, **kwargs)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self._compiled_eval = torch.compile(
            self.forward_eval, *args, **kwargs
        )
        self._compiled_eval_wgrad = torch.compile(
            self.forward_eval_wgrad, *args, **kwargs
        )

    def forward_eval_wgrad(
        self, 
        *args, 
        argnums = 0, 
        has_aux = False, 
        **kwargs,
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

from churten.optimizer import FuncAdam
if __name__ == "__main__": 
    torch.manual_seed(0)
    
    tmodel = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1),
    )

    device = "cuda"

    optim = FuncAdam(lr = 1e-3, device=device)
    model = Model(
        tmodel,
        F.binary_cross_entropy_with_logits,
        optim,
        device=device
    )

    model.compile()

    params, buffers = model.params_and_buffers_dicts

    print("model:" , model)
    print("model state dict:" , model.model.state_dict())

    input = torch.ones(5, 10, device=device)
    target = torch.ones(5, 1, device=device)
    weight = torch.ones(5, 1, device=device)

    it = [(input, target, weight)]
    
    
    step_out = model.fit(
        it, it
    )

    print("from step:", step_out)



