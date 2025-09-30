import os

from typing import Any
from typing import Optional

from copy import deepcopy
from dataclasses import field

import torch
from torch import Tensor

from tensordict import NonTensorData
from tensordict import TensorDictBase
from tensordict import TensorDict
from tensordict import TensorClass

class OptimState(TensorClass["nocast"]):
    params : TensorDictBase = field(default_factory = TensorDict)
    grads : TensorDictBase = field(default_factory = TensorDict)
    lr: Tensor = field(default=torch.as_tensor(1e-3, dtype=torch.float32))
    _update_args : Optional[tuple[list[Any], dict[str, Any]]] = None
    def check_state(self) -> None:
        raise NotImplementedError("Use a subclass defined for a paritcular optimizer!")
    
    def init(
        self, 
        _params : dict[str, Tensor] | TensorDict
    ):
        self.params = TensorDict(
            _params, 
            batch_size=self.batch_size,
        )
        #self.device = self.params.device
        self.grads = deepcopy(self.params).zero_()
        #self.lr = self.lr.to(dtype=torch.float32, device=self.device)

    @property
    def update_args(self) -> tuple[list[Any], dict[str, Any]]:
        if self._update_args is None:
            return self.set_update_args()
        else:
            return self._update_args

    def set_update_args(self) -> tuple[list[Any], dict[str, Any]]:
        self._update_args = ([
            list(self.params.values()), 
            list(self.grads.values()), 
        ], {"lr" : self.lr},)

        return self._update_args

class GradientTransformations:
    def __init__(
        self, 
        state : OptimState,
        device : str | torch.device = "cpu",
        #dtype : torch.dtype = torch.float32,
    ):
        self._state = state.to(
            device = device, #dtype = dtype, 
        )
        self.num_stacked_states = self._state.numel()
   
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val 

    def init(self, params : dict[str, Tensor]):
        params_td = TensorDict(params, batch_size=self._state.batch_size)
        self.state = self.state.to(device=params_td.device, dtype=params_td.dtype)
        self.state.init(params_td)

    def apply_gradients(
        self,
        grads : dict[str, Tensor],
    ):
        self._state.grads.update_(grads, keys_to_update=list(grads.keys()))
        
        if self._state.batch_dims == 0:
            args, kwargs = self._state.update_args
            self.step(*args, **kwargs)
        else:
            for istate in range(self.num_stacked_states):
                args, kwargs = self._state[istate].update_args
                self.step(*args, **kwargs)
    
    @classmethod
    def step(
        cls, 
        *args : Any,
        **kwargs : Any,
    ):
        raise NotImplementedError("Use a subclass defined for a paritcular optimizer!")

def make_tensor_attr(
    name : str,
    val : Any,
    batch_size = [],
    dtype = torch.float32,
    device : str | torch.device = "cpu"
):
    res = val if isinstance(val, Tensor) \
          else torch.as_tensor(val, dtype=dtype, device=device)
    res.squeeze_()
    if res.shape == batch_size:
        return res
    elif res.shape == ():
        return torch.full(
            batch_size, 
            res.item(), 
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(
            f"{name} must be a singleton or a structure with "
            f"shape {batch_size}, but given {res.shape}"
        )

def make_non_tensor_attr(
    name : str,
    val : Any,
    batch_size = [],
):
    if isinstance(val, list):
        if len(batch_size) == 0:
            return NonTensorData(val, batch_size=batch_size)
        if len(val) == batch_size[0]:
            return torch.stack([NonTensorData(v, batch_size=[]) for v in val])
        raise ValueError(f"{name} must be a singleton or a structure with shape {batch_size}, but given {len(val)}") 
    else:
        return NonTensorData(val, batch_size=batch_size)




