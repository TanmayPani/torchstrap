import os

from typing import Any
from typing import Callable

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
        self.grads = deepcopy(self.params).zero_()

    def update_args(
        self
    )->tuple[list[Any], dict[str, Any]]:
        return [
            list(self.params.values()), 
            list(self.grads.values()), 
        ], {}
        

class GradientTransformations:
    def __init__(
        self, 
        state : OptimState,
    ):
        self._state = state
        self.num_stacked_states = self._state.numel()
        
    @property
    def state(self):
        return self._state

    def apply_gradients(
        self,
        grads : dict[str, Tensor],
    ):
        self._state.grads.update_(grads, keys_to_update=list(grads.keys()))
    
        for istate in range(self.num_stacked_states):
            args, kwargs = self._state[istate].update_args() 
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
):
    res = val if isinstance(val, Tensor) else torch.as_tensor(val, dtype=dtype)
    if res.shape != torch.Size(batch_size):
        if res.numel() == 1:
            return torch.full(
                batch_size, 
                res.squeeze_(), 
                dtype=dtype,
            )
        else:
            raise ValueError(f"{name} must be a singleton or a structure with shape {batch_size}, but given {res.shape}")
    return res

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




