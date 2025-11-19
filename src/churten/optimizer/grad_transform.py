from enum import Enum
from abc import ABC

#from dataclasses import dataclass, field, KW_ONLY
import threading

from copy import deepcopy
from functools import partial 

from beartype.typing import Self, Any, Optional, Union
from beartype.door import is_bearable

import torch
from torch import Tensor

from optree import PyTreeSpec, treespec_accessors, treespec_dict
from optree import tree_flatten, tree_unflatten, tree_flatten_one_level
from optree import tree_transpose_map_with_path 

from optree.dataclasses import dataclass, field, InitVar

from churten.utils.typing import Vector, Scalar 

def tensor_factory(default, **kwargs):
    return partial(torch.as_tensor, default, **kwargs)

def factory_kwargs(x : Tensor) -> dict[str, Any]:
    return dict(
        device = x.device,
        dtype = x.dtype,
    )

@dataclass(namespace="churten.state")
class OptimState:
    param_spec     : PyTreeSpec   = field(default_factory=treespec_dict, pytree_node=False)
    batch_dim      : Optional[int]= field(default=None)
    params         : list[Tensor] = field(default_factory=list            )
    state_steps    : list[Vector] = field(default_factory=list            )
    grads          : list[Tensor] = field(default_factory=list            )
    maximize       : bool         = field(default=False, pytree_node=False)
    foreach        : bool         = field(default=False, pytree_node=False)
    capturable     : bool         = field(default=False, pytree_node=False)
    differentiable : bool         = field(default=False, pytree_node=False)
    fused          : bool         = field(default=True , pytree_node=False)

    def __post_init__(self):
        _zeros = partial(torch.zeros, device = self.device, dtype = torch.float32)
        _zero_tensor = partial(torch.tensor, 0.0, device = self.device, dtype=torch.float32)
        
        if len(self.state_steps) == 0:
            self.state_steps = [
                _zeros(self.shape) if self.capturable or self.fused \
                else _zero_tensor().repeat(self.shape) for p in self.params
            ]

        if len(self.grads) == 0:
            self.gradients = None
        
        self.validate_fields()

    def validate_fields(self):
        _as_tensor = partial(torch.as_tensor, device = self.device, dtype=torch.float32)
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if field.type == Vector:
                value_tensor = _as_tensor(value).squeeze_()
                if value_tensor.shape == self.shape:
                    setattr(self, name, value_tensor)
                elif value_tensor.numel() == 1:
                    setattr(self, name, value_tensor.repeat(self.shape))
                else:
                    raise ValueError(
                        f"Got tensor {value} with incompatible shape "
                        f"{value_tensor.shape} for state  with shape {self.shape}"
                    )
                continue
            
            if field.type == list[Tensor] or field.type == list[Vector]:
                if len(value) > 0 and len(value) != self.num_params_per_state:
                    raise ValueError(
                        f"Attribute {name} is supposed to be defined on a "
                        f"per-parameter basis, but got {len(value)} {name} for "
                        f"{self.num_params_per_state} parameters."
                    )

            if field.type == list[Tensor]:
                for t in value:
                    if self.batch_dim is not None and t.shape[self.batch_dim] != self.num_states:
                        raise ValueError(
                            f"Tensor from list {name} should have {self.num_states} "
                            f"elements along non None batch_dim = {self.batch_dim}"
                        )
                continue
            
            if field.type == list[Vector]:
                for t in value:
                    if t.numel() != self.num_states:
                        raise ValueError(
                            f"Vector from list {name} should have {self.num_states} "
                            f"elements along non None batch_dim = {self.batch_dim}"
                        )
                continue
        
    @property
    def gradients(self):
        return tree_unflatten(self.param_spec, self.grads)

    @gradients.setter
    def gradients(
        self, 
        grads : Optional[dict[str, Tensor]], 
    ):
        if grads is None:
            self.grads[:] = [
                torch.zeros_like(
                    p, memory_format=torch.preserve_format
                ) for p in self.params
            ]
        else:
            self.grads[:] = [
                acc(grads) 
                for acc in treespec_accessors(self.param_spec)
            ]

    @property
    def shape(self):
        return () if self.batch_dim is None else (self.num_states,)

    @property
    def device(self):
        return self.params[0].device

    @property
    def num_params_per_state(self):
        return len(self.params)

    @property
    def num_states(self):
        if self.batch_dim is None:
            return 1
        return self.params[0].shape[self.batch_dim]
    
    @classmethod
    def from_param_dict(
        cls : type[Self], 
        param_dict : dict[str, Tensor],
        /,
        *,
        batch_dim : Optional[int] = None,
        **kwargs : Any,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def _from_pytree(
        cls : type[Self], 
        param_pytree : Any,
        /,
        **kwargs,
    ) -> Self:
        params, param_spec = tree_flatten(param_pytree)
        for key, val in kwargs.items():
            if key not in cls.__dataclass_fields__:
                continue 
            field = cls.__dataclass_fields__[key]
            if field.type == Vector:
                kwargs[key] = torch.as_tensor(
                    val, device=params[0].device, dtype=torch.float32
                )
            
        return cls(param_spec=param_spec, params=params, **kwargs)

    def unbind_leaf(self, path, x):
        if isinstance(x, Tensor):
            return torch.unbind(x, self.batch_dim)
        elif path[0] == "batch_dim":
            return (None,)*self.num_states
        else:
            return (x,)*self.num_states

    def unbind(self):
        if self.batch_dim is None:
            return (self,)
        
        return tree_transpose_map_with_path(
            self.unbind_leaf, self, none_is_leaf=True, namespace="churten.state", 
        )

class GradientTransformation(type):
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(
        cls, 
        name : str, 
        bases : tuple, 
        attrs : dict[str, Any], 
    ):
        if "state_type" not in attrs:
            raise AttributeError(
                "Every Optimizer (GradientTransformation) class needs "
                "to be linked with a state class inheriting from OptimState"
            )
        if "update" not in attrs:
            raise AttributeError(
                "Need to define an 'update' method to make an optimizer class!"
            )

        return super().__new__(cls, name, bases, attrs)

    def __call__(
        cls, 
        params : Optional[dict[str, Tensor]] = None, 
        **kwargs,
    ) -> tuple[Self, OptimState | list[OptimState] | None]:
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__()        
        return (
            cls._instances[cls], 
            cls.state_class.from_param_dict(params, **kwargs) if params else None,
        )

    @property
    def state_class(cls):
        return getattr(cls, "state_type")
    
    def apply_gradient(                                                                      
        cls,
        states : OptimState, 
        active_states : Optional[list[int]] = None,             
    ) -> tuple[OptimState, ...]:                                      
        apply_indices = active_states or list(range(states.num_states))
        _states = states.unbind()
        for istate in apply_indices:
            cls.update(_states[istate])
                                                                                             
        return _states

                                                                            
