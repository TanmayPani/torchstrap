import os
from functools import partial
from typing import Optional, Any, Self, Callable
from collections import OrderedDict
from collections.abc import Iterator
import dataclasses

import torch
from torch import Tensor, Size

from optree.dataclasses import dataclass, field, InitVar

from optree import PyTree, PyTreeTypeVar, PyTreeSpec
from optree import dict_insertion_ordered
from optree import tree_transpose_map, tree_transpose_map_with_path 
from optree import tree_flatten, tree_unflatten
from optree import treespec_entries

from churten.utils.typing import Vector 
from churten.callbacks import Checkpoint

@dataclasses.dataclass
class State:
    param_dict : dict[str, Tensor]        
    buffer_dict : dict[str, Tensor] 
    optimizer_state : "OptimState"
    batch_size : Size = dataclasses.field(default_factory=Size)
    model_index_list : list[int] = dataclasses.field(default_factory=list)
    model_status_list : list[bool] = dataclasses.field(default_factory=list)
    extra_state : dict[str, Any] = dataclasses.field(default_factory=dict)
    _state_list : list["State"] = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self):
        if len(self.model_index_list) == 0:
            self.model_index_list = list(range(self.num_replicas))
            self.model_status_list = [True]*self.num_replicas

        if self.num_replicas == 1: 
            self._state_list = [self]
        else:
            _unbind_fn = partial(torch.unbind, dim=0)
            _param_dict_list = tree_transpose_map(
                _unbind_fn, self.param_dict,
            )
            if len(self.buffer_dict) > 0:
                _buffer_dict_list = tree_transpose_map(
                    _unbind_fn, self.buffer_dict,
                )
            else:
                _buffer_dict_list = [{}]*self.num_replicas
            _optim_state_list = self.optimizer_state.unbind()
            self._state_list = [
                State(
                    _params, 
                    _buffers, 
                    _optim_state, 
                    model_index_list=[imodel], 
                    model_status_list=[self.model_status_list[imodel]], 
                    extra_state=self.extra_state,
                ) for (
                    imodel, 
                    _params, 
                    _buffers, 
                    _optim_state,
                ) in zip(
                    self.model_index_list, 
                    _param_dict_list, 
                    _buffer_dict_list, 
                    _optim_state_list,
                )
            ]
            
    @classmethod
    def from_stacked_params_and_buffers(
        cls : type[Self],
        params_and_buffers_dict : tuple[dict[str, Tensor], dict[str, Tensor]],
        optimizer_state_cls : type["OptimState"],
        /,
        batch_size : tuple[int], 
        **kwargs : Any, 
    ) -> Self :
        optimizer_state = optimizer_state_cls.from_param_dict(
            params_and_buffers_dict[0], 
            batch_size=batch_size, 
            **kwargs,
        )

        instance = cls(
            param_dict = params_and_buffers_dict[0],
            buffer_dict = params_and_buffers_dict[1],
            optimizer_state = optimizer_state,
            batch_size = torch.Size(batch_size),
        )

        return instance

    @property
    def num_replicas(self) -> int:
        if len(self.optimizer_state.batch_size) == 0:
            return 1
        return self.optimizer_state.batch_size[0]

    @property
    def device(self) -> torch.device:
        return next(iter(self.param_dict.values())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.param_dict.values())).dtype

    @property
    def active_models(self) -> list[int]:
        return [
            imodel 
            for imodel, status in zip(self.model_index_list, self.model_status_list) \
            if status
        ]

    def mask_model(self, index : int):
        self.model_status_list[index] = False

    @property
    def num_active_models(self) -> int:
        return len(self.active_models)

    def set_gradients(self, grads):
        self.optimizer_state.gradients = grads

    def __getitem__(self, index) -> Self:
        return self._state_list[index]

    def __len__(self) -> int:
        return len(self._state_list)

    def vmap_in_dims(self, args : Optional[tuple] = None):
        if "in_dims" not in self.extra_state:
            in_dim_params_and_buffers = (0, 0)
            if args is not None:
                in_dim_args = in_dim_params_and_buffers + tuple(
                    0 if isinstance(arg, Tensor) else None for arg in args
                )
                self.extra_state["in_dims"] = in_dim_args
            else:
                self.extra_state["in_dims"] = in_dim_params_and_buffers

        return self.extra_state["in_dims"]

    def to_file(
        self,
        root_dir : Optional[str] = None, 
        file_name : Optional[str] = None,
    ):
        root_dir = root_dir or "checkpoint"
        file_name = file_name or "state_dict.pt" 
        if self.batch_size == ():
            if not os.path.exists(root_dir):
                os.makedirs(root_dir, exist_ok=True)
            file_path = os.path.join(root_dir, file_name)
            torch.save(
                self.state_dict(), 
                file_path,
            )

        else:
            for i in self.active_models:
                dir_path = os.path.join(root_dir, f"replica_{i}")
                self[i].to_file(dir_path, file_name)

    def from_file(
        self,
        root_dir : Optional[str] = None, 
        file_name : Optional[str] = None,
        **kwargs,
    ):
        root_dir = root_dir or "checkpoint"
        file_name = file_name or "state_dict.pt" 
        if self.batch_size == ():
            _state_dict = torch.load(
                os.path.join(root_dir, file_name), 
                map_location=kwargs.pop("map_location", "cpu"),
                mmap=kwargs.pop("mmap", True),
                weights_only=kwargs.pop("weights_only", True),
                **kwargs,
            )
            self.load_state_dict(_state_dict)
        else:
            for i in self.active_models:
                dir_path = os.path.join(root_dir, f"replica_{i}")
                self[i].from_file(dir_path, file_name, **kwargs)
       
    def state_dict(self):
        _state_dict = {}
        if self.batch_size == ():
            _state_dict["parameters"] = {}
            _state_dict["buffers"] = {}
            for name, param in self.param_dict.items():
                _state_dict["parameters"][name] = param.detach().cpu()
            for name, buffer in self.buffer_dict.items():
                _state_dict["buffers"][name] = buffer.detach().cpu()
            _state_dict["optimizer"] = self.optimizer_state.state_dict()
        else:
            for i in self.active_models:
                _state_dict[f"replica_{i}"] = self[i].state_dict()
        
        _state_dict["metadata"] = {}
        _state_dict["metadata"]["extra_state"] = self.extra_state
        _state_dict["metadata"]["batch_size"] = self.batch_size
        _state_dict["metadata"]["model_index_list"] = self.model_index_list
        _state_dict["metadata"]["model_status_list"] = self.model_status_list
        return _state_dict

    @torch.no_grad()
    def load_state_dict(self, state : dict[str, Any],):
        if self.batch_size == ():
            for name, param in self.param_dict.items():
                param.copy_(state["parameters"][name])
                del state["parameters"][name]

            for name, buffer in self.buffer_dict.items():
                buffer.copy_(state["buffers"][name])
                del state["buffers"][name]

            self.optimizer_state.load_state_dict(state["optimizer"])
        else:
            for i in self.active_models:
                self[i].load_state_dict(state[f"model_{i}"])

        self.extra_state = state["metadata"]["extra_state"]
        self.batch_size = state["metadata"]["batch_size"]
        self.model_index_list = state["metadata"]["model_index_list"]
        self.model_status_list = state["metadata"]["model_status_list"]

@dataclass(namespace="churten.state")
class OptimState:
    param_names    : list[str]    = field(default_factory=list, pytree_node=False)
    batch_size     : Size         = field(default_factory=Size)
    params         : list[Tensor] = field(default_factory=list)
    state_steps    : list[Vector] = field(default_factory=list)
    grads          : list[Tensor] = field(default_factory=list)
    maximize       : bool         = field(default=False, pytree_node=False)
    foreach        : bool         = field(default=False, pytree_node=False)
    capturable     : bool         = field(default=False, pytree_node=False)
    differentiable : bool         = field(default=False, pytree_node=False)
    fused          : bool         = field(default=True , pytree_node=False)

    def state_dict(self):
        _state_dict = {}
        for name, field in self.__dataclass_fields__.items():
            _state_dict[name] = getattr(self, name)

        return _state_dict


    @torch.no_grad()
    def load_state_dict(self, state):
        for name, field in self.__dataclass_fields__.items():
            self_val = getattr(self, name)
            if field.type in [list[Tensor], list[Vector]]:
                for t_val, self_t_val in zip(state[name], self_val):
                    self_t_val.copy_(t_val)
            elif field.type == Vector:
                self_val.copy_(state[name])
            else:
                setattr(self, name, state[name])

            del state[name]

    def __post_init__(self):
        _zeros = partial(
            torch.zeros, 
            device = self.params[0].device, 
            dtype = torch.float32,
        )
        _zero_tensor = partial(
            torch.tensor, 
            0.0, 
            device = self.params[0].device, 
            dtype=torch.float32,
        )
        
        if len(self.state_steps) == 0:
            self.state_steps = [
                _zeros(self.batch_size) if self.capturable or self.fused \
                else _zero_tensor().repeat(self.batch_size) for p in self.params
            ]

        if len(self.grads) == 0:
            self.gradients = None
        
        self.validate_fields()

    def validate_fields(self):
        _as_tensor = partial(
            torch.as_tensor, 
            device = self.params[0].device, 
            dtype=torch.float32,
        )

        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if field.type == Vector:
                value_tensor = _as_tensor(value).squeeze_()
                if value_tensor.shape == self.batch_size:
                    setattr(self, name, value_tensor)
                elif value_tensor.numel() == 1:
                    setattr(self, name, value_tensor.repeat(self.batch_size))
                else:
                    raise ValueError(
                        f"Got tensor {value} with incompatible shape "
                        f"{value_tensor.shape} for state  with shape {self.batch_size}"
                    )
                continue
            
            if field.type == list[Tensor] or field.type == list[Vector]:
                if len(value) > 0 and len(value) != len(self.params):
                    raise ValueError(
                        f"Attribute {name} is supposed to be defined on a "
                        f"per-parameter basis, but got {len(value)} {name} for "
                        f"{len(self.params)} parameters."
                    )

            if field.type == list[Tensor]:
                for t in value:
                    if t.shape[:len(self.batch_size)] != self.batch_size:
                        raise ValueError(
                            f"Tensor from list {name} should have shape {self.batch_size} "
                            f"along the first {len(self.batch_size)} dimensions, "
                            f"but got {t.shape[:len(self.batch_size)]}"
                        )
                continue
            
            if field.type == list[Vector]:
                for t in value:
                    if t.shape != self.batch_size:
                        raise ValueError(
                            f"Vector from list {name} should have shape {self.batch_size} "
                            f"but got tensor with shape {t.shape}"
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
            if len(self.grads) == 0:
                self.grads = [
                    torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ) for p in self.params
                ]
            else:
                glist = [g.zero_() for g in self.grads]
        else:
            glist = [
                g.copy_(grads[param_name]) 
                for g, param_name in zip(self.grads, self.param_names)
            ]

    @classmethod
    def from_param_dict(
        cls : type[Self], 
        param_dict : dict[str, Tensor],
        /,
        batch_size : tuple | list,
        **kwargs : Any,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def _from_pytree(
        cls : type[Self], 
        param_pytree : Any,
        /,
        batch_size : tuple | list,
        **kwargs,
    ) -> Self:
        with dict_insertion_ordered(True, namespace="churten.state"):
            params, param_spec = tree_flatten(
                param_pytree, namespace="churten.state",
            )
        param_names = treespec_entries(param_spec)
        for key, val in kwargs.items():
            if key not in cls.__dataclass_fields__:
                continue 
            field = cls.__dataclass_fields__[key]
            if field.type == Vector:
                kwargs[key] = torch.as_tensor(
                    val, device=params[0].device, dtype=torch.float32
                )
            
        return cls(param_names=param_names, batch_size=torch.Size(batch_size), params=params, **kwargs)

    def unbind_leaf(self, path, x):
        if isinstance(x, Tensor):
            return torch.unbind(x, 0)
        elif path[0] == "batch_size":
            return (torch.Size(),)*self.batch_size[0]
        else:
            return (x,)*self.batch_size[0]

    def unbind(self):
        if len(self.batch_size) == 0:
            return [self]
        
        return tree_transpose_map_with_path(
            self.unbind_leaf, 
            self, 
            none_is_leaf=True, 
            namespace="churten.state", 
        )

