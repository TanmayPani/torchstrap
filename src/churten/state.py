import os
from itertools import chain
from functools import partial
from copy import deepcopy
from typing import Callable, Optional, Any, Self
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

from .optimizer import GradientTransformation, OptimState
from .callbacks import Callback, PrintLog, PassThroughScoring, EpochTimer
from .history import History

@dataclass
class State:
    param_dict : dict[str, Tensor]      
    buffer_dict : dict[str, Tensor] = field(default_factory=dict)
    optimizer_state : OptimState = field(default_factory=list, init=False)
    active_models : list[int] = field(default_factory=list, init=False)
    extra : dict[str, Any] = field(default_factory=dict, init=False)
    is_initialized : bool = field(default=False, init=False) 

    def initialize_state(
        self, 
        optimizer : GradientTransformation,
        *,
        batch_dim : Optional[int] = 0, 
        **kwargs : Any, 
    ) -> None:
        _zeros_like = partial(torch.zeros_like, memory_format=torch.preserve_format)
        self.optimizer_state = optimizer.state_class.from_param_dict(
            self.param_dict, batch_dim=batch_dim, **kwargs
        )

        self.active_models = list(range(self.optimizer_state.num_states))
        self.is_initialized = True

    @property
    def num_replicas(self) -> int:
        return self.optimizer_state.num_states

    @property
    def device(self) -> torch.device:
        return next(iter(self.param_dict.values())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.param_dict.values())).dtype

    @property
    def initialized(self) -> bool:
        return self.is_initialized

    def vmap_in_dims(self, args : Optional[tuple] = None):
        if "in_dims" not in self.extra:
            in_dim_params_and_buffers = (0, 0)
            if args is not None:
                in_dim_args = in_dim_params_and_buffers + tuple(
                    0 if isinstance(arg, Tensor) else None for arg in args
                )
                self.extra["in_dims"] = in_dim_args
            else:
                self.extra["in_dims"] = in_dim_params_and_buffers

        return self.extra["in_dims"]


