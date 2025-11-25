from functools import partial
from beartype.typing import Optional

from optree.dataclasses import dataclass, field

from optree import tree_map
from optree import PyTree, PyTreeSpec
from optree import tree_map, tree_transpose_map, tree_iter 
from optree import tree_flatten, tree_unflatten, tree_flatten_one_level
from optree import tree_structure, treespec_one_level, treespec_list, treespec_leaf

import torch
from torch import Tensor
from torch.optim.adam import adam

from churten.utils.typing import Vector, FloatScalarLike
from churten.state import OptimState
from churten.optimizer.grad_transform import GradientTransformation

@dataclass(namespace="churten.state")
class AdamState(OptimState):
    exp_avgs               : list[Tensor] = field(default_factory=list)
    exp_avg_sqs            : list[Tensor] = field(default_factory=list)
    max_exp_avg_sqs        : list[Tensor] = field(default_factory=list)
    lr                     : Vector       = field(default_factory=partial(torch.as_tensor, 1e-3 ))
    beta1                  : Vector       = field(default_factory=partial(torch.as_tensor, 0.9  ))
    beta2                  : Vector       = field(default_factory=partial(torch.as_tensor, 0.999))
    eps                    : Vector       = field(default_factory=partial(torch.as_tensor, 1e-8 ))
    weight_decay           : Vector       = field(default_factory=partial(torch.as_tensor, 1e-2 ))
    amsgrad                : bool         = field(default=False, pytree_node=False)
    decoupled_weight_decay : bool         = field(default=True , pytree_node=False)

    def __post_init__(self):
        _zeros_like = partial(torch.zeros_like, memory_format=torch.preserve_format)
        if len(self.exp_avgs) == 0:
            self.exp_avgs = [_zeros_like(p) for p in self.params]

        if len(self.exp_avg_sqs) == 0:
            self.exp_avg_sqs = [_zeros_like(p) for p in self.params]

        if self.amsgrad and len(self.max_exp_avg_sqs) == 0:
            self.max_exp_avg_sqs = [_zeros_like(p) for p in self.params]
        super().__post_init__()

    @classmethod
    def from_param_dict(
        cls, 
        param_pytree : dict[str, Tensor],
        /,
        batch_size : tuple[int],
        *,
        lr           : FloatScalarLike = 1e-3,
        beta1        : FloatScalarLike = 0.9 ,
        beta2        : FloatScalarLike = 0.999,    
        eps          : FloatScalarLike = 1e-8,    
        weight_decay : FloatScalarLike = 1e-2,   
        **kwargs : bool,
    ):
        return cls._from_pytree(
            param_pytree,
            batch_size=batch_size,
            lr = lr, 
            beta1 = beta1,  
            beta2 = beta2,
            eps = eps,
            weight_decay = weight_decay,
            **kwargs,
        )

class Adam(metaclass=GradientTransformation):
    state_class : type[OptimState] = AdamState
    @classmethod
    def update(cls, state : AdamState) -> AdamState:
        if state.batch_size != ():
            raise ValueError(
                "Multiple replicas in a single optimizer step not supported yet."
            )

        adam(
            state.params,
            state.grads,
            state.exp_avgs,
            state.exp_avg_sqs,
            state.max_exp_avg_sqs,
            state.state_steps,
            amsgrad = state.amsgrad,
            has_complex = False,
            beta1 = state.beta1,
            beta2 = state.beta2,
            lr = state.lr,
            weight_decay = state.weight_decay,
            eps = state.eps,
            maximize = state.maximize,
            foreach = state.foreach,
            capturable = state.capturable,
            differentiable = state.differentiable,
            fused = state.fused,
            grad_scale = None,
            found_inf = None,
            decoupled_weight_decay = state.decoupled_weight_decay,
        )

        return state


