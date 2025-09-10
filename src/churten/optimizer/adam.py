from copy import deepcopy
from collections.abc import Sequence
from typing import Any, Optional
from dataclasses import field

import torch
from torch import Tensor
from torch.optim.adam import adam

from tensordict import TensorDictBase
from tensordict import TensorDict

from .grad_transform import GradientTransformations
from .grad_transform import OptimState
from .grad_transform import make_tensor_attr, make_non_tensor_attr

class AdamOptState(OptimState):
    exp_avgs : TensorDictBase = field(default_factory = TensorDict)
    exp_avg_sqs : TensorDictBase = field(default_factory = TensorDict)
    max_exp_avg_sqs : TensorDictBase = field(default_factory = TensorDict)
    state_steps : TensorDictBase = field(default_factory = TensorDict)
    lr: Tensor = field(default=torch.as_tensor(1e-3, dtype=torch.float32))
    beta1: Tensor = field(default=torch.as_tensor(0.9, dtype=torch.float32))
    beta2: Tensor = field(default=torch.as_tensor(0.999, dtype=torch.float32))
    eps: Tensor = field(default=torch.as_tensor(1e-8, dtype=torch.float32))
    weight_decay: Tensor = field(default=torch.as_tensor(0, dtype=torch.float32))
    amsgrad               : bool = False
    maximize              : bool = False
    decoupled_weight_decay: bool = False
    foreach               : bool = False
    capturable            : bool = False
    differentiable        : bool = False
    fused                 : bool = False
    grad_scale: Optional[Tensor] = None
    found_inf: Optional[Tensor] = None
    has_complex: bool = False

    def check_state(self):
        if not self.capturable and self.foreach:
            raise ValueError(
                "FuncAdam doesn't support capturable=False and foreach=True"
            )
        
        if not torch.all(torch.ge(self.lr, 0.0)):
            raise ValueError(f"Invalid learning rate: {self.lr}")
        if not torch.all(torch.ge(self.eps, 0.0)):
            raise ValueError(f"Invalid epsilon value: {self.eps}")
        if not torch.all(torch.logical_and(torch.ge(self.beta1, 0.0), torch.lt(self.beta1, 1.0))):
            raise ValueError(f"Invalid beta parameter at index 0: {self.beta1}")
        if not torch.all(torch.logical_and(torch.ge(self.beta2, 0.0), torch.lt(self.beta2, 1.0))):
            raise ValueError(f"Invalid beta parameter at index 1: {self.beta2}")
        if not torch.all(torch.ge(self.weight_decay, 0.0)):
            raise ValueError(f"Invalid weight_decay value: {self.weight_decay}")
               
    def init(self, _params: dict[str, Tensor] | TensorDict):
        super().init(_params)
        self.exp_avgs = deepcopy(self.params).zero_()
        self.exp_avg_sqs = deepcopy(self.params).zero_()
        self.max_exp_avg_sqs = deepcopy(self.params).zero_()
        for param_name in self.params.keys():
            self.state_steps[param_name] = torch.zeros(
                self.batch_size, dtype=torch.float32, device=self.device,
            )

    def update_args(self) -> tuple[list[Any], dict[str, Any]]:
        args, kwargs = super().update_args()

        args.extend([
            list(self.exp_avgs.values()       ), 
            list(self.exp_avg_sqs.values()    ), 
            list(self.max_exp_avg_sqs.values()), 
            list(self.state_steps.values()    ),
        ])

        args.extend([
            getattr(self, key) for key in  (
                "foreach",
                "capturable",
                "differentiable",
                "fused",
                "grad_scale",
                "found_inf",
                "has_complex",
                "decoupled_weight_decay",
            )
        ])

        kwargs.update({
            key : self.__dict__["_tensordict"][key] for key in (
                "lr", 
                "beta1", 
                "beta2",
                "weight_decay",
                "eps",
                "maximize",
                "amsgrad",
            )
        })

        return args, kwargs


class FuncAdam(GradientTransformations):
    def __init__(
        self, 
        lr = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: bool = False,
        maximize: bool = False,
        capturable: bool = True,
        differentiable: bool = False,
        fused: bool = True,
        decoupled_weight_decay: bool = False,
        batch_size : Sequence[int] = [],
        **kwargs,
    ):
        
        optim_state = {
            "lr"                     : make_tensor_attr("lr"                    , lr                    , batch_size=batch_size ),
            "beta1"                  : make_tensor_attr("betas[0]"              , betas[0]              , batch_size=batch_size ), 
            "beta2"                  : make_tensor_attr("betas[1]"              , betas[1]              , batch_size=batch_size ),
            "eps"                    : make_tensor_attr("eps"                   , eps                   , batch_size=batch_size ),
            "weight_decay"           : make_tensor_attr("weight_decay"          , weight_decay          , batch_size=batch_size ),

            "amsgrad"                : make_non_tensor_attr("amsgrad"               , amsgrad               , batch_size=batch_size ),
            "maximize"               : make_non_tensor_attr("maximize"              , maximize              , batch_size=batch_size ), 
            "foreach"                : make_non_tensor_attr("foreach"               , foreach               , batch_size=batch_size ),
            "capturable"             : make_non_tensor_attr("capturable"            , capturable            , batch_size=batch_size ),
            "differentiable"         : make_non_tensor_attr("differentiable"        , differentiable        , batch_size=batch_size ),
            "fused"                  : make_non_tensor_attr("fused"                 , fused                 , batch_size=batch_size ),
            "decoupled_weight_decay" : make_non_tensor_attr("decoupled_weight_decay", decoupled_weight_decay, batch_size=batch_size ),
            "batch_size" : batch_size
        }
        super().__init__(AdamOptState(**optim_state), **kwargs)


    @classmethod
    def step(cls, *args, **kwargs):
        adam(*args, **kwargs)

