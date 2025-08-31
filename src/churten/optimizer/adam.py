from copy import deepcopy
from collections.abc import Sequence
from collections.abc import Mapping

from typing import Self
from typing import Union
from typing import Optional

from dataclasses import field

import torch
from torch import Tensor
from torch.optim.adam import adam

from tensordict import TensorDict

from .grad_transform import GradientTransformations
from .grad_transform import OptimState

type Params = Mapping[str, Tensor]

class AdamOptState(OptimState):
    params : TensorDict = field(default_factory = TensorDict)
    grads : TensorDict = field(default_factory = TensorDict)
    exp_avgs : TensorDict = field(default_factory = TensorDict)
    exp_avg_sqs : TensorDict = field(default_factory = TensorDict)
    max_exp_avg_sqs : TensorDict = field(default_factory = TensorDict)
    state_steps : TensorDict = field(default_factory = TensorDict)
    lr: Union[float, Tensor] = 1e-3
    beta1: Union[float, Tensor] = 0.9
    beta2: Union[float, Tensor] = 0.999
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False
    foreach: Optional[bool] = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: Optional[bool] = None
    decoupled_weight_decay: bool = False

    def check_state(self):
        #print("post_init called!", self.__class__)
        if isinstance(self.lr, Tensor):
            if self.foreach and not self.capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if len(self.batch_size) > 0 and sum(self.batch_size) > 1:
                if self.lr.shape != self.batch_size:
                    raise ValueError("Tensor lr must have the same shape as the batch")

            else:
                if self.lr.numel() != 1:
                    raise ValueError("Tensor lr must have 1 element!")
        
        if not 0.0 <= self.lr:
            raise ValueError(f"Invalid learning rate: {self.lr}")
        if not 0.0 <= self.eps:
            raise ValueError(f"Invalid epsilon value: {self.eps}")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {self.beta1}")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {self.beta2}")
        if not 0.0 <= self.weight_decay:
            raise ValueError(f"Invalid weight_decay value: {self.weight_decay}")
        if not (
            (isinstance(self.beta1, float) and isinstance(self.beta2, float))
            or (isinstance(self.beta1, Tensor) and isinstance(self.beta2, Tensor))
        ):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(self.beta1, Tensor):
            if not self.capturable and self.foreach:
                raise ValueError(
                    "beta1 as a Tensor is not supported for capturable=False and foreach=True"
                )
            if self.beta1.shape != self.batch_size:
                raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(self.beta2, Tensor):
            if not self.capturable and self.foreach:
                raise ValueError(
                    "beta2 as a Tensor is not supported for capturable=False and foreach=True"
                )
            if self.beta2.shape != self.batch_size:
                raise ValueError("Tensor betas[1] must be 1-element")
       
    def reset(self, params: TensorDict)->Self:
        self.params = params
        self.grads = deepcopy(params).zero_()
        self.exp_avgs = deepcopy(params).zero_()
        self.exp_avg_sqs = deepcopy(params).zero_()
        self.max_exp_avg_sqs = deepcopy(params).zero_()
        for param_name in params.keys():
            self.state_steps[param_name] = torch.zeros(
                self.batch_size, dtype=torch.float32, device=params.device
            )
        return self

class FuncAdam(GradientTransformations):
    def __init__(
        self, 
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = True,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        decoupled_weight_decay: bool = False,
        batch_size : Optional[Sequence[int]] = None, 
    ):

        optim_state = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": fused,
            "decoupled_weight_decay": decoupled_weight_decay,
            "batch_size" : batch_size
        }
        super().__init__(adam, AdamOptState(**optim_state))

