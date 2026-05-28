from beartype.typing import Any, Optional

import torch
from torch import Tensor

from churten.state import State

class GradientTransformation(type):
    def __new__(
        cls, 
        name : str, 
        bases : tuple, 
        attrs : dict[str, Any], 
    ):
        if "state_class" not in attrs:
            raise TypeError(
                "Every Optimizer (GradientTransformation) class needs "
                "to be linked with a state class inheriting from OptimState"
            )
        if "update" not in attrs:
            raise TypeError(
                "Need to define an 'update' method to make an optimizer class!"
            )

        return super().__new__(cls, name, bases, attrs)

    def __call__(
        cls, 
        params_and_buffers_dict : Optional[tuple[dict[str, Tensor], dict[str, Tensor]]] = None,
        **kwargs : Any,
    ):      
        return (
            cls, 
            State.from_stacked_params_and_buffers(
                params_and_buffers_dict, 
                getattr(cls, "state_class"), 
                **kwargs,
            ),
        ) if params_and_buffers_dict else cls

    def apply_gradient(
        cls,
        states : State,
    ) -> State:
        active_mask = torch.as_tensor(
            states.model_status_list,
            device=states.device,
            dtype=torch.bool,
        )
        getattr(cls, "update")(states.optimizer_state, active_mask=active_mask)
        return states
