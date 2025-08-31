from typing import Self
from typing import Callable
from collections.abc import Mapping

from torch import Tensor

from tensordict import TensorDict
from tensordict import NonTensorDataBase
from tensordict import TensorClass

#type Params = Mapping[str, Tensor]

class OptimState(TensorClass["nocast"]):
    def check_state(self) -> None:
        raise NotImplementedError("Use a subclass defined for a paritcular optimizer!")

    def reset(self, params : TensorDict) -> Self:
        raise NotImplementedError("Use a subclass defined for a paritcular optimizer!")

class GradientTransformations:
    def __init__(
        self, 
        update_fn : Callable,
        state : OptimState,
    ):
        self.update_fn = update_fn
        self.state = state
        #self.state.check_state()

    def init(self, params : TensorDict) -> OptimState:
        return self.state.reset(params)

    def _update_all(
        self,
        state : OptimState,
    ) -> OptimState:
        for istate in range(state.batch_size[0]):
            state[istate] = self._update(state[istate])
        return state

    def _update(
        self, 
        state : OptimState, 
    ) -> OptimState:
        _state = {}
        for key, val in state.items():
            if isinstance(val, Mapping):
                _state[key] = list(val.values())
            elif isinstance(val, NonTensorDataBase):
                _state[key] = val.data
            else:
                _state[key] = val

        self.update_fn(**_state)

        return state

    def update(
        self,
        state : OptimState,
    ):
        if len(state.batch_size) > 0 and state.batch_size[0] > 1:
            return self._update_all(state)
        return self._update(state)





