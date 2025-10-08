"""Contains history class and helper functions."""

from typing import Optional, Sequence
from tensordict import TensorDict, LazyStackedTensorDict

# pylint: disable=invalid-name
class _none:
    """Special placeholder since ``None`` is a valid value."""

class History:
    def __init__(
        self,
        num_replicas : int = 1,
        path : str = "history"
    ):
        self.batch_size = [num_replicas,]
        self._history = LazyStackedTensorDict(stack_dim=0, batch_size=self.batch_size)
        self._last_epoch = 0
        self._path = path

    @property
    def history(self):
        return self._history

    def append(
        self, 
        #batch_size : Optional[Sequence[int]] = None, 
        **kwargs
    ):
        self._history.append(TensorDict(kwargs, batch_size=self.batch_size))
        self._last_epoch += 1

    def flush(self):
        self._history.memmap_(self._path)
