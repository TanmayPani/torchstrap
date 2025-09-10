from functools import partial

from typing import Any
from typing import Callable
from typing import Optional
from typing import Protocol, runtime_checkable
from collections.abc import Sequence

import torch
from torch import LongTensor
from torch import Tensor, Size

from torchdata.nodes import SamplerWrapper
from torchdata.nodes import ParallelMapper
from torchdata.nodes import Prefetcher
from torchdata.nodes import PinMemory
from torchdata.nodes import Loader, Header

from . import TensorBatchSampler

@runtime_checkable
class DatasetLike(Protocol):
    def __len__(self) -> int : 
        ...
    def __getitem__(
        self, 
        idx : int | Sequence[int] | Tensor
    ) -> Any:
        ...

@runtime_checkable
class TensorLike(DatasetLike, Protocol):
    @classmethod
    def __torch_function__(
        cls, 
        func : Callable, 
        types : Sequence[type], 
        args : Sequence[Any], 
        kwargs : dict[str, Any] | None = None,
    ) -> Any : 
        ...

@runtime_checkable
class DatasetLike1(DatasetLike, Protocol):
    def __getitems__(self, idx : Sequence[int] | Tensor) -> TensorLike | list[TensorLike]:
        ...

class Collate:
    dataset : DatasetLike 
    def __init__(
        self,
        data : DatasetLike,
    ):
        self.dataset = data
        if isinstance(self.dataset, TensorLike):
            self.call_fn = self.getitem_tensor
        elif isinstance(self.dataset, DatasetLike1):
            self.call_fn = self.getitems
        else:
            self.call_fn = self.getitem

    def __call__(self, idx : Tensor):
        return self.call_fn(idx)
    
    def getitem_tensor(self, idx : Tensor):
        flattened_idx = torch.flatten(idx)
        return self.dataset[flattened_idx].reshape(idx.shape)

    def getitems(self, idx : Tensor):
        return self.dataset.__getitems__(
            torch.flatten(idx).tolist()
        ).reshape(idx.shape)
    
    def getitem(self, idx : Tensor):
        return torch.stack(
            [self.dataset[i] for i in idx.tolist()], 
            dim=0,
        ).reshape(idx.shape)

def get_stacked_batch_loader(
    dataset: DatasetLike,
    batch_sampler : TensorBatchSampler,
    num_workers: int = 0,
    collate_fn : Optional[Callable] = None,
    pin_memory : bool = False,
    load_only_first : Optional[int] = None,
):
    map_fn = collate_fn or Collate(dataset) 

    node = SamplerWrapper(batch_sampler)
    node = ParallelMapper(
        node, 
        map_fn=map_fn, 
        num_workers=num_workers, 
        method="process", 
        in_order=True,
    )

    if pin_memory:
        node = PinMemory(node)
    
    if num_workers > 0:
        node = Prefetcher(node, prefetch_factor=num_workers*2)

    if load_only_first is not None:
        node = Header(node, load_only_first)

    return Loader(node)


