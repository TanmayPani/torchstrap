from __future__ import annotations
from typing import Optional
import tqdm

import pyarrow as pa

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset

from tensordict import MemoryMappedTensor
from tensordict import TensorDict

class TensorDictDataset(Dataset[TensorDict | tuple[Tensor, ...]]):
    _tensordict : TensorDict
    def __init__(self, td : TensorDict, **tensors : Tensor) -> None:
        self._tensordict = td
        self._tdict_extra = TensorDict(**tensors, batch_size=self._tensordict.batch_size)

    def __getitem__(self, index):
        return self._tensordict[index]

    def __len__(self):
        return self._tensordict.size(0)


def arrow_to_tensordict(
    table : pa.Table,
    target : Optional[Tensor] = None,
    weight : Optional[Tensor] = None,
    *,
    labels : dict[str, Tensor] = {},
    dtype : torch.dtype = torch.float32,
    columns : Optional[list[str]] = None,
    prefix : Optional[str] = None,
    max_write_chunk : Optional[int] = None,
):
    columns = columns if columns is not None else table.column_names
    input_table = table.select(columns)

    data = TensorDict(
        dict(
            input = torch.zeros((len(columns),), dtype=dtype),
            target = torch.zeros((), dtype=dtype),
            weight = torch.ones((), dtype=dtype),
            **{key : torch.zeros((), dtype=label.dtype) for key, label in labels.items()},
        ),
        batch_size=[],
    ).expand(len(table)).memmap_like(prefix=prefix)

    rbatches = input_table.to_batches(max_chunksize=max_write_chunk)

    target = target if target is not None else torch.zeros(len(input_table))
    weight = weight if weight is not None else torch.ones(len(input_table))
   
    i = 0
    pbar = tqdm.tqdm(total=len(input_table))
    for rbatch in rbatches:
        batch_size = len(rbatch)
        pbar.update(batch_size)
        batch  = TensorDict(
            dict(
                input = torch.as_tensor(list(zip(*rbatch.to_pydict().values())), dtype=dtype),
                target = target[i : i + batch_size],
                weight = weight[i : i + batch_size],
                **{k : v[i:i+batch_size] for k, v in labels.items()},
            ),
            batch_size=(batch_size,),
            device="cpu",
        )

        data[i : i + batch_size] = batch
        i += batch_size
    return data

