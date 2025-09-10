from __future__ import annotations
from dataclasses import field
from typing import Optional, Self, Sequence
from numpy.ma import indices
import tqdm
import numpy as np
import pyarrow as pa

import torch
from torch import as_tensor
from torch import Tensor
from torch.nn import Module

from tensordict import MemoryMappedTensor
from tensordict import TensorClass
from tensordict import TensorDict

class TabularDataset(TensorClass):
    inputs : Tensor
    targets : Tensor
    sample_weights : Tensor
    indices : Tensor
    stratification : Optional[Tensor] = None

    def unpack_data(self):
        return self.inputs, self.targets, self.sample_weights

    @classmethod
    def from_arrow(
        cls, 
        input_table : pa.Table,
        targets : np.ndarray,
        sample_weights : Optional[np.ndarray] = None,
        *,
        column_names : list[str] = [],
        stratification : Optional[str | np.ndarray] = None,
        dtype = torch.float32,
        prefix : Optional[str] = None, 
        max_write_chunk = None,
        index_offset : int = 0,
    ):
       
        _stratification = None
        if isinstance(stratification, str):
            if stratification in input_table.column_names:
                _stratification = input_table[stratification].to_numpy()
        else:
            _stratification = stratification

        if len(column_names) > 0:
            input_table = input_table.select(column_names)
        else:
            column_names = input_table.column_names 
 
        _stratification_tensor = None
        if isinstance(_stratification, np.ndarray):
            _stratification = np.stack([np.asarray(targets), _stratification], axis=-1)
            _stratification_tensor = as_tensor(_stratification, dtype=torch.int)

        if sample_weights is None:
            sample_weights = np.ones(len(input_table))
        try:
            assert len(targets) == len(input_table)
            assert len(sample_weights) == len(input_table)
        except AssertionError:
            raise ValueError("input pa.Table and the ndarrays targets and" 
                             "sample_weights must all have the same lengths"
                             f"but got them with lengths {(len(input_table), len(targets), len(sample_weights))}" 
                             "respectively...")


        target_tensor = as_tensor(targets, dtype=dtype).unsqueeze_(1)
        sample_weights_tensor = as_tensor(sample_weights, dtype=dtype).unsqueeze_(1)
        index_tensor = torch.arange(len(input_table))+index_offset

        _stratification_init = MemoryMappedTensor.empty_like(
            _stratification_tensor,
        ) if _stratification_tensor is not None else None

        data = cls(
            inputs = MemoryMappedTensor.empty((len(input_table), len(column_names),), dtype=dtype),
            targets = MemoryMappedTensor.empty_like(target_tensor),
            sample_weights = MemoryMappedTensor.empty_like(sample_weights_tensor),
            indices = MemoryMappedTensor.empty_like(index_tensor),
            stratification = _stratification_init,
            batch_size=[len(input_table)],
            device="cpu"
        )

        data.memmap_(prefix=prefix)

        rbatches = input_table.to_batches(max_chunksize=max_write_chunk)

        print(f"num_batches : {len(rbatches)}")
        
        i = 0
        pbar = tqdm.tqdm(total=len(input_table))
        for rbatch in rbatches:
            _batch = len(rbatch)
            #print(f"Writing {i} to {i+_batch}...")
            pbar.update(_batch)
            _inputs_pylist = list(zip(*rbatch.to_pydict().values()))
            data[i : i + _batch] = cls(
                inputs = as_tensor(_inputs_pylist, dtype=dtype),
                targets = target_tensor[i : i + _batch],
                sample_weights = sample_weights_tensor[i : i + _batch],
                indices = index_tensor[i:i+_batch],
                stratification = _stratification_tensor[i : i + _batch] if _stratification_tensor is not None else None,
                batch_size=[_batch],
                device="cpu"
            )

            i += _batch
        return data

#class Collate(Module):
#    def __init__(
#        self, 
#        transform=None,
#        device:str="cpu",
#    ):
#        super().__init__()
#        self.transform = transform
#        self.device = torch.device(device)
#
#    def __call__(self, x: TabularDataset):
#        # move data to RAM
#        if self.device.type == "cuda":
#            out = x.pin_memory()
#        else:
#            out = x
#        if self.device:
#            # move data to gpu
#            out = out.to(self.device)
#        if self.transform:
#            # apply transforms on gpu
#            out.inputs = self.transform(out.inputs)
#        return out













