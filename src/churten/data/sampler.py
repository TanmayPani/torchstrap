import math
import itertools
from typing import Optional, Sequence
from collections.abc import Sized, Iterator

import torch
from torch import Tensor, LongTensor
from torch import Generator
from torch.utils.data import Sampler, SequentialSampler
from torch.utils.data import BatchSampler

def _split(
    input : Tensor,
    sizes : Sequence[float | int],
    generator : Optional[Generator] = None,
):
    g = generator or torch.default_generator
    index = torch.randperm(input.shape[0], generator=g)
    
    input_size = input.shape[0]
    sum_of_sizes = sum(sizes)
    split_sizes = [math.floor((size/sum_of_sizes)*input.shape[0]) for size in sizes]
    remainder = input_size - sum(split_sizes)
    for irem in range(remainder):
        isplit = irem % len(split_sizes)
        split_sizes[isplit] += 1

    split_indices = [
        index[offset - length : offset] 
        for offset, length in zip(itertools.accumulate(split_sizes), split_sizes)
    ]

    return [input[idx] for idx in split_indices]


def _stratified_split(
    input : Tensor,
    label : Tensor,
    sizes : Sequence[float | int],
    generator : Optional[Generator] = None,
):
    assert input.shape[0] == label.shape[0]

    unique_labels, index_to_label_map = torch.unique(
        label,
        dim = 0,
        return_inverse = True
    )

    g = generator or torch.default_generator
    label_splits = [
        _split(
            input[index_to_label_map == l],
            sizes,
            generator=g,
        ) for l in range(unique_labels.shape[0])
    ]
    
    splits = [torch.cat(lsplit) for lsplit in zip(*label_splits)]

    return [
        split[
            torch.randperm(split.shape[0], generator=g)
        ] for split in splits
    ] 

def _random_split(
    input : Tensor,
    sizes : Sequence[float | int],
    stratify : Optional[Tensor] = None,
    generator : Optional[Generator] = None,
):

    if stratify is not None:
        return _stratified_split(
            input, 
            stratify, 
            sizes,
            generator=generator,
        )
    else:
        return _split(
            input,
            sizes,
            generator=generator,
        )

def random_split(
    *inputs : Tensor,
    sizes : Sequence[float | int],
    stratify : Optional[Tensor] = None,
    generator : Optional[Generator] = None,
):

    lengths = [input.shape[0] for input in inputs]
    assert all(l == lengths[0] for l in lengths)
    index_splits = _random_split(
        torch.arange(lengths[0]), 
        sizes, 
        stratify=stratify, 
        generator=generator,
    )

    return [input[idx] for idx in index_splits for input in inputs]

def undersample(
    *inputs : Tensor,
    size : float | int,
    stratify : Optional[Tensor] = None,
    generator : Optional[Generator] = None,
):    

    lengths = [input.shape[0] for input in inputs]
    assert all(l == lengths[0] for l in lengths)

    sub_size : int = 0
    if isinstance(size, float):
        if size > 1.0:
            raise ValueError("Size argument must be <= 1.0 if float")
        sub_size = math.floor(size*lengths[0])
    elif isinstance(size, int):
        if size > lengths[0]:
            raise ValueError("Size argument must be < input.shape[0] if int")
        sub_size = size
    
    if sub_size == 0:
        raise ValueError("Can't make subsample of size 0!!")

    g = generator or torch.default_generator

    if stratify is None:
        sub_index = torch.randperm(lengths[0], generator=g)[0:sub_size]
    else:
        assert lengths[0] == stratify.shape[0]
        unique_labels, index_to_label_map = torch.unique(
            stratify,
            dim = 0,
            return_inverse = True
        )

        index = torch.arange(lengths[0])
        label_index_list = [
            index[index_to_label_map == l] 
            for l in range(unique_labels.shape[0])
        ] 
        label_sub_size_list = [
            math.floor(float(sub_size/lengths[0])*lidx.shape[0]) 
            for lidx in label_index_list
        ]
        remainder = sub_size - sum(label_sub_size_list)
        for i in range(remainder):
            label_sub_size_list[i % unique_labels.shape[0]] += 1

        sub_index = torch.cat([
            id[torch.randperm(id.shape[0], generator=g)[0:s]] 
            for id, s in zip(label_index_list, label_sub_size_list)
        ])[torch.randperm(sub_size, generator=g)]

    return [input[sub_index] for input in inputs]

class TensorBatchSampler(Sampler[LongTensor]):
    def __init__(
        self, 
        source : Tensor,
        batch_size : int = 32,
        batch_dim : int = 0,
        drop_last : bool = False,
    ) -> None:
        self.source = source
        self.batch_dim = batch_dim
        if self.source.shape[self.batch_dim] < batch_size:
            raise ValueError("Subsample size should be greater than batch size!")

        self.batch_sampler = BatchSampler(
            SequentialSampler(torch.arange(self.source.shape[self.batch_dim]).to("meta")),
            batch_size = batch_size,
            drop_last = drop_last,
        )

    def __iter__(self) -> Iterator[LongTensor]:
        for batch_indices in self.batch_sampler:
            index_slice = (slice(None),)*self.batch_dim
            yield self.source[*index_slice, batch_indices, ...]
 
    def __len__(self) -> int:
        return len(self.batch_sampler)


class MultiSubsetBatchSampler(TensorBatchSampler):
    def __init__(
        self, 
        dataset: Sized, 
        subsample_size : float | int = 1.0,
        num_replicas : int = 2,
        batch_size : int = 32,
        seed : Optional[int] = None,
        stratify : Optional[Tensor] = None,
        drop_last : bool = False,
    ) -> None:

        self.length = len(dataset)
        self.subsample_size = subsample_size if isinstance(subsample_size, int) else math.floor(self.length*subsample_size)
        
        self.all_indices = torch.arange(self.length)
        self.seed = int(torch.empty((), dtype=torch.long).item()) % 2**32 if seed is None else seed

        rng_list = [Generator().manual_seed(self.seed + icopy) for icopy in range(num_replicas)]

        stacked_indices = torch.stack([
            undersample(
                self.all_indices,
                size=self.subsample_size,
                stratify=stratify,
                generator=rng,
            )[0] for rng in rng_list
        ])

        super().__init__(stacked_indices, batch_size=batch_size, batch_dim=1, drop_last=drop_last)

def train_test_multi_subset_samplers(
    dataset: Sized, 
    train_size : float | int,
    subsample_size : float | int = 1.0,
    num_replicas : int = 2,
    batch_size : int = 32,
    seed : Optional[int] = None,
    stratify : Optional[Tensor] = None,
    drop_last : bool = False,
):
    length = len(dataset)
    subsample_size = subsample_size if isinstance(subsample_size, int) else math.floor(length*subsample_size)

    train_size = train_size if isinstance(train_size, int) else math.floor(subsample_size*train_size)
    if train_size > subsample_size:
        raise ValueError("training set size must be smaller than subset size.")
    valid_size = subsample_size - train_size
        
    all_indices = torch.arange(length)
    seed = int(torch.empty((), dtype=torch.long).item()) % 2**32 if seed is None else seed

    train_subset_index_list = []
    valid_subset_index_list = []

    for icopy in range(num_replicas):
        g = Generator().manual_seed(seed + icopy)
        subset_index = undersample(
            all_indices,
            size=subsample_size,
            stratify=stratify,
            generator=g,
        )[0]

        subset_stratify = stratify[subset_index] if stratify is not None else None

        train_subset_index, valid_subset_index = random_split(
            subset_index, 
            sizes=(train_size, valid_size),
            stratify=subset_stratify,
            generator=g,
        )

        train_subset_index_list.append(train_subset_index)
        valid_subset_index_list.append(valid_subset_index)

    train_subset_stacked = torch.stack(train_subset_index_list)
    valid_subset_stacked = torch.stack(valid_subset_index_list)

    return (TensorBatchSampler(train_subset_stacked, batch_size=batch_size, batch_dim=1, drop_last=drop_last), 
        TensorBatchSampler(valid_subset_stacked, batch_size=batch_size, batch_dim=1, drop_last=drop_last))

