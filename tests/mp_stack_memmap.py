from functools import partial
from copy import deepcopy

from typing import Callable
from typing import Optional
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.func import grad_and_value, grad
from torch.func import functional_call
from torch.func import stack_module_state

from tensordict import TensorClass, TensorDict

from torchdata.nodes import SamplerWrapper
from torchdata.nodes import ParallelMapper
from torchdata.nodes import Loader, Header

from churten.data import TabularDataset
from churten.data import train_test_multi_subset_samplers
from churten.data import TensorBatchSampler

from churten.archs.nn import MLP
from churten.optimizer import FuncAdam

import torch.multiprocessing

def collate(data, indices):
    flattened_idx = torch.flatten(indices)
    return data[flattened_idx].reshape(indices.shape)
    #return data.gather(1, indices)

def get_loader(
    dataset: Sequence,
    batch_sampler : TensorBatchSampler,
    collate_fn : Optional[Callable] = None,
    num_workers: int = 0,
    load_only_first : Optional[int] = None,
):
    map_fn = collate_fn or getattr(dataset, "__getitems__", None)
    if map_fn is None:
        if isinstance(dataset, Tensor):
            map_fn = dataset.__getitem__
        else:
            raise ValueError("Either give a collate_fn or define a __getitems__ to sample mini-batches!")

    node = SamplerWrapper(batch_sampler)
    node = ParallelMapper(
        node, 
        map_fn=map_fn, 
        num_workers=num_workers, 
        method="process", 
        in_order=True,
    )

    if load_only_first is not None:
        node = Header(node, load_only_first)

    return Loader(node)

def loss_fn(
    base_model, 
    params, 
    buffers, 
    X, y, weight,
):
    y_pred = functional_call(base_model, (params, buffers), X)
    loss = binary_cross_entropy_with_logits(y_pred, y, weight=weight)
    return loss

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("start!")
    torch.manual_seed(0)
    base_model = MLP(
        layer_sizes = [21, 256, 256, 256, 1]
    )
    device = "cuda" 
    num_replicas = 10
    model_list : list = [deepcopy(base_model).to(device=device)]
    model_list.extend(deepcopy(model_list[0]) for _ in range(1, num_replicas))
    _params, _buffers = stack_module_state(model_list)
    params = TensorDict.from_dict(_params).requires_grad_(False)
    buffers = TensorDict.from_dict(_buffers).requires_grad_(False)

    base_model = base_model.to("meta")
    criterion = partial(loss_fn, base_model)
    
    optim = FuncAdam(fused = True, batch_size=[num_replicas])
    optim_state = optim.init(params).to(device=device)
 
    dataset = TabularDataset.load_memmap("det_lvl_ds")

    print("Subsampling and splitting dataset into train and valid batches ...")
    train_sampler, valid_sampler = train_test_multi_subset_samplers(
        dataset, 
        train_size = 0.6,
        subsample_size = 50000,
        num_replicas = num_replicas,
        batch_size = 10000,
        seed = 0,
        stratify = dataset.stratification,
    )
    print(f"Received {len(train_sampler)} train and {len(valid_sampler)} valid batches.")

    print("Creating loaders ...")
    train_loader = get_loader(
        dataset,
        train_sampler,
        collate_fn=partial(collate, dataset),
        #num_workers=5,
        load_only_first=3
    )
    
    valid_loader = get_loader(
        dataset,
        valid_sampler,
        collate_fn=partial(collate, dataset),
        #num_workers=5,
        load_only_first=3
    )

    criterion = partial(loss_fn, base_model)
    grads = deepcopy(params).zero_()

    nepochs = 50

    for iepoch in range(nepochs):
        print(f"Epoch: [{iepoch}/{nepochs}]")
        train_loss_sum = torch.zeros((num_replicas), device=device)
        for batch in train_loader:
            batch = batch.to(device=device)
            X, y, weight = batch.inputs, batch.targets, batch.sample_weights
            grads, losses = torch.vmap(grad_and_value(criterion))(
                params.to_dict(), buffers.to_dict(), X, y, weight
            )
            optim_state.grads = TensorDict.from_dict(grads)
            optim_state = optim.update(optim_state)
            train_loss_sum += losses

        train_loss_sum /= len(train_sampler)

        valid_loss_sum = torch.zeros((num_replicas), device=device)
        for batch in valid_loader:
            batch = batch.to(device=device)
            X, y, weight = batch.inputs, batch.targets, batch.sample_weights
            losses = torch.vmap(criterion)(
                params.to_dict(), buffers.to_dict(), X, y, weight
            )

            valid_loss_sum += losses

        valid_loss_sum /= len(valid_sampler)

        print("___Training losses:", train_loss_sum, ", Valid losses:", valid_loss_sum)








        

