from functools import partial
from copy import deepcopy

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.func import grad_and_value
from torch.func import functional_call
from torch.func import stack_module_state

from churten.data import TabularDataset
from churten.data import train_test_multi_subset_samplers
from churten.data import get_stacked_batch_loader

from churten.archs.nn import MLP
from churten.optimizer import FuncAdam

device = "cuda" 

def module_call(
    base_model,
    params,
    buffers,
    X,
):
    return functional_call(base_model, (params, buffers), X)


def loss_fn(
    base_model, 
    params, 
    buffers, 
    batch,
):
    batch = batch.to(device=next(iter(params.values())).device)
    y_pred = module_call(base_model, params, buffers, batch.inputs)
    loss = binary_cross_entropy_with_logits(y_pred, batch.targets, weight=batch.sample_weights)
    return loss

def fit(
    optimizer,
    base_model,
    params,
    buffers,
    nepochs,
    train_loader,
    valid_loader,
):
    for iepoch in range(nepochs):
        print("Epoch:", iepoch)
        train_loss_sum = torch.zeros((num_replicas), device=device)
        for batch in train_loader:
            grads, losses = torch.vmap(
                grad_and_value(partial(loss_fn, base_model))
            )(
                params, buffers, batch
            )
            optimizer.apply_gradients(grads)
            train_loss_sum += losses

        train_loss_sum /= len(train_sampler)
        print("___Training losses:", train_loss_sum.mean(), "+/-", train_loss_sum.std())

        valid_loss_sum = torch.zeros((num_replicas), device=device)
        for batch in valid_loader:
            losses = torch.vmap(partial(loss_fn, base_model))(
                params, buffers, batch
            )
            valid_loss_sum += losses

        valid_loss_sum /= len(valid_sampler)
        print("___Valid losses:", valid_loss_sum.mean(), "+/-", valid_loss_sum.std())

def infer(
    base_model,
    params,
    buffers,
    data_loader,
):
    return torch.cat(
        [
            torch.vmap(
                partial(module_call, base_model)
            )(
                params, buffers, batch.to(device=next(iter(params.values())).device).inputs
            ) for batch in data_loader
        ]
    )


if __name__ == "__main__":
    print("start!")
    torch.manual_seed(0)
    base_model = MLP(
        layer_sizes = [21, 256, 256, 256, 1]
    )
    num_replicas = 10
    model_list : list = [deepcopy(base_model).to(device=device)]
    model_list.extend(deepcopy(model_list[0]) for _ in range(1, num_replicas))
    stacked_params, stacked_buffers = stack_module_state(model_list)
    for params in stacked_params.values():
        params.requires_grad_(False)
    
    optim = FuncAdam(batch_size=[num_replicas])
    optim.state.init(stacked_params)
    optim.state.to(device=device)
    
    for buffers in stacked_buffers.values():
        buffers.requires_grad_(False)
 
    base_model = base_model.to("meta")
    dataset = TabularDataset.load_memmap("det_lvl_ds")

    print("Subsampling and splitting dataset into train and valid batches ...")
    train_sampler, valid_sampler = train_test_multi_subset_samplers(
        dataset, 
        train_size = 0.6,
        subsample_size = 500000,
        num_replicas = num_replicas,
        batch_size = 10000,
        seed = 0,
        stratify = dataset.stratification,
    )
    print(f"Received {len(train_sampler)} train and {len(valid_sampler)} valid batches.")

    print("Creating loaders ...")
    train_loader = get_stacked_batch_loader(dataset, train_sampler)
    
    valid_loader = get_stacked_batch_loader(dataset, valid_sampler)


    nepochs = 50

    fit(
        optim,
        base_model,
        stacked_params,
        stacked_buffers,
        nepochs,
        train_loader,
        valid_loader,
    )









        

