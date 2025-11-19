from copy import deepcopy
from functools import partial

import torch
from torch.nn import Sequential, Linear, ReLU
from torch.optim import AdamW
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.func import functional_call, vmap, grad_and_value, stack_module_state

from churten.optimizer import Adam

def loss_fn(base_model, params, inputs, targets):
    preds = functional_call(base_model, (params, {}), inputs)
    return binary_cross_entropy_with_logits(preds, targets)

@torch.no_grad()
def functional_impl(net, inputs, targets, num_replicas=2):
    nets = [deepcopy(net) for _ in range(num_replicas)]
    params, _ = stack_module_state(nets)
    optimizer = Adam
    state = optimizer.init(params, batch_dim=0)

    criterion = partial(loss_fn, nets[0].to("meta"))
    grad, loss = vmap(grad_and_value(criterion))(
        params, 
        inputs.expand(num_replicas, 10, 5), 
        targets.expand(num_replicas, 10, 1),
    )
    optimizer.apply_gradient(grad, state, batch_dim=0)

    return state

def vanilla_impl(net, inputs, targets):
    optim = AdamW(net.named_parameters())
    loss = binary_cross_entropy_with_logits(net(inputs), targets)
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    return dict(net.named_parameters())

if __name__ == "__main__":

    net = Sequential(
        Linear(5, 10),
        ReLU(),
        Linear(10, 1),
    )

    inputs = torch.rand(10, 5)
    targets = torch.cat([torch.zeros(5, 1), torch.ones(5, 1)])

    fstate = functional_impl(net, inputs, targets)
    sparams = vanilla_impl(net, inputs, targets)

    for s in fstate:
        for ipar, par in enumerate(s.param_names):
            assert torch.allclose(
                s.params[ipar], sparams[par]
            )

    print("Pass")

     



