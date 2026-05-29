from copy import deepcopy
from functools import partial

import torch
from torch.nn import Sequential, Linear, ReLU
from torch.optim import AdamW
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.func import functional_call, vmap, grad_and_value, stack_module_state

from torchstrap.optimizer import AdamState
import optree


if __name__ == "__main__":

    net = Sequential(
        Linear(5, 10),
        ReLU(),
        Linear(10, 1),
    )

    states, _= stack_module_state([net, deepcopy(net)])

    print(states["0.weight"].shape)

    adam_state = AdamState.from_param_dict(states, batch_dim=0)
    spc = optree.tree_structure(adam_state, namespace="torchstrap.state")
    print(optree.treespec_paths(spc))
    with torch.no_grad():
        stub = adam_state.unbind()
        print(stub[0].exp_avgs[1].add_(2))
        print(adam_state.exp_avgs[1])

    print(adam_state.batch_dim, stub[0].batch_dim)
 #   print(list(adam_state.__dataclass_fields__.keys()))



