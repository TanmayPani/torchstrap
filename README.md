# torchstrap

**Train an ensemble of N model replicas in parallel on a single GPU — vectorized, not looped.**

`torchstrap` is a PyTorch training framework for *model ensembles*. Instead of looping
over N models, every replica is a slice along a leading "batch" dimension and the
entire ensemble is trained in one vectorized pass with [`torch.func.vmap`](https://pytorch.org/docs/stable/func.html).
Models run **statelessly** — parameters and buffers live outside the `nn.Module` and
are threaded in explicitly via `functional_call` — which makes per-replica
early-stopping, checkpointing, and bootstrap resampling first-class.

The headline component is a **fused, batched Adam optimizer** written in
[Helion](https://github.com/pytorch/helion) (a Triton-emitting DSL). It updates all
N replicas in a single GPU launch per parameter and is **up to 4.5× faster** than the
loop most people would write, with **no extra memory**.

---

## Benchmark: fused batched Adam vs. vanilla PyTorch

The baseline ("vanilla") is the obvious approach: `N` independent
`torch.optim.Adam` instances, each `.step()`-ed in a Python loop. `torchstrap` replaces
that with one `adam_step_` call over the stacked `(N, *param)` state, dispatched to a
fused Helion CUDA kernel.

Single optimizer step, fp32, NVIDIA RTX 4070 Laptop GPU (torch 2.9 / CUDA 12.8),
median of 20 runs:

| Workload                                  |  N  | vanilla (loop) | **torchstrap (fused)** | speedup | peak mem (vanilla → torchstrap) |
| ----------------------------------------- | --: | -------------: | ------------------: | :-----: | :--------------------------: |
| 6 tensors · 265k params/replica           |   8 |        738 µs  |          **394 µs** | **1.9×**|        33.3 → 32.3 MB        |
| 6 tensors · 265k params/replica           |  32 |       2.99 ms  |         **1.14 ms** | **2.6×**|       130.3 → 129.3 MB       |
| 6 tensors · 265k params/replica           | 100 |      12.80 ms  |         **2.84 ms** | **4.5×**|       405.1 → 403.9 MB       |

The speedup grows with ensemble size — at `N=100` the Python-loop launch overhead
dominates the baseline, while `torchstrap` issues one fused launch per parameter. Peak
memory **matches vanilla** because the kernel fuses the whole Adam update (moment
updates, bias correction, `sqrt`, decoupled weight decay) and allocates **zero
`(N, *param)` temporaries**.

Reproduce: `uv run python test/optimizer/bench_inplace_adam.py`

---

## Highlights

- **Vectorized ensembles.** N replicas train in one `vmap`-ed forward/backward pass;
  no Python loop over models.
- **Fused batched Adam.** A custom `torch.library` op (`torchstrap::adam_step_`) with a
  Helion CUDA kernel that updates all replicas in a single launch, with branchless
  per-replica masking (frozen replicas are skipped inside the kernel — no
  snapshot/restore). Pure-PyTorch CPU fallback; identical numerics to
  `torch.optim.Adam` (verified to `atol=rtol=1e-5` across the amsgrad / maximize /
  decoupled-weight-decay matrix, including distinct per-replica hyperparameters).
- **Per-replica everything.** Each replica can have its own learning rate,
  early-stopping schedule, and checkpoint — useful for bootstrap ensembles and
  hyperparameter sweeps.
- **Stateless by design.** Weights live in an explicit `State` pytree, not inside the
  module, enabling clean functional transforms and `torch.compile`.
- **skorch-style callbacks.** `EarlyStopping`, `Checkpoint`, `LRScheduler`,
  scoring, and logging hooks — all replica-aware.
- **Runtime type safety.** The whole package is `beartype`-checked at runtime.

---

## Install

Requires Python ≥ 3.13. Uses [`uv`](https://docs.astral.sh/uv/) (the lockfile pins
torch's CPU/CUDA wheel automatically via `UV_TORCH_BACKEND=auto`):

```bash
uv sync
```

`helion` is included for the fused CUDA kernel; on CPU-only installs `torchstrap`
transparently falls back to the pure-PyTorch Adam implementation.

---

## Quickstart

Train a 100-member bootstrap ensemble of MLP classifiers in parallel:

```python
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import binary_cross_entropy_with_logits

from torchstrap.stateless import StatelessModule
from torchstrap.optimizer import Adam
from torchstrap.callbacks import Checkpoint, EarlyStopping

def make_mlp(*sizes):
    layers = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [Linear(a, b), ReLU()]
    layers += [Linear(sizes[-2], sizes[-1])]
    return Sequential(*layers)

# Deep-copies the model into 100 independently-initialized replicas, stacks their
# params/buffers, and builds the optimizer + State. `optimizer` is the Adam *class*.
ensemble, optimizer, state = StatelessModule.init(
    make_mlp,
    Adam,
    model_init_args=(2, 512, 512, 1),
    num_replicas=100,
    device="cuda",
    init_randomness="different",          # each replica gets distinct initial weights
)

# `data_iterator` yields (input, target, sample_weight) with a leading replica dim,
# e.g. bootstrap-resampled minibatches of shape (num_replicas, batch, *features).
history = ensemble.fit(
    optimizer,
    binary_cross_entropy_with_logits,
    state,
    data_iterator,
    callbacks=[
        ("checkpoint",     Checkpoint(monitor="train_loss_best")),  # per-replica save
        ("early_stopping", EarlyStopping(monitor="train_loss")),    # per-replica freeze
    ],
)
```

Per-replica predictions on a grid (the ensemble mean is a calibrated probability):

```python
from functools import partial
from torch.func import vmap, functional_call
from torch.nn.functional import sigmoid

def predict(model, params, buffers, x):
    return sigmoid(functional_call(model, (params, buffers), x))

with torch.inference_mode():
    # points: (num_replicas, num_points, 2)
    probs = vmap(partial(predict, ensemble._base_model))(
        state.param_dict, state.buffer_dict, points
    )
    ensemble_mean = probs.mean(dim=0)     # average over the 100 replicas
```

A full runnable version (with plots of the loss curve and decision boundary) lives
in [`examples/spirals/spirals_parallel.py`](examples/spirals/spirals_parallel.py):

```bash
uv run python examples/spirals/spirals_parallel.py
```

It trains 100 bootstrap replicas on a noisy two-spirals dataset and renders the
ensemble's averaged decision boundary — a cheap, well-calibrated uncertainty estimate.

---

## How it fits together

Four small abstractions interlock:

| Component          | Role                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `StatelessModule`  | The trainer. Holds the base model on the `meta` device (no weights) and runs the `vmap`-ed forward/backward.  |
| `State`            | All training state as a pytree: stacked `param_dict` / `buffer_dict` + optimizer state, with a leading `N` dim. Unbinds into per-replica views for masking/checkpointing. |
| `GradientTransformation` | Optimizers are **classes**, not instances (a metaclass). `Adam` defines an `AdamState` and an `update` classmethod; `apply_gradient` drives the fused kernel. |
| `Callback`         | skorch-style hooks (`on_epoch_end`, `on_grad_computed`, …), all replica-aware.                                |

The training step computes per-replica gradients with
`vmap(grad_and_value(forward))`, stores them on the `State`, and calls the fused
`adam_step_` over the whole stacked ensemble.

---

## Testing & benchmarking

The files under `test/optimizer/` are standalone scripts:

```bash
uv run python test/optimizer/test_helion_adam_parity.py    # CPU vs CUDA numerical parity
uv run python test/optimizer/test_inplace_adam.py          # parity vs torch.optim.Adam
uv run python test/optimizer/test_inplace_adam_masking.py  # per-replica freeze semantics
uv run python test/optimizer/test_inplace_adam_vmap.py     # vmap composability
uv run python test/optimizer/bench_inplace_adam.py         # the benchmark above
```

Set `TORCHSTRAP_HELION_EFFORT={none,quick,full}` to trade kernel autotuning time for
steady-state speed (`quick` is the default and already converges to the
full-effort-optimal config on these workloads).
