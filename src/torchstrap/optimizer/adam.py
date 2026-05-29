from functools import partial
from beartype.typing import Optional

from optree.dataclasses import dataclass, field

from optree import tree_map, tree_map_

import torch
from torch import Tensor

from torchstrap.utils.typing import Vector, FloatScalarLike
from torchstrap.state import OptimState
from torchstrap.optimizer.grad_transform import GradientTransformation


@dataclass(namespace="torchstrap.state")
class AdamState(OptimState):
    exp_avgs               : list[Tensor] = field(default_factory=list)
    exp_avg_sqs            : list[Tensor] = field(default_factory=list)
    max_exp_avg_sqs        : list[Tensor] = field(default_factory=list)
    lr                     : Vector       = field(default_factory=partial(torch.as_tensor, 1e-3 ))
    beta1                  : Vector       = field(default_factory=partial(torch.as_tensor, 0.9  ))
    beta2                  : Vector       = field(default_factory=partial(torch.as_tensor, 0.999))
    eps                    : Vector       = field(default_factory=partial(torch.as_tensor, 1e-8 ))
    weight_decay           : Vector       = field(default_factory=partial(torch.as_tensor, 1e-2 ))
    amsgrad                : bool         = field(default=False, pytree_node=False)
    decoupled_weight_decay : bool         = field(default=True , pytree_node=False)

    def __post_init__(self):
        _zeros_like = partial(torch.zeros_like, memory_format=torch.preserve_format)
        if len(self.exp_avgs) == 0:
            self.exp_avgs = [_zeros_like(p) for p in self.params]

        if len(self.exp_avg_sqs) == 0:
            self.exp_avg_sqs = [_zeros_like(p) for p in self.params]

        if self.amsgrad and len(self.max_exp_avg_sqs) == 0:
            self.max_exp_avg_sqs = [_zeros_like(p) for p in self.params]
        super().__post_init__()

    @classmethod
    def from_param_dict(
        cls,
        param_pytree : dict[str, Tensor],
        /,
        batch_size : tuple[int, ...],
        *,
        lr           : FloatScalarLike = 1e-3,
        beta1        : FloatScalarLike = 0.9 ,
        beta2        : FloatScalarLike = 0.999,
        eps          : FloatScalarLike = 1e-8,
        weight_decay : FloatScalarLike = 1e-2,
        **kwargs : bool,
    ):
        return cls._from_pytree(
            param_pytree,
            batch_size=batch_size,
            lr = lr,
            beta1 = beta1,
            beta2 = beta2,
            eps = eps,
            weight_decay = weight_decay,
            **kwargs,
        )


def _bump_state_steps(state_steps: list[Tensor], active_mask: Tensor) -> None:
    if not state_steps:
        return
    inc = active_mask.to(state_steps[0].dtype)
    for s in state_steps:
        s.add_(inc)


def _update_one_param(
    p, g, m, v, mx, step,
    *,
    lr, beta1, beta2, eps, weight_decay,
    amsgrad: bool,
    maximize: bool,
    decoupled_weight_decay: bool,
):
    R = step.shape[0]
    extra = (1,) * (p.dim() - 1)
    b1_b  = beta1.view(R, *extra)
    b2_b  = beta2.view(R, *extra)
    eps_b = eps.view(R, *extra)
    wd_b  = weight_decay.view(R, *extra)
    lr_b  = lr.view(R, *extra)

    bc1_b = (1 - beta1.pow(step)).view(R, *extra)
    bc2_b = (1 - beta2.pow(step)).view(R, *extra)

    if maximize:
        g_eff = g.neg()
    else:
        g_eff = g

    if decoupled_weight_decay:
        p.mul_(1 - lr_b * wd_b)
    else:
        if g_eff is g:
            g_eff = g.add(p * wd_b)
        else:
            g_eff.add_(p * wd_b)

    m.lerp_(g_eff, 1 - b1_b)
    v.mul_(b2_b).add_(g_eff * g_eff * (1 - b2_b))

    if amsgrad:
        torch.maximum(mx, v, out=mx)
        denom = mx.sqrt()
    else:
        denom = v.sqrt()
    denom.div_(bc2_b.sqrt()).add_(eps_b)
    denom.mul_(bc1_b).div_(lr_b)
    p.addcdiv_(m, denom, value=-1.0)


def _adam_step_impl(
    params:          list[Tensor],
    grads:           list[Tensor],
    exp_avgs:        list[Tensor],
    exp_avg_sqs:     list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps:     list[Tensor],
    lr:              Tensor,
    beta1:           Tensor,
    beta2:           Tensor,
    eps:             Tensor,
    weight_decay:    Tensor,
    active_mask:     Tensor,
    amsgrad:         bool,
    maximize:        bool,
    decoupled_weight_decay: bool,
) -> Tensor:
    _bump_state_steps(state_steps, active_mask)

    inactive_idx = (~active_mask).nonzero(as_tuple=False).flatten()
    has_inactive = inactive_idx.numel() > 0

    # state_steps already handled by the masked bump above; only the per-param
    # tensors need snapshot/restore around the in-place math.
    mutable_lists = (
        params, exp_avgs, exp_avg_sqs,
        *((max_exp_avg_sqs,) if amsgrad else ()),
    )

    snapshots = None
    if has_inactive:
        snapshots = tree_map(
            lambda t: t.index_select(0, inactive_idx).clone(),
            mutable_lists,
        )

    mx_or_placeholder = max_exp_avg_sqs if amsgrad else [None] * len(params)
    per_param = partial(
        _update_one_param,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
        amsgrad=amsgrad, maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
    )
    tree_map_(
        per_param,
        params, grads, exp_avgs, exp_avg_sqs,
        mx_or_placeholder, state_steps,
        none_is_leaf=True,
    )

    if has_inactive and snapshots is not None:
        tree_map_(
            lambda t, s: t.index_copy_(0, inactive_idx, s),
            mutable_lists, snapshots,
        )

    # Dummy return so the op is composable with torch.func.vmap, which requires
    # Tensor outputs. The value is unused; all real outputs are the declared
    # mutations of `mutates_args`.
    return torch.zeros((), device=params[0].device, dtype=params[0].dtype)


def _adam_step_fake(
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
    lr, beta1, beta2, eps, weight_decay, active_mask,
    amsgrad, maximize, decoupled_weight_decay,
):
    return torch.empty((), device=params[0].device, dtype=params[0].dtype)


def _adam_step_vmap(
    info, in_dims,
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
    lr, beta1, beta2, eps, weight_decay, active_mask,
    amsgrad, maximize, decoupled_weight_decay,
):
    # Treat the outer vmap dim AS the replica dim. The deferred composition
    # (R > 1 inside vmap) is not handled here — callers should pass R == 1
    # single-replica state inside the vmap'd function. Per-param tensors then
    # arrive as (B, 1, *p_i), state_steps / hyperparams as (B, 1), and we
    # squeeze the singleton inner dim so the kernel sees a regular (R=B,*) layout.
    def lead(t, d):
        return t if d is None else t.movedim(d, 0)

    def lead_squeeze(t, d):
        t = lead(t, d)
        if t.dim() >= 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        return t

    params_b          = tree_map(lead_squeeze, params,          in_dims[0])
    grads_b           = tree_map(lead_squeeze, grads,           in_dims[1])
    exp_avgs_b        = tree_map(lead_squeeze, exp_avgs,        in_dims[2])
    exp_avg_sqs_b     = tree_map(lead_squeeze, exp_avg_sqs,     in_dims[3])
    max_exp_avg_sqs_b = (tree_map(lead_squeeze, max_exp_avg_sqs, in_dims[4])
                         if amsgrad else [])
    state_steps_b     = tree_map(lead_squeeze, state_steps,    in_dims[5])

    B = params_b[0].shape[0]

    def hyp(t, d):
        if d is not None:
            t = t.movedim(d, 0)
            if t.dim() >= 2 and t.shape[1] == 1:
                t = t.squeeze(1)
            return t
        return t.expand(B) if t.dim() == 0 else t

    lr_b, b1_b, b2_b, eps_b, wd_b = (
        hyp(lr,           in_dims[6]),
        hyp(beta1,        in_dims[7]),
        hyp(beta2,        in_dims[8]),
        hyp(eps,          in_dims[9]),
        hyp(weight_decay, in_dims[10]),
    )

    am_d = in_dims[11]
    if am_d is not None:
        mask_b = active_mask.movedim(am_d, 0)
        if mask_b.dim() >= 2 and mask_b.shape[1] == 1:
            mask_b = mask_b.squeeze(1)
    elif active_mask.dim() == 0:
        mask_b = active_mask.expand(B)
    else:
        mask_b = active_mask
    mask_b = mask_b.to(dtype=torch.bool)

    out = adam_step_(
        params_b, grads_b, exp_avgs_b, exp_avg_sqs_b, max_exp_avg_sqs_b,
        state_steps_b,
        lr_b, b1_b, b2_b, eps_b, wd_b, mask_b,
        amsgrad=amsgrad, maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
    )
    return out, None


# Assignment form (not @decorator) so beartype's claw doesn't wrap the
# resulting CustomOpDef and strip its register_fake / register_vmap methods.
adam_step_ = torch.library.custom_op(
    "torchstrap::adam_step_",
    _adam_step_impl,
    mutates_args={"params", "exp_avgs", "exp_avg_sqs",
                  "max_exp_avg_sqs", "state_steps"},
)
adam_step_.register_fake(_adam_step_fake)
adam_step_.register_vmap(_adam_step_vmap)


# -----------------------------------------------------------------------------
# CUDA backend: Helion fused kernel.
#
# Same op surface as the default Python in-place kernel above; registered as a
# CUDA-specific override via `register_kernel("cuda", ...)`. The Helion kernel
# does the per-replica gather + masked store inside one launch per param,
# eliminating the (R, *p) transient buffers (denom, g**2) that the PyTorch
# kernel above allocates.
# -----------------------------------------------------------------------------
try:
    import os
    import helion
    import helion.language as hl
    _HELION_AUTOTUNE_EFFORT = os.environ.get("TORCHSTRAP_HELION_EFFORT", "quick")

    # Seed-config hook for `autotune_seed_configs=`: seeds the autotuner's
    # initial population from a vetted config without constraining the search.
    # Left `None` on purpose. A single captured config can NOT be seeded here
    # because amsgrad/maximize/decoupled_wd are hl.constexpr — each of the 8
    # flag combinations compiles to a distinct specialization with a different
    # config-encoding dimension, so seeding one config raises an encoding-length
    # AssertionError on the other specializations. And it isn't needed: quick
    # autotune already converges to the same optimum a full-effort run finds
    # (e.g. block_sizes=[1, 32] on the multifold MLP shapes). The hook stays so a
    # user who pins the flags for a build can opt in.
    _ADAM_SEED_CONFIG = None  # type: helion.Config | None

    # `static_shapes=False` compiles a single shape-generic kernel — without it,
    # every distinct (R, per_rep) combination triggers a fresh autotune cycle.
    # `autotune_effort='quick'` keeps first-run startup bounded; set
    # `TORCHSTRAP_HELION_EFFORT=full` for a peak-perf autotune run.
    _HELION_KERNEL_KWARGS: dict = dict(
        static_shapes=False,
        autotune_effort=_HELION_AUTOTUNE_EFFORT,
    )
    if _ADAM_SEED_CONFIG is not None:
        _HELION_KERNEL_KWARGS["autotune_seed_configs"] = [_ADAM_SEED_CONFIG]

    @helion.kernel(**_HELION_KERNEL_KWARGS)
    def _adam_kernel_helion(
        p:    Tensor,   # (R, per_rep)
        g:    Tensor,
        m:    Tensor,
        v:    Tensor,
        mx:   Tensor,
        lr_r: Tensor,   # (R,)
        b1_r: Tensor,
        b2_r: Tensor,
        eps_r: Tensor,
        wd_r: Tensor,
        mask_r: Tensor,
        bc1_r: Tensor,
        bc2_r: Tensor,
        amsgrad: hl.constexpr,
        maximize: hl.constexpr,
        decoupled_wd: hl.constexpr,
    ) -> None:
        R, n = p.shape
        for tile_r, tile_n in hl.tile([R, n]):
            # Per-replica scalars load once per replica-tile as (block_r,) and
            # reshape to (block_r, 1) so they broadcast over the per_rep axis.
            lr_e   = lr_r[tile_r][:, None]
            b1_e   = b1_r[tile_r][:, None]
            b2_e   = b2_r[tile_r][:, None]
            eps_e  = eps_r[tile_r][:, None]
            wd_e   = wd_r[tile_r][:, None]
            bc1_e  = bc1_r[tile_r][:, None]
            bc2_e  = bc2_r[tile_r][:, None]
            # Full (block_r, block_n) mask: inactive replicas are simply never
            # stored, so no read-back / torch.where is needed.
            active = (mask_r[tile_r][:, None] != 0) & (tile_n.index[None, :] >= 0)

            p_e = p[tile_r, tile_n]
            g_e = g[tile_r, tile_n]
            m_e = m[tile_r, tile_n]
            v_e = v[tile_r, tile_n]

            if maximize:
                g_e = -g_e
            if decoupled_wd:
                p_e = p_e * (1.0 - lr_e * wd_e)
            else:
                g_e = g_e + p_e * wd_e

            m_new = m_e * b1_e + g_e * (1.0 - b1_e)
            v_new = v_e * b2_e + g_e * g_e * (1.0 - b2_e)

            if amsgrad:
                mx_e   = mx[tile_r, tile_n]
                mx_new = torch.maximum(mx_e, v_new)
                denom  = torch.sqrt(mx_new) / torch.sqrt(bc2_e) + eps_e
                hl.store(mx, [tile_r, tile_n], mx_new, extra_mask=active)
            else:
                denom = torch.sqrt(v_new) / torch.sqrt(bc2_e) + eps_e

            p_new = p_e - (lr_e / bc1_e) * m_new / denom

            hl.store(p, [tile_r, tile_n], p_new, extra_mask=active)
            hl.store(m, [tile_r, tile_n], m_new, extra_mask=active)
            hl.store(v, [tile_r, tile_n], v_new, extra_mask=active)


    def _adam_step_cuda(
        params:          list[Tensor],
        grads:           list[Tensor],
        exp_avgs:        list[Tensor],
        exp_avg_sqs:     list[Tensor],
        max_exp_avg_sqs: list[Tensor],
        state_steps:     list[Tensor],
        lr:              Tensor,
        beta1:           Tensor,
        beta2:           Tensor,
        eps:             Tensor,
        weight_decay:    Tensor,
        active_mask:     Tensor,
        amsgrad:         bool,
        maximize:        bool,
        decoupled_weight_decay: bool,
    ) -> Tensor:
        # Bump state-steps once for all params (active replicas only).
        _bump_state_steps(state_steps, active_mask)

        # Helion's `tensor[index]` gather wants a numeric dtype, not bool.
        mask_int = active_mask.to(torch.uint8)

        mx_iter = max_exp_avg_sqs if amsgrad else [None] * len(params)
        for p, g, m, v, mx, s in zip(
            params, grads, exp_avgs, exp_avg_sqs, mx_iter, state_steps,
        ):
            R = s.shape[0]
            bc1 = 1.0 - beta1.pow(s)
            bc2 = 1.0 - beta2.pow(s)

            # Reshape the stacked (R, *p_i) state to (R, per_rep) so the kernel
            # tiles over (replica, per_replica) — contiguous, so this is a view.
            # `mx` must be a real tensor for Helion's signature; when not
            # amsgrad, alias to `p` and the constexpr branch never touches it.
            mx_arg = (mx if amsgrad else p).reshape(R, -1)

            _adam_kernel_helion(
                p.reshape(R, -1), g.reshape(R, -1), m.reshape(R, -1),
                v.reshape(R, -1), mx_arg,
                lr, beta1, beta2, eps, weight_decay, mask_int,
                bc1, bc2,
                hl.constexpr(amsgrad),
                hl.constexpr(maximize),
                hl.constexpr(decoupled_weight_decay),
            )

        return torch.zeros((), device=params[0].device, dtype=params[0].dtype)


    adam_step_.register_kernel("cuda")(_adam_step_cuda)

    _HAS_HELION = True
except ImportError:
    _HAS_HELION = False


class Adam(metaclass=GradientTransformation):
    state_class : type[OptimState] = AdamState

    @classmethod
    def update(cls, state: AdamState, active_mask: Tensor) -> AdamState:
        if state.batch_size == ():
            params          = [p.unsqueeze(0) for p in state.params]
            grads           = [g.unsqueeze(0) for g in state.grads]
            exp_avgs        = [m.unsqueeze(0) for m in state.exp_avgs]
            exp_avg_sqs     = [v.unsqueeze(0) for v in state.exp_avg_sqs]
            max_exp_avg_sqs = (
                [mx.unsqueeze(0) for mx in state.max_exp_avg_sqs]
                if state.amsgrad else []
            )
            state_steps  = [s.view(1) for s in state.state_steps]
            lr   = state.lr.view(1)
            beta1 = state.beta1.view(1)
            beta2 = state.beta2.view(1)
            eps  = state.eps.view(1)
            wd   = state.weight_decay.view(1)
        else:
            params          = state.params
            grads           = state.grads
            exp_avgs        = state.exp_avgs
            exp_avg_sqs     = state.exp_avg_sqs
            max_exp_avg_sqs = state.max_exp_avg_sqs if state.amsgrad else []
            state_steps     = state.state_steps
            lr, beta1, beta2, eps, wd = (
                state.lr, state.beta1, state.beta2, state.eps, state.weight_decay,
            )

        adam_step_(
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
            state_steps,
            lr, beta1, beta2, eps, wd,
            active_mask,
            amsgrad=state.amsgrad,
            maximize=state.maximize,
            decoupled_weight_decay=state.decoupled_weight_decay,
        )
        return state
