# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Hopper HC combine and rounded Gemma RMSNorm in a single launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_combine_norm import (
    _prepare_hc_combine_norm,
    _prune_hc_combine_norm_configs,
)
from flaggems_vllm.ops.hc_combine_norm import (
    hc_combine_norm as _generic_hc_combine_norm,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_combine_norm_hopper"),
    key=["SPEC"],
    strategy=repr,
    rep=20,
    use_cuda_graph=False,
    prune_configs_by={"early_config_prune": _prune_hc_combine_norm_configs},
)
@triton.jit
def _hc_combine_norm_hopper_kernel(
    R,
    B,
    I,
    W,
    O,
    Y,
    SPEC: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SR: tl.constexpr = SPEC[3]
    SB: tl.constexpr = SPEC[4]
    SI: tl.constexpr = SPEC[5]
    SHARED: tl.constexpr = SPEC[6]
    HAS_LOGITS: tl.constexpr = SPEC[7]
    pid = tl.program_id(0).to(tl.int64)
    row = pid // HC
    branch = pid % HC
    # NaN -> integer conversion can make masked padding nonzero. Match the
    # native reduction extent for integer/bool residuals before normalization.
    if O.dtype.element_ty.is_int():
        tl.static_assert(
            BLOCK_H == 512, "Integer outputs require native reduction tile"
        )
        PADDED_H: tl.constexpr = 512 * triton.next_power_of_2(triton.cdiv(H, 512))
        TILE_H: tl.constexpr = min(BLOCK_H, PADDED_H)
        NUM_TILES: tl.constexpr = PADDED_H // TILE_H
    else:
        TILE_H: tl.constexpr = BLOCK_H
        NUM_TILES: tl.constexpr = triton.next_power_of_2(triton.cdiv(H, BLOCK_H))
    tiles = tl.arange(0, NUM_TILES)
    columns = tiles[:, None] * TILE_H + tl.arange(0, TILE_H)[None, :]
    mask = columns < H
    offsets = branch * H + columns
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    residual = tl.load(R + row * SR + offsets, mask, other=0.0).to(tl.float32)
    block = tl.load(B + row * SB + columns, mask, other=0.0).to(tl.float32)
    if HAS_LOGITS:
        injection = tl.load(I + row * SI + branch).to(tl.float32)
        gate = 2.0 * tl.sigmoid(injection / HC)
        block = block * gate
    combined = (residual + block).to(O.dtype.element_ty)
    weight_offsets = columns if SHARED else offsets
    if not HAS_LOGITS:
        weight = tl.load(W + weight_offsets, mask, other=0.0).to(tl.float32)
    tl.store(O + row * HC * H + offsets, combined, mask)
    rounded = combined.to(tl.float32)
    sum_sq = tl.sum(tl.sum(rounded * rounded, axis=1), axis=0)
    inverse_rms = tl.rsqrt(sum_sq / H + EPS)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    if HAS_LOGITS:
        weight = tl.load(W + weight_offsets, mask, other=0.0).to(tl.float32)
    normalized = rounded * inverse_rms
    result = normalized + normalized * weight
    tl.store(Y + row * HC * H + offsets, result, mask)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the native interface, including the materialized rounding boundary."""
    if eps < 0:
        return _generic_hc_combine_norm(
            residual, block_output, injection_logits, norm_weight, eps, hc_count
        )
    combined, normalized, spec = _prepare_hc_combine_norm(
        residual, block_output, injection_logits, norm_weight, hc_count
    )
    if spec[0] == 0:
        return combined, normalized
    with runtime.torch_device_fn.device(residual.device):
        _hc_combine_norm_hopper_kernel[(spec[0] * spec[2],)](
            residual,
            block_output,
            injection_logits,
            norm_weight,
            combined,
            normalized,
            spec,
            eps,
            launch_pdl=True,
        )
    return combined, normalized
