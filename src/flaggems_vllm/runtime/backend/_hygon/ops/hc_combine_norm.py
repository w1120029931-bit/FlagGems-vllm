# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_combine_norm import _prepare_hc_combine_norm
from flaggems_vllm.ops.hc_combine_norm import (
    hc_combine_norm as _generic_hc_combine_norm,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_combine_norm_hygon"),
    key=["SPEC"],
    strategy=[repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_combine_norm_kernel(
    R, B, LOGITS, W, OUT, Y, SPEC: tl.constexpr, EPS: tl.constexpr
):
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SR: tl.constexpr = SPEC[3]
    SB: tl.constexpr = SPEC[4]
    SI: tl.constexpr = SPEC[5]
    SHARED: tl.constexpr = SPEC[6]
    HAS_LOGITS: tl.constexpr = SPEC[7]
    pid = tl.program_id(0).to(tl.int64)
    row, branch = pid // HC, pid % HC
    C0: tl.constexpr = triton.next_power_of_2(H + 1) // 2
    C1: tl.constexpr = triton.next_power_of_2(H - C0) if H > C0 else 0
    WIDTHS: tl.constexpr = (C0, C1)
    CHUNKS: tl.constexpr = 2 if C1 else 1
    if HAS_LOGITS:
        logit = tl.load(LOGITS + row * SI + branch).to(tl.float32)
        gate = 2.0 * tl.sigmoid(logit / HC)
    values = ()
    weights = ()
    for tile in tl.static_range(CHUNKS):
        columns = (C0 if tile else 0) + tl.arange(0, WIDTHS[tile])
        offsets = branch * H + columns
        mask = columns < H
        residual = tl.load(R + row * SR + offsets, mask, other=0.0).to(tl.float32)
        block = tl.load(B + row * SB + columns, mask, other=0.0).to(tl.float32)
        weight_offsets = columns if SHARED else offsets
        weight = tl.load(W + weight_offsets, mask, other=0.0).to(tl.float32)
        if HAS_LOGITS:
            block *= gate
        combined = (residual + block).to(OUT.dtype.element_ty)
        tl.store(OUT + row * HC * H + offsets, combined, mask)
        values += (combined.to(tl.float32),)
        weights += (weight,)
    squared = values[0] * values[0]
    if C1:
        partial = tl.sum(tl.reshape(squared, (C1, C0 // C1)), axis=1)
        partial += values[1] * values[1]
        sum_sq = tl.sum(partial, axis=0)
    else:
        sum_sq = tl.sum(squared, axis=0)
    inverse_rms = tl.rsqrt(tl.fma(sum_sq, 1.0 / H, EPS))
    for tile in tl.static_range(CHUNKS):
        columns = (C0 if tile else 0) + tl.arange(0, WIDTHS[tile])
        offsets = branch * H + columns
        normalized = values[tile] * inverse_rms
        normalized = tl.fma(normalized, weights[tile], normalized)
        tl.store(Y + row * HC * H + offsets, normalized, columns < H)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eps < 0 or (
        isinstance(residual, torch.Tensor)
        and residual.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return _generic_hc_combine_norm(
            residual, block_output, injection_logits, norm_weight, eps, hc_count
        )
    output, normalized, spec = _prepare_hc_combine_norm(
        residual, block_output, injection_logits, norm_weight, hc_count
    )
    if spec[0] == 0:
        return output, normalized
    with runtime.torch_device_fn.device(residual.device):
        _hc_combine_norm_kernel[(spec[0] * spec[2],)](
            residual,
            block_output,
            injection_logits,
            norm_weight,
            output,
            normalized,
            SPEC=spec,
            EPS=eps,
        )
    return output, normalized


__all__ = ["hc_combine_norm"]
