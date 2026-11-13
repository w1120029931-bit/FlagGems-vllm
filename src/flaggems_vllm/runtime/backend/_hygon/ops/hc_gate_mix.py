# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Hygon HC gated mixing with row tiling and sequential FP32 accumulation."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_gate_mix import _prepare_hc_gate_mix
from flaggems_vllm.ops.hc_gate_mix import hc_gate_mix as _generic_hc_gate_mix
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_gate_mix_hygon"),
    key=["SPEC"],
    strategy=repr,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_gate_mix_hygon_kernel(
    X, G, Y, SPEC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    N: tl.constexpr = SPEC[0]
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SX: tl.constexpr = SPEC[3]
    SG: tl.constexpr = SPEC[4]
    rows = (tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    columns = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (rows[:, None] < N) & (columns[None, :] < H)
    acc = tl.zeros((BLOCK_M, BLOCK_H), tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * H + columns[None, :]
        gate = tl.load(G + rows[:, None] * SG + offsets, mask, other=0.0).to(tl.float32)
        x = tl.load(X + rows[:, None] * SX + offsets, mask, other=0.0).to(tl.float32)
        if X.dtype.element_ty == tl.bfloat16:
            sigmoid = libdevice.fast_dividef(1.0, 1.0 + tl.exp(-gate))
        else:
            sigmoid = tl.sigmoid(gate)
        acc = tl.fma(sigmoid, x, acc)
    acc /= HC
    tl.store(Y + rows[:, None] * H + columns[None, :], acc, mask)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor) and x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return _generic_hc_gate_mix(x, gate, hc_count)
    output, spec = _prepare_hc_gate_mix(x, gate, hc_count)
    if output.numel() == 0:
        return output
    with runtime.torch_device_fn.device(x.device):
        _hc_gate_mix_hygon_kernel[
            lambda meta: (
                triton.cdiv(spec[1], meta["BLOCK_H"]),
                triton.cdiv(spec[0], meta["BLOCK_M"]),
            )
        ](x, gate, output, spec)
    return output


__all__ = ["hc_gate_mix"]
