# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Moore Threads HC gated mixing with row tiling and native accumulation order."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_gate_mix import _prepare_hc_gate_mix
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_gate_mix_mthreads"),
    key=["SPEC"],
    strategy=repr,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_gate_mix_mthreads_kernel(
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
        weight = tl.sigmoid(gate)
        acc = tl.fma(weight, x, acc)
    acc /= HC
    tl.store(Y + rows[:, None] * H + columns[None, :], acc, mask)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS HC_GATE_MIX")
    output, spec = _prepare_hc_gate_mix(x, gate, hc_count)
    if output.numel() == 0:
        return output
    with runtime.torch_device_fn.device(x.device):
        _hc_gate_mix_mthreads_kernel[
            lambda meta: (
                triton.cdiv(spec[1], meta["BLOCK_H"]),
                triton.cdiv(spec[0], meta["BLOCK_M"]),
            )
        ](x, gate, output, spec)
    return output


__all__ = ["hc_gate_mix"]
