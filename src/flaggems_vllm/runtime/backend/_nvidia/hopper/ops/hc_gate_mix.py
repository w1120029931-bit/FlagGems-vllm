# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Hopper HC gated mixing with native stream order and PDL."""

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
    configs=runtime.get_tuned_config("hc_gate_mix_hopper"),
    key=["SPEC"],
    strategy=repr,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_gate_mix_hopper_kernel(
    X, G, Y, SPEC: tl.constexpr, BLOCK_H: tl.constexpr, launch_pdl: tl.constexpr
):
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SX: tl.constexpr = SPEC[3]
    SG: tl.constexpr = SPEC[4]
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = columns < H
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * H + columns
        gate = tl.load(G + row * SG + offsets, mask, other=0.0).to(tl.float32)
        x = tl.load(X + row * SX + offsets, mask, other=0.0).to(tl.float32)
        acc += tl.sigmoid(gate) * x
    acc /= HC
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(Y + row * H + columns, acc, mask)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Apply the native gated HC mean using the Hopper launch path."""
    logger.debug("GEMS HC_GATE_MIX HOPPER")
    output, spec = _prepare_hc_gate_mix(x, gate, hc_count)
    if output.numel() == 0:
        return output
    with runtime.torch_device_fn.device(x.device):
        _hc_gate_mix_hopper_kernel[
            lambda meta: (spec[0], triton.cdiv(spec[1], meta["BLOCK_H"]))
        ](x, gate, output, spec, launch_pdl=True)
    return output


__all__ = ["hc_gate_mix"]
