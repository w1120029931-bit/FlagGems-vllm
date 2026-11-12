# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Canonical HC gated mixing with independent input dtypes and row strides."""

import logging
from numbers import Integral

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = frozenset(
    getattr(torch, name)
    for name in (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "bool",
        "float8_e4m3fn",
        "float8_e5m2",
    )
    if hasattr(torch, name)
)


def _prepare_hc_gate_mix(x, gate, hc_count):
    if not isinstance(x, torch.Tensor) or not isinstance(gate, torch.Tensor):
        raise TypeError("hc_gate_mix requires two tensor inputs")
    if x.layout != torch.strided or gate.layout != torch.strided:
        raise NotImplementedError("hc_gate_mix requires strided tensors")
    if x.ndim != 2 or gate.ndim != 2 or x.shape != gate.shape:
        raise ValueError("x and gate must have the same 2-D shape")
    if not isinstance(hc_count, Integral):
        raise TypeError("hc_count must be an integer")
    hc_count = int(hc_count)
    if not -(1 << 63) <= hc_count < (1 << 63):
        raise ValueError("hc_count must fit the native signed-int64 schema")
    width = x.shape[1]
    if hc_count == 0 or width % hc_count or (hc_count < 0 and width != 0):
        raise ValueError("hc_count must divide width and produce a nonnegative output")
    if x.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("x and gate require unit inner strides")
    if x.device != gate.device:
        raise ValueError("x and gate must be on the same device")
    if x.device.type in ("cpu", "meta"):
        raise NotImplementedError("hc_gate_mix requires accelerator tensors")
    if x.dtype not in _SUPPORTED_DTYPES or gate.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError("hc_gate_mix input dtype is unsupported")
    n, width = x.shape
    h = width // hc_count
    output = torch.empty((n, h), dtype=x.dtype, device=x.device)
    return output, (n, h, hc_count, x.stride(0), gate.stride(0))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_gate_mix"),
    key=["SPEC"],
    strategy=repr,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_gate_mix_kernel(X, G, Y, SPEC: tl.constexpr, BLOCK_H: tl.constexpr):
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SX: tl.constexpr = SPEC[3]
    SG: tl.constexpr = SPEC[4]
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = columns < H
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * H + columns
        gate = tl.load(G + row * SG + offsets, mask, other=0.0).to(tl.float32)
        x = tl.load(X + row * SX + offsets, mask, other=0.0).to(tl.float32)
        acc = tl.fma(tl.sigmoid(gate), x, acc)
    acc /= HC
    tl.store(Y + row * H + columns, acc, mask)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Return the FP32 gated HC mean cast to x.dtype, as a fresh (N, H) tensor."""
    logger.debug("GEMS HC_GATE_MIX")
    output, spec = _prepare_hc_gate_mix(x, gate, hc_count)
    if output.numel() == 0:
        return output
    with runtime.torch_device_fn.device(x.device):
        _hc_gate_mix_kernel[
            lambda meta: (spec[0], triton.cdiv(spec[1], meta["BLOCK_H"]))
        ](x, gate, output, spec)
    return output


__all__ = ["hc_gate_mix"]
