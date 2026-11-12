# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""HC-scaled SiLU with the vLLM FP32 arithmetic contract."""

from numbers import Integral

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

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


def _prepare_hc_silu(x, hc_count):
    if not isinstance(x, torch.Tensor):
        raise TypeError("hc_silu requires a tensor")
    if x.layout != torch.strided:
        raise NotImplementedError("hc_silu requires strided tensors")
    if x.ndim != 2 or x.stride(1) != 1:
        raise ValueError("hc_silu requires a 2-D tensor with unit inner stride")
    if not isinstance(hc_count, Integral):
        raise TypeError("hc_count must be an integer")
    hc_count = int(hc_count)
    if not -(1 << 63) <= hc_count < (1 << 63):
        raise ValueError("hc_count must fit the native int64 argument")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError("hc_silu dtype is unsupported by native vLLM")
    if x.shape[1] == 0:
        raise ValueError("hc_silu requires a nonzero input width")
    if x.device.type in ("cpu", "meta"):
        raise NotImplementedError("hc_silu requires accelerator tensors")
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return output, (x.shape[0], x.shape[1], x.stride(0)), hc_count


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_silu"),
    key=["SPEC"],
    strategy=[repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_silu_kernel(
    X, Y, SPEC: tl.constexpr, HC: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * SPEC[2] + col, col < SPEC[1], other=0.0)
    scaled = x.to(tl.float32) / HC
    y = scaled * tl.sigmoid(scaled)
    tl.store(Y + row * SPEC[1] + col, y, col < SPEC[1])


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Return SiLU(x / hc_count), computed in FP32 and cast once to x.dtype."""
    output, spec, hc_count = _prepare_hc_silu(x, hc_count)
    if spec[0] == 0:
        return output
    grid = lambda meta: (spec[0], triton.cdiv(spec[1], meta["BLOCK_SIZE"]))
    with runtime.torch_device_fn.device(x.device):
        _hc_silu_kernel[grid](x, output, spec, hc_count)
    return output


__all__ = ["hc_silu"]
