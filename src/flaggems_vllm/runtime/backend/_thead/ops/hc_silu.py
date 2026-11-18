# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_silu import _prepare_hc_silu
from flaggems_vllm.ops.hc_silu import hc_silu as _generic_hc_silu
from flaggems_vllm.runtime.backend._thead.ops._hc_utils import _hc_cast
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_silu_thead"),
    key=["SPEC"],
    strategy=[repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_silu_thead_kernel(
    X, Y, SPEC: tl.constexpr, HC: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + row * SPEC[2] + col, col < SPEC[1], other=0.0)
    scaled = x.to(tl.float32) / HC
    y = scaled * tl.sigmoid(scaled)
    if Y.dtype.element_ty == tl.float16:
        # Some backends fuse FP16 conversion with +0 and lose a negative zero.
        rounded = y.to(tl.float16)
        bits = rounded.to(tl.uint16, bitcast=True)
        sign = ((y.to(tl.uint32, bitcast=True) >> 16) & 0x8000).to(tl.uint16)
        y = tl.where((bits & 0x7FFF) == 0, sign, bits).to(tl.float16, bitcast=True)
    tl.store(Y + row * SPEC[1] + col, _hc_cast(y, Y.dtype.element_ty), col < SPEC[1])


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor) and x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return _generic_hc_silu(x, hc_count)
    output, spec, hc_count = _prepare_hc_silu(x, hc_count)
    if spec[0] == 0:
        return output
    with runtime.torch_device_fn.device(x.device):
        _hc_silu_thead_kernel[
            lambda meta: (spec[0], triton.cdiv(spec[1], meta["BLOCK_SIZE"]))
        ](x, output, spec, hc_count)
    return output


__all__ = ["hc_silu"]
