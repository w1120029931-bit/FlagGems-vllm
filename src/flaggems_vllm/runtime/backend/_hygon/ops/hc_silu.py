# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Hygon HC-scaled SiLU with contiguous output tiles spanning token rows."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_silu import _prepare_hc_silu
from flaggems_vllm.ops.hc_silu import hc_silu as _generic_hc_silu
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_silu_hygon"),
    key=["SPEC"],
    strategy=[repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_silu_hygon_kernel(
    X, Y, SPEC: tl.constexpr, HC: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    indices = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < SPEC[0] * SPEC[1]
    if SPEC[2] == SPEC[1]:
        offsets = indices
    else:
        offsets = indices // SPEC[1] * SPEC[2] + indices % SPEC[1]
    scaled = tl.load(X + offsets, mask, other=0.0).to(tl.float32) / HC
    if X.dtype.element_ty == tl.bfloat16:
        sigmoid = libdevice.fast_dividef(1.0, 1.0 + tl.exp(-scaled))
    else:
        sigmoid = tl.sigmoid(scaled)
    value32 = scaled * sigmoid
    value = value32.to(Y.dtype.element_ty)
    if Y.dtype.element_ty == tl.float16:
        # HCU mixed-FMA conversion can lose the sign of an FP16 zero.
        bits = value.to(tl.uint16, bitcast=True)
        sign = ((value32.to(tl.uint32, bitcast=True) >> 16) & 0x8000).to(tl.uint16)
        value = tl.where((bits & 0x7FFF) == 0, sign, bits).to(tl.float16, bitcast=True)
    tl.store(Y + indices, value, mask)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor) and x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return _generic_hc_silu(x, hc_count)
    output, spec, hc_count = _prepare_hc_silu(x, hc_count)
    if spec[0] == 0:
        return output
    with runtime.torch_device_fn.device(x.device):
        _hc_silu_hygon_kernel[
            lambda meta: (triton.cdiv(spec[0] * spec[1], meta["BLOCK_SIZE"]),)
        ](x, output, spec, hc_count)
    return output


__all__ = ["hc_silu"]
