# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MUSA HC SiLU with tiles spanning token rows."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_silu import _prepare_hc_silu
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_silu_mthreads"),
    key=["SPEC"],
    strategy=[repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_silu_mthreads_kernel(
    X, Y, SPEC: tl.constexpr, HC: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    indices = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < SPEC[0] * SPEC[1]
    if SPEC[2] == SPEC[1]:
        input_offsets = indices
    else:
        input_offsets = indices // SPEC[1] * SPEC[2] + indices % SPEC[1]
    x = tl.load(X + input_offsets, mask, other=0.0).to(tl.float32) / HC
    y = x * tl.sigmoid(x)
    tl.store(Y + indices, y, mask)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Preserve native arithmetic for contiguous and gapped input rows."""
    output, spec, hc_count = _prepare_hc_silu(x, hc_count)
    if spec[0] == 0:
        return output
    grid = lambda meta: (triton.cdiv(spec[0] * spec[1], meta["BLOCK_SIZE"]),)
    with runtime.torch_device_fn.device(x.device):
        _hc_silu_mthreads_kernel[grid](x, output, spec, hc_count)
    return output


__all__ = ["hc_silu"]
