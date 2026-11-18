# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_gate_mix import _prepare_hc_gate_mix
from flaggems_vllm.ops.hc_gate_mix import hc_gate_mix as _generic_hc_gate_mix
from flaggems_vllm.runtime.backend._thead.ops._hc_utils import _hc_cast
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_gate_mix_thead"),
    key=["SPEC"],
    strategy=repr,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_gate_mix_thead_kernel(X, G, Y, SPEC: tl.constexpr, BLOCK_H: tl.constexpr):
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
    tl.store(Y + row * H + columns, _hc_cast(acc, Y.dtype.element_ty), mask)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor) and x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return _generic_hc_gate_mix(x, gate, hc_count)
    output, spec = _prepare_hc_gate_mix(x, gate, hc_count)
    if output.numel() == 0:
        return output
    with runtime.torch_device_fn.device(x.device):
        _hc_gate_mix_thead_kernel[
            lambda meta: (
                spec[0],
                triton.cdiv(spec[1], meta["BLOCK_H"]),
            )
        ](x, gate, output, spec)
    return output


__all__ = ["hc_gate_mix"]
