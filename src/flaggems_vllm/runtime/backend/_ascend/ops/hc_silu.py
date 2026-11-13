# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_silu import _prepare_hc_silu
from flaggems_vllm.utils import libentry, libtuner

from ._hc_utils import _load_f32, _store_f32


def _prune_hc_silu_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    block_size = 256 if args["DTYPE"] is not None else 4096
    return [config for config in configs if config.kwargs["BLOCK_SIZE"] == block_size]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_silu_ascend"),
    prune_configs_by={"early_config_prune": _prune_hc_silu_configs},
    key=["SPEC", "DTYPE"],
    strategy=[repr, repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_silu_ascend_kernel(
    X,
    Y,
    SPEC: tl.constexpr,
    HC: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    N: tl.constexpr = SPEC[0]
    H: tl.constexpr = SPEC[1]
    SX: tl.constexpr = SPEC[2]
    CONTIGUOUS: tl.constexpr = SX == H
    BLOCK_H: tl.constexpr = BLOCK_SIZE // BLOCK_M
    COLUMN_TILES: tl.constexpr = triton.cdiv(H, BLOCK_H)
    if CONTIGUOUS:
        TILES: tl.constexpr = triton.cdiv(N * H, BLOCK_SIZE)
    else:
        TILES: tl.constexpr = triton.cdiv(N, BLOCK_M) * COLUMN_TILES
    pid = tl.program_id(0).to(tl.int64)
    for tile in range(pid, TILES, tl.num_programs(0)):
        if CONTIGUOUS:
            output_offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            input_offsets = output_offsets
            mask = output_offsets < N * H
        else:
            rows = tile // COLUMN_TILES * BLOCK_M + tl.arange(0, BLOCK_M)
            columns = tile % COLUMN_TILES * BLOCK_H + tl.arange(0, BLOCK_H)
            input_offsets = rows[:, None] * SX + columns[None, :]
            output_offsets = rows[:, None] * H + columns[None, :]
            mask = (rows[:, None] < N) & (columns[None, :] < H)
        x = _load_f32(X, input_offsets, mask, DTYPE)
        scaled = x / HC
        y = scaled * tl.sigmoid(scaled)
        if Y.dtype.element_ty == tl.float16:
            # Preserve the FP32 sign when FP16 conversion rounds to zero.
            rounded = y.to(tl.float16)
            bits = rounded.to(tl.uint16, bitcast=True)
            sign = ((y.to(tl.uint32, bitcast=True) >> 16) & 0x8000).to(tl.uint16)
            y = tl.where((bits & 0x7FFF) == 0, sign, bits).to(tl.float16, bitcast=True)
        _store_f32(Y, output_offsets, y, mask, DTYPE)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    output, spec, hc_count = _prepare_hc_silu(x, hc_count)
    if (
        x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and "910B" in runtime.torch_device_fn.get_device_name(x.device).upper()
    ):
        raise NotImplementedError("hc_silu does not support FP8 on Ascend 910B")
    if spec[0] == 0:
        return output
    dtype = {torch.float64: tl.float64, torch.uint64: tl.uint64}.get(x.dtype)
    with runtime.torch_device_fn.device(x.device):
        cores = runtime.torch_device_fn.get_device_properties(x.device).vector_core_num
        if spec[1] == spec[2]:
            grid = lambda meta: (
                min(cores, triton.cdiv(spec[0] * spec[1], meta["BLOCK_SIZE"])),
            )
        else:
            grid = lambda meta: (
                min(
                    cores,
                    triton.cdiv(spec[0], meta["BLOCK_M"])
                    * triton.cdiv(spec[1], meta["BLOCK_SIZE"] // meta["BLOCK_M"]),
                ),
            )
        _hc_silu_ascend_kernel[grid](
            x.view(torch.int32) if dtype is not None else x,
            output.view(torch.int32) if dtype is not None else output,
            spec,
            hc_count,
            DTYPE=dtype,
        )
    return output


__all__ = ["hc_silu"]
