# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_gate_mix import _prepare_hc_gate_mix
from flaggems_vllm.utils import libentry, libtuner

from ._hc_utils import _load_f32, _store_f32


def _prune_hc_gate_mix_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    narrow = args["X_DTYPE"] is not None or args["G_DTYPE"] is not None
    block_m = 1 if narrow else 8
    return [config for config in configs if config.kwargs["BLOCK_M"] == block_m]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_gate_mix_ascend"),
    prune_configs_by={"early_config_prune": _prune_hc_gate_mix_configs},
    key=["SPEC", "X_DTYPE", "G_DTYPE"],
    strategy=[repr, repr, repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_gate_mix_ascend_kernel(
    X,
    G,
    Y,
    SPEC: tl.constexpr,
    X_DTYPE: tl.constexpr,
    G_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    N: tl.constexpr = SPEC[0]
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SX: tl.constexpr = SPEC[3]
    SG: tl.constexpr = SPEC[4]
    COLUMN_TILES: tl.constexpr = triton.cdiv(H, BLOCK_H)
    TILES: tl.constexpr = triton.cdiv(N, BLOCK_M) * COLUMN_TILES
    pid = tl.program_id(0).to(tl.int64)
    for tile in range(pid, TILES, tl.num_programs(0)):
        rows = tile // COLUMN_TILES * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = tile % COLUMN_TILES * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (rows[:, None] < N) & (columns[None, :] < H)
        acc = tl.full((BLOCK_M, BLOCK_H), 0.0, tl.float32)
        for branch in tl.static_range(HC):
            offsets = branch * H + columns[None, :]
            gate = _load_f32(G, rows[:, None] * SG + offsets, mask, G_DTYPE)
            values = _load_f32(X, rows[:, None] * SX + offsets, mask, X_DTYPE)
            acc = tl.fma(tl.sigmoid(gate), values, acc)
        _store_f32(Y, rows[:, None] * H + columns[None, :], acc / HC, mask, X_DTYPE)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    output, spec = _prepare_hc_gate_mix(x, gate, hc_count)
    if (
        any(
            tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            for tensor in (x, gate)
        )
        and "910B" in runtime.torch_device_fn.get_device_name(x.device).upper()
    ):
        raise NotImplementedError("hc_gate_mix does not support FP8 on Ascend 910B")
    if output.numel() == 0:
        return output
    view_dtypes = {torch.float64: tl.float64, torch.uint64: tl.uint64}
    x_dtype = view_dtypes.get(x.dtype)
    gate_dtype = view_dtypes.get(gate.dtype)
    with runtime.torch_device_fn.device(x.device):
        cores = runtime.torch_device_fn.get_device_properties(x.device).vector_core_num
        grid = lambda meta: (
            min(
                cores,
                triton.cdiv(spec[0], meta["BLOCK_M"])
                * triton.cdiv(spec[1], meta["BLOCK_H"]),
            ),
        )
        _hc_gate_mix_ascend_kernel[grid](
            x.view(torch.int32) if x_dtype is not None else x,
            gate.view(torch.int32) if gate_dtype is not None else gate,
            output.view(torch.int32) if x_dtype is not None else output,
            spec,
            X_DTYPE=x_dtype,
            G_DTYPE=gate_dtype,
        )
    return output


__all__ = ["hc_gate_mix"]
