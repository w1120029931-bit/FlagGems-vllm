# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_silu import _prepare_hc_silu
from flaggems_vllm.ops.hc_silu import hc_silu as _generic_hc_silu
from flaggems_vllm.utils import libentry, libtuner


def _prune_hc_silu_configs(configs, named_args, **kwargs):
    # Match native scalar versus packed arithmetic, including signed zeros.
    width = named_args["SPEC"][1]
    scalar = width <= 256
    configs = [
        config
        for config in configs
        if (config.kwargs["BLOCK_SIZE"] <= config.num_warps * 64) == scalar
    ]
    if named_args["SPEC"][2] != width:
        blocks = [config.kwargs["BLOCK_SIZE"] for config in configs]
        block = min((size for size in blocks if size >= width), default=max(blocks))
        configs = [config for config in configs if config.kwargs["BLOCK_SIZE"] == block]
    return configs


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_silu_metax"),
    prune_configs_by={"early_config_prune": _prune_hc_silu_configs},
    key=["SPEC"],
    strategy=[repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_silu_metax_kernel(
    X, Y, SPEC: tl.constexpr, HC: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_SIZE)
    if SPEC[2] == SPEC[1]:
        indices = pid * BLOCK_SIZE + columns
        offsets = indices
        mask = indices < SPEC[0] * SPEC[1]
    else:
        tiles: tl.constexpr = triton.cdiv(SPEC[1], BLOCK_SIZE)
        row = pid // tiles
        columns += (pid % tiles) * BLOCK_SIZE
        indices = row * SPEC[1] + columns
        offsets = row * SPEC[2] + columns
        mask = columns < SPEC[1]
    scaled = tl.load(X + offsets, mask, other=0.0).to(tl.float32) / HC
    value = scaled * tl.sigmoid(scaled)
    tl.store(Y + indices, value, mask)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if isinstance(x, torch.Tensor) and not x.dtype.is_floating_point:
        return _generic_hc_silu(x, hc_count)
    output, spec, hc_count = _prepare_hc_silu(x, hc_count)
    if spec[0] == 0:
        return output
    with runtime.torch_device_fn.device(x.device):
        _hc_silu_metax_kernel[
            lambda meta: (
                (
                    triton.cdiv(spec[0] * spec[1], meta["BLOCK_SIZE"])
                    if spec[2] == spec[1]
                    else spec[0] * triton.cdiv(spec[1], meta["BLOCK_SIZE"])
                ),
            )
        ](x, output, spec, hc_count)
    return output


__all__ = ["hc_silu"]
