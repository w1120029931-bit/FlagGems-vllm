# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_combine_norm import _prepare_hc_combine_norm
from flaggems_vllm.utils import libentry, libtuner


def _prune_hc_combine_norm_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if (
        args["EPS"] >= 0
        and args["out_ptr"].dtype == torch.bfloat16
        and args["HC_DIM"] in (2048, 2560)
    ):
        return configs
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_H"] == 512 and config.num_warps == 4
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_combine_norm_metax"),
    prune_configs_by={"early_config_prune": _prune_hc_combine_norm_configs},
    key=["SPEC", "EPS"],
    strategy=[repr, lambda eps: eps < 0],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_combine_norm_metax_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SPEC: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_H)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    pid = tl.program_id(0)
    row = pid // HC
    stream = pid % HC
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    w_offs = offs_inner if W_SHARED else offs

    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(
        block_ptr + row * stride_block + offs_inner,
        mask_inner,
        other=0.0,
    )
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if inj_ptr is None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    if inj_ptr is not None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offs, y, mask_inner)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, normalized, spec = _prepare_hc_combine_norm(
        residual, block_output, injection_logits, norm_weight, hc_count
    )
    if spec[0] == 0:
        return output, normalized
    with runtime.torch_device_fn.device(residual.device):
        _hc_combine_norm_metax_kernel[(spec[0] * spec[2],)](
            block_output,
            residual,
            injection_logits,
            norm_weight,
            output,
            normalized,
            block_output.stride(0),
            residual.stride(0),
            spec[5],
            output.stride(0),
            normalized.stride(0),
            HC_DIM=spec[1],
            HC=spec[2],
            W_SHARED=spec[6],
            EPS=eps,
            SPEC=spec,
        )
    return output, normalized


__all__ = ["hc_combine_norm"]
