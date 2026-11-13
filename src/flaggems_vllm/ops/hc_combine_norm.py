# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical HC combine and grouped RMSNorm with a portable Triton kernel."""

from math import gcd
from numbers import Integral

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

# The numeric dtypes verified against the pinned vLLM 0.32 native operator.
# Each input has its own dtype; all arithmetic is FP32, including FP64 inputs.
_SUPPORTED_DTYPES = frozenset(
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.uint16,
        torch.int32,
        torch.uint32,
        torch.int64,
        torch.uint64,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
)


def _prepare_hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    hc_count: int,
):
    """Validate metadata and allocate outputs for generic and backend kernels.

    Return (combined, normalized, SPEC) where SPEC contains
    (N, H, HC, stride_residual, stride_block, stride_logits, weight_shared, has_logits).
    Epsilon is a separate kernel argument and has no positivity restriction.
    """
    tensors = (residual, block_output, norm_weight)
    matrices = (residual, block_output)
    if injection_logits is not None:
        tensors += (injection_logits,)
        matrices += (injection_logits,)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError(
            "hc_combine_norm requires tensors and optional injection logits"
        )
    if any(tensor.layout != torch.strided for tensor in tensors):
        raise NotImplementedError("hc_combine_norm requires strided tensors")
    if any(tensor.ndim != 2 for tensor in matrices):
        raise ValueError("residual, block_output and injection_logits must be 2-D")
    if not isinstance(hc_count, Integral):
        raise TypeError("hc_count must be an integer")
    hc_count = int(hc_count)
    if hc_count <= 0:
        raise ValueError("hc_count must be positive")
    n, width = residual.shape
    if width == 0 or width % hc_count:
        raise ValueError("residual width must be positive and divisible by hc_count")
    hidden = width // hc_count
    if block_output.shape != (n, hidden):
        raise ValueError("block_output must have shape (N, H)")
    if injection_logits is not None and injection_logits.shape != (n, hc_count):
        raise ValueError("injection_logits must have shape (N, hc_count)")
    if any(tensor.stride(1) != 1 for tensor in matrices):
        raise ValueError("residual, block_output and logits require unit inner stride")
    if not norm_weight.is_contiguous() or norm_weight.numel() not in (hidden, width):
        raise ValueError(
            "norm_weight must be contiguous with H or hc_count * H elements"
        )
    if any(tensor.dtype not in _SUPPORTED_DTYPES for tensor in tensors):
        raise NotImplementedError(
            "hc_combine_norm input dtype is unsupported by native vLLM"
        )
    device = residual.device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError("hc_combine_norm inputs must be on the same device")
    if device.type in ("cpu", "meta"):
        raise NotImplementedError("hc_combine_norm requires accelerator tensors")
    output = torch.empty((n, width), dtype=residual.dtype, device=device)
    normalized = torch.empty((n, width), dtype=residual.dtype, device=device)
    spec = (
        n,
        hidden,
        hc_count,
        residual.stride(0),
        block_output.stride(0),
        injection_logits.stride(0) if injection_logits is not None else 0,
        norm_weight.numel() == hidden,
        injection_logits is not None,
    )
    return output, normalized, spec


def _prune_hc_combine_norm_configs(configs, named_args, **kwargs):
    """Keep native reduction order before integer/bool output conversion."""
    if named_args["O"].dtype.is_floating_point:
        return configs
    return [
        config
        for config in configs
        if config.kwargs.get("BLOCK_H") == 512 and config.num_warps == 4
    ]


def _native_reduction_vector(residual, block_output, norm_weight, hidden):
    """Match native JIT alignment: runtime strides/pointers specialize at 16 only."""
    output_alignment = 16 if residual.shape[1] % 16 == 0 else 1
    widths = [gcd(hidden, 16 // residual.element_size(), output_alignment)]
    for tensor, branch_stride, row_stride in (
        (residual, hidden, residual.stride(0)),
        (block_output, 0, block_output.stride(0)),
        (norm_weight, 0 if norm_weight.numel() == hidden else hidden, 0),
    ):
        vector = 16 // tensor.element_size() if tensor.data_ptr() % 16 == 0 else 1
        row_alignment = 16 if row_stride % 16 == 0 else 1
        widths.append(gcd(vector, branch_stride, row_alignment))
    return min(max(widths), 4 * triton.next_power_of_2(triton.cdiv(hidden, 512)))


@triton.jit
def _native_square_sum(rounded, H: tl.constexpr, VECTOR: tl.constexpr):
    # Preserve native FP32 rounding when negative epsilon cancels the mean square.
    TILES: tl.constexpr = triton.next_power_of_2(triton.cdiv(H, 512))
    LANES: tl.constexpr = min(128, 512 // VECTOR)
    LOCAL: tl.constexpr = 512 // LANES
    WARPS: tl.constexpr = LANES // 32
    GROUPS: tl.constexpr = max(1, VECTOR // 4)
    lanes = tl.arange(0, LANES)
    first = lanes * VECTOR + (1 % VECTOR) + (1 // VECTOR) * LANES * VECTOR
    x1 = tl.gather(rounded, tl.broadcast_to(first[None, :], (TILES, LANES)), 1)
    sums = x1 * x1
    x0 = tl.gather(
        rounded, tl.broadcast_to((lanes * VECTOR)[None, :], (TILES, LANES)), 1
    )
    sums = tl.fma(x0, x0, sums)
    for index in tl.static_range(2, LOCAL):
        columns = lanes * VECTOR + index % VECTOR + (index // VECTOR) * LANES * VECTOR
        value = tl.gather(rounded, tl.broadcast_to(columns[None, :], (TILES, LANES)), 1)
        sums = tl.fma(value, value, sums)
    sums = tl.reshape(sums, (TILES, WARPS, 32))
    for step in tl.static_range(5):
        halves = tl.trans(tl.reshape(sums, (TILES, WARPS, 2, 16 >> step)), (0, 1, 3, 2))
        low, high = tl.split(halves)
        sums = low + high
    sums = tl.reshape(sums, (TILES, WARPS))
    for step in tl.static_range((2 if WARPS == 4 else 1 if WARPS == 2 else 0)):
        halves = tl.trans(tl.reshape(sums, (TILES, 2, WARPS >> (step + 1))), (0, 2, 1))
        low, high = tl.split(halves)
        sums = low + high
    sums = tl.reshape(sums, (TILES,))
    groups = tl.arange(0, GROUPS)
    total = tl.gather(sums, groups, axis=0)
    for tile in tl.static_range(GROUPS, TILES, GROUPS):
        total = total + tl.gather(sums, groups + tile, axis=0)
    for step in tl.static_range((2 if GROUPS == 4 else 1 if GROUPS == 2 else 0)):
        halves = tl.trans(tl.reshape(total, (2, GROUPS >> (step + 1))))
        low, high = tl.split(halves)
        total = low + high
    return tl.sum(total, axis=0)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_combine_norm"),
    prune_configs_by={"early_config_prune": _prune_hc_combine_norm_configs},
    key=["SPEC", "EPS", "NATIVE_VEC"],
    strategy=[repr, lambda eps: eps < 0, repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_combine_norm_kernel(
    R,
    B,
    I,
    W,
    O,
    Y,
    SPEC: tl.constexpr,
    EPS: tl.constexpr,
    NATIVE_VEC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SR: tl.constexpr = SPEC[3]
    SB: tl.constexpr = SPEC[4]
    SI: tl.constexpr = SPEC[5]
    SHARED: tl.constexpr = SPEC[6]
    HAS_LOGITS: tl.constexpr = SPEC[7]
    if EPS < 0:
        TILE_H: tl.constexpr = 512
        TILES: tl.constexpr = triton.next_power_of_2(triton.cdiv(H, 512))
    elif O.dtype.element_ty.is_floating():
        TILE_H: tl.constexpr = BLOCK_H
        TILES: tl.constexpr = triton.next_power_of_2(triton.cdiv(H, TILE_H))
    else:
        tl.static_assert(
            BLOCK_H == 512, "Integer outputs require native reduction tile"
        )
        # NaN injection may cast masked combine lanes to nonzero integer/bool
        # values. Native includes those lanes in RMS, so preserve its padded
        # extent even when autotuning a different tile decomposition.
        NATIVE_EXTENT: tl.constexpr = 512 * triton.next_power_of_2(triton.cdiv(H, 512))
        TILE_H: tl.constexpr = min(BLOCK_H, NATIVE_EXTENT)
        TILES: tl.constexpr = NATIVE_EXTENT // TILE_H

    pid = tl.program_id(0).to(tl.int64)
    row = pid // HC
    branch = pid % HC
    columns = tl.arange(0, TILES)[:, None] * TILE_H + tl.arange(0, TILE_H)[None, :]
    mask = columns < H
    offsets = branch * H + columns
    residual = tl.load(R + row * SR + offsets, mask, other=0.0).to(tl.float32)
    block = tl.load(B + row * SB + columns, mask, other=0.0).to(tl.float32)
    if HAS_LOGITS:
        logit = tl.load(I + row * SI + branch).to(tl.float32)
        gate = 2.0 * tl.sigmoid(logit / HC)
        block = block * gate
    combined = (residual + block).to(O.dtype.element_ty)
    weight_offsets = columns if SHARED else offsets
    if not HAS_LOGITS:
        weight = tl.load(W + weight_offsets, mask, other=0.0).to(tl.float32)
    tl.store(O + row * HC * H + offsets, combined, mask)

    # The materialized residual dtype is a mandatory rounding boundary.
    rounded = combined.to(tl.float32)
    if EPS < 0:
        sum_sq = _native_square_sum(rounded, H, NATIVE_VEC)
    else:
        sum_sq = tl.sum(tl.sum(rounded * rounded, axis=1), axis=0)
    inverse_rms = tl.rsqrt(tl.fma(sum_sq, 1.0 / H, EPS))
    if HAS_LOGITS:
        weight = tl.load(W + weight_offsets, mask, other=0.0).to(tl.float32)
    normalized = rounded * inverse_rms
    normalized = tl.fma(normalized, weight, normalized)
    tl.store(Y + row * HC * H + offsets, normalized, mask)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine HC branches and normalize the rounded result independently.

    Residual, block output and optional logits are 2-D with unit inner strides
    and arbitrary row strides. None logits apply the block with unit weight.
    Norm weights are contiguous, shared or per-branch, and may have any rank.
    Numeric input dtypes may differ. Both fresh outputs
    have residual's shape, dtype and device. Epsilon may be negative, zero or nonfinite.
    This is a forward inference operation; no backward is provided.
    """
    output, normalized, spec = _prepare_hc_combine_norm(
        residual, block_output, injection_logits, norm_weight, hc_count
    )
    if spec[0] == 0:
        return output, normalized
    native_vector = (
        _native_reduction_vector(residual, block_output, norm_weight, spec[1])
        if eps < 0
        else 0
    )
    with runtime.torch_device_fn.device(residual.device):
        _hc_combine_norm_kernel[(spec[0] * spec[2],)](
            residual,
            block_output,
            injection_logits,
            norm_weight,
            output,
            normalized,
            SPEC=spec,
            EPS=eps,
            NATIVE_VEC=native_vector,
        )
    return output, normalized


__all__ = ["hc_combine_norm"]
