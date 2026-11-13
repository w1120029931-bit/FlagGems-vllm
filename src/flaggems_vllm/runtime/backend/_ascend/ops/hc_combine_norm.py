# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hc_combine_norm import (
    _native_reduction_vector,
    _prepare_hc_combine_norm,
)
from flaggems_vllm.runtime.backend._ascend.ops._hc_utils import (
    _load_f32,
    _round_f32,
    _store_f32,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_combine_norm_ascend"),
    key=["SPEC"],
    strategy=[repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_combine_norm_ascend_kernel(
    R,
    B,
    LOGITS,
    W,
    OUT,
    Y,
    SPEC: tl.constexpr,
    EPS: tl.constexpr,
    DTYPES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    N: tl.constexpr = SPEC[0]
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SR: tl.constexpr = SPEC[3]
    SB: tl.constexpr = SPEC[4]
    SI: tl.constexpr = SPEC[5]
    SHARED: tl.constexpr = SPEC[6]
    HAS_LOGITS: tl.constexpr = SPEC[7]
    BLOCK_H: tl.constexpr = triton.next_power_of_2(H)
    columns = tl.arange(0, BLOCK_H)
    rows = tl.arange(0, BLOCK_M)
    if SHARED:
        weight = _load_f32(W, columns, columns < H, DTYPES[3])
    for tile in range(
        tl.program_id(0).to(tl.int64), triton.cdiv(N * HC, BLOCK_M), tl.num_programs(0)
    ):
        group = (tile * BLOCK_M + rows).to(tl.int64)
        row, branch = group // HC, group % HC
        mask = (group[:, None] < N * HC) & (columns[None, :] < H)
        residual = _load_f32(
            R,
            row[:, None] * SR + branch[:, None] * H + columns[None, :],
            mask,
            DTYPES[0],
        )
        block = _load_f32(B, row[:, None] * SB + columns[None, :], mask, DTYPES[1])
        if HAS_LOGITS:
            logits = _load_f32(LOGITS, row * SI + branch, group < N * HC, DTYPES[2])
            gate = 2.0 * tl.sigmoid(logits / HC)
            block = block * gate[:, None]
        combined = _round_f32(
            residual + block,
            DTYPES[4] if DTYPES[4] is not None else OUT.dtype.element_ty,
        )
        offsets = group[:, None] * H + columns[None, :]
        _store_f32(OUT, offsets, combined, mask, DTYPES[4])
        rounded = combined
        sum_sq = tl.sum(rounded * rounded, axis=1)
        inverse_rms = tl.rsqrt(tl.fma(sum_sq, 1.0 / H, EPS))
        normalized = rounded * inverse_rms[:, None]
        if SHARED:
            normalized = tl.fma(normalized, weight[None, :], normalized)
        else:
            weight = _load_f32(
                W, branch[:, None] * H + columns[None, :], mask, DTYPES[3]
            )
            normalized = tl.fma(normalized, weight, normalized)
        _store_f32(Y, offsets, normalized, mask, DTYPES[5])


def _prune_hc_combine_norm_bf16_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    block_m = 8 if args["SPEC"][1] == 2048 else 4
    return [config for config in configs if config.kwargs["BLOCK_M"] == block_m]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_combine_norm_bf16_ascend"),
    prune_configs_by={"early_config_prune": _prune_hc_combine_norm_bf16_configs},
    key=["SPEC"],
    strategy=[repr],
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_combine_norm_bf16_kernel(
    R,
    B,
    LOGITS,
    W,
    OUT,
    Y,
    SPEC: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    N: tl.constexpr = SPEC[0]
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SR: tl.constexpr = SPEC[3]
    SB: tl.constexpr = SPEC[4]
    SI: tl.constexpr = SPEC[5]
    SHARED: tl.constexpr = SPEC[6]
    HAS_LOGITS: tl.constexpr = SPEC[7]
    ROWS: tl.constexpr = BLOCK_M // HC
    rows = tl.arange(0, ROWS)
    branches = tl.arange(0, HC)
    columns = tl.arange(0, H)
    if SHARED:
        weight = tl.load(W + columns, columns < H, other=0).to(tl.float32)
    for tile in range(
        tl.program_id(0).to(tl.int64), triton.cdiv(N, ROWS), tl.num_programs(0)
    ):
        row = tile * ROWS + rows
        valid_rows = row < N
        mask = valid_rows[:, None, None] & (columns[None, None, :] < H)
        residual = tl.load(
            R
            + row[:, None, None] * SR
            + branches[None, :, None] * H
            + columns[None, None, :],
            mask,
            other=0,
        ).to(tl.float32)
        block = tl.load(
            B + row[:, None] * SB + columns[None, :],
            valid_rows[:, None] & (columns[None, :] < H),
            other=0,
        ).to(tl.float32)
        if HAS_LOGITS:
            logits = tl.load(
                LOGITS + row[:, None] * SI + branches[None, :],
                valid_rows[:, None],
                other=0,
            ).to(tl.float32)
            gate = 2.0 * tl.sigmoid(logits / HC)
            injected = block[:, None, :] * gate[:, :, None]
        else:
            injected = block[:, None, :]
        combined = (residual + injected).to(tl.bfloat16).to(tl.float32)
        offsets = (row[:, None, None] * HC + branches[None, :, None]) * H + columns[
            None, None, :
        ]
        tl.store(OUT + offsets, combined, mask)
        sum_sq = tl.sum(combined * combined, axis=2)
        inverse_rms = tl.rsqrt(tl.fma(sum_sq, 1.0 / H, EPS))
        normalized = combined * inverse_rms[:, :, None]
        if SHARED:
            normalized = tl.fma(normalized, weight[None, None, :], normalized)
        else:
            weight = tl.load(
                W + branches[:, None] * H + columns[None, :],
                columns[None, :] < H,
                other=0,
            ).to(tl.float32)
            normalized = tl.fma(normalized, weight[None, :, :], normalized)
        tl.store(Y + offsets, normalized, mask)


@triton.jit
def _native_square_sum(rounded, H: tl.constexpr, VECTOR: tl.constexpr):
    TILES: tl.constexpr = triton.next_power_of_2(triton.cdiv(H, 512))
    LANES: tl.constexpr = 512 // VECTOR if 512 // VECTOR < 128 else 128
    LOCAL: tl.constexpr = 512 // LANES
    WARPS: tl.constexpr = LANES // 32
    GROUPS: tl.constexpr = VECTOR // 4 if VECTOR // 4 > 1 else 1
    rounded = tl.reshape(rounded, (TILES * 512,))
    lanes = tl.arange(0, TILES * LANES)
    base = (lanes // LANES) * 512 + (lanes % LANES) * VECTOR
    first = base + (1 % VECTOR) + (1 // VECTOR) * LANES * VECTOR
    x1 = tl.gather(rounded, first, axis=0)
    sums = x1 * x1
    x0 = tl.gather(rounded, base, axis=0)
    sums = tl.fma(x0, x0, sums)
    for index in tl.static_range(2, LOCAL):
        columns = base + index % VECTOR + (index // VECTOR) * LANES * VECTOR
        value = tl.gather(rounded, columns, axis=0)
        sums = tl.fma(value, value, sums)
    for step in tl.static_range(5):
        offsets = tl.arange(0, TILES * WARPS * (16 >> step))
        low_offsets = (offsets // (16 >> step)) * (32 >> step) + offsets % (16 >> step)
        low = tl.gather(sums, low_offsets, axis=0)
        high = tl.gather(sums, low_offsets + (16 >> step), axis=0)
        sums = low + high
    for step in tl.static_range((2 if WARPS == 4 else 1 if WARPS == 2 else 0)):
        offsets = tl.arange(0, TILES * (WARPS >> (step + 1)))
        low_offsets = (offsets // (WARPS >> (step + 1))) * (WARPS >> step) + offsets % (
            WARPS >> (step + 1)
        )
        low = tl.gather(sums, low_offsets, axis=0)
        high = tl.gather(sums, low_offsets + (WARPS >> (step + 1)), axis=0)
        sums = low + high
    groups = tl.arange(0, GROUPS)
    total = tl.gather(sums, groups, axis=0)
    for tile in tl.static_range(GROUPS, TILES, GROUPS):
        total = total + tl.gather(sums, groups + tile, axis=0)
    for step in tl.static_range((2 if GROUPS == 4 else 1 if GROUPS == 2 else 0)):
        offsets = tl.arange(0, GROUPS >> (step + 1))
        low = tl.gather(total, offsets, axis=0)
        high = tl.gather(total, offsets + (GROUPS >> (step + 1)), axis=0)
        total = low + high
    return tl.sum(total, axis=0)


@libentry()
@triton.jit
def _hc_combine_norm_compat_kernel(
    R,
    B,
    LOGITS,
    W,
    OUT,
    Y,
    SPEC: tl.constexpr,
    EPS: tl.constexpr,
    DTYPES: tl.constexpr,
    MATERIALIZED_DTYPE: tl.constexpr,
    NATIVE_VEC: tl.constexpr,
):
    H: tl.constexpr = SPEC[1]
    HC: tl.constexpr = SPEC[2]
    SR: tl.constexpr = SPEC[3]
    SB: tl.constexpr = SPEC[4]
    SI: tl.constexpr = SPEC[5]
    SHARED: tl.constexpr = SPEC[6]
    HAS_LOGITS: tl.constexpr = SPEC[7]
    TILE_H: tl.constexpr = 512
    TILES: tl.constexpr = triton.next_power_of_2(triton.cdiv(H, 512))

    pid = tl.program_id(0).to(tl.int64)
    row = pid // HC
    branch = pid % HC
    columns = tl.arange(0, TILES * TILE_H)
    mask = columns < H
    offsets = branch * H + columns
    residual = _load_f32(R, row * SR + offsets, mask, DTYPES[0])
    block = _load_f32(B, row * SB + columns, mask, DTYPES[1])
    if HAS_LOGITS:
        logit = _load_f32(LOGITS, row * SI + branch, True, DTYPES[2])
        gate = 2.0 * tl.sigmoid(logit / HC)
        block = block * gate
    RESULT_DTYPE: tl.constexpr = (
        MATERIALIZED_DTYPE if MATERIALIZED_DTYPE is not None else OUT.dtype.element_ty
    )
    combined = _round_f32(residual + block, RESULT_DTYPE)
    weight_offsets = columns if SHARED else offsets
    if not HAS_LOGITS:
        weight = _load_f32(W, weight_offsets, mask, DTYPES[3])
    _store_f32(OUT, row * HC * H + offsets, combined, mask, DTYPES[4])

    # The materialized residual dtype is a mandatory rounding boundary.
    rounded = combined
    if EPS < 0:
        sum_sq = _native_square_sum(rounded, H, NATIVE_VEC)
    else:
        squares = tl.reshape(rounded * rounded, (TILES, TILE_H))
        sum_sq = tl.sum(tl.sum(squares, axis=1), axis=0)
    denominator = tl.fma(sum_sq, 1.0 / H, EPS)
    inverse_rms = tl.rsqrt(denominator)
    if RESULT_DTYPE.is_int():
        product = denominator * inverse_rms
        error = tl.fma(-product, inverse_rms, 1.0)
        product_error = tl.fma(denominator, inverse_rms, -product)
        error = tl.fma(-product_error, inverse_rms, error)
        refined = tl.fma(0.5 * inverse_rms, error, inverse_rms)
        finite_positive = (
            (denominator > 0.0)
            & (denominator < float("inf"))
            & (inverse_rms < float("inf"))
        )
        inverse_rms = tl.where(finite_positive, refined, inverse_rms)
    if HAS_LOGITS:
        weight = _load_f32(W, weight_offsets, mask, DTYPES[3])
    normalized = rounded * inverse_rms
    normalized = tl.fma(normalized, weight, normalized)
    _store_f32(Y, row * HC * H + offsets, normalized, mask, DTYPES[5])


@libentry()
@triton.jit
def _hc_convert_64bit_kernel(
    SRC,
    DST,
    COUNT: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    SRC_DTYPE: tl.constexpr,
    DST_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < COUNT
    source_offsets = offsets // WIDTH * STRIDE + offsets % WIDTH
    value = _load_f32(SRC, source_offsets, mask, SRC_DTYPE)
    _store_f32(DST, offsets, value, mask, DST_DTYPE)


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
    if (
        any(
            tensor is not None
            and tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            for tensor in (residual, block_output, injection_logits, norm_weight)
        )
        and "910B" in runtime.torch_device_fn.get_device_name(residual.device).upper()
    ):
        raise NotImplementedError("hc_combine_norm does not support FP8 on Ascend 910B")
    if spec[0] == 0:
        return output, normalized
    tensors = (
        residual,
        block_output,
        injection_logits,
        norm_weight,
        output,
        normalized,
    )
    dtypes = tuple(
        (
            tl.float64
            if tensor is not None and tensor.dtype == torch.float64
            else (
                tl.uint64
                if tensor is not None and tensor.dtype == torch.uint64
                else None
            )
        )
        for tensor in tensors
    )
    args = tuple(
        (
            (tensor.as_strided((tensor.numel(),), (1,)) if index == 3 else tensor).view(
                torch.int32
            )
            if dtype is not None
            else tensor
        )
        for index, (tensor, dtype) in enumerate(zip(tensors, dtypes))
    )
    with runtime.torch_device_fn.device(residual.device):
        if (
            eps < 0
            or not residual.is_floating_point()
            or spec[1] > 4096
            or tl.float64 in dtypes
            or tl.uint64 in dtypes
        ):
            native_vector = (
                _native_reduction_vector(residual, block_output, norm_weight, spec[1])
                if eps < 0
                else 0
            )
            staged_args = list(args)
            staged_dtypes = list(dtypes)
            staged_spec = list(spec)
            if tl.float64 in dtypes or tl.uint64 in dtypes:
                # Separate 64-bit conversion avoids CANN SimplifyOps failure here.
                for index, dtype in enumerate(dtypes):
                    if dtype not in (tl.float64, tl.uint64):
                        continue
                    tensor = tensors[index]
                    buffer = torch.empty(
                        tensor.shape, dtype=torch.float32, device=tensor.device
                    )
                    staged_args[index] = buffer
                    staged_dtypes[index] = None
                    if index < 4:
                        width = tensor.shape[1] if index < 3 else tensor.numel()
                        stride = spec[3 + index] if index < 3 else width
                        _hc_convert_64bit_kernel[(triton.cdiv(tensor.numel(), 256),)](
                            args[index],
                            buffer,
                            COUNT=tensor.numel(),
                            WIDTH=width,
                            STRIDE=stride,
                            SRC_DTYPE=dtype,
                            DST_DTYPE=None,
                            BLOCK=256,
                            num_warps=4,
                            num_stages=1,
                        )
                        if index < 3:
                            staged_spec[3 + index] = width
            _hc_combine_norm_compat_kernel[(spec[0] * spec[2],)](
                *staged_args,
                SPEC=tuple(staged_spec),
                EPS=eps,
                DTYPES=tuple(staged_dtypes),
                MATERIALIZED_DTYPE=dtypes[4],
                NATIVE_VEC=native_vector,
                num_warps=4,
                num_stages=1,
            )
            if tl.float64 in dtypes or tl.uint64 in dtypes:
                for index in (4, 5):
                    if dtypes[index] not in (tl.float64, tl.uint64):
                        continue
                    tensor = tensors[index]
                    _hc_convert_64bit_kernel[(triton.cdiv(tensor.numel(), 256),)](
                        staged_args[index],
                        args[index],
                        COUNT=tensor.numel(),
                        WIDTH=tensor.shape[1],
                        STRIDE=tensor.shape[1],
                        SRC_DTYPE=None,
                        DST_DTYPE=dtypes[index],
                        BLOCK=256,
                        num_warps=4,
                        num_stages=1,
                    )
        else:
            cores = runtime.torch_device_fn.get_device_properties(
                residual.device
            ).vector_core_num
            grid = lambda meta: (
                min(cores, triton.cdiv(spec[0] * spec[2], meta["BLOCK_M"])),
            )
            if (
                spec[0] >= 128
                and spec[1] in (2048, 2560)
                and spec[2] == 4
                and all(
                    tensor is None or tensor.dtype == torch.bfloat16
                    for tensor in tensors[:4]
                )
            ):
                _hc_combine_norm_bf16_kernel[grid](*args, SPEC=spec, EPS=eps)
            else:
                _hc_combine_norm_ascend_kernel[grid](
                    *args,
                    SPEC=spec,
                    EPS=eps,
                    DTYPES=dtypes,
                )
    return output, normalized


__all__ = ["hc_combine_norm"]
