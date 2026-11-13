# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Storage conversions used by the Ascend HC kernels."""

import triton
import triton.language as tl


@triton.jit
def _load_f32(P, offsets, mask, DTYPE: tl.constexpr = None):
    if DTYPE is None:
        DTYPE = P.dtype.element_ty
    if DTYPE == tl.float64:
        words = P.to(tl.pointer_type(tl.uint32))
        lo = tl.load(words + 2 * offsets, mask, other=0)
        hi = tl.load(words + 2 * offsets + 1, mask, other=0)
        exponent = ((hi >> 20) & 0x7FF).to(tl.int32)
        fraction_hi = hi & 0xFFFFF
        mantissa = (fraction_hi << 3) | ((lo >> 29) & 7)
        discarded = (lo & 0x1FFFFFFF).to(tl.int32)
        increment = (discarded > 0x10000000) | (
            (discarded == 0x10000000) & ((mantissa & 1) != 0)
        )
        normal = (
            ((exponent - 896).to(tl.uint32) << 23) + mantissa + increment.to(tl.uint32)
        )

        shift = tl.minimum(tl.maximum(926 - exponent, 30), 53)
        one = tl.full((), 1, tl.uint32)
        significand_hi = fraction_hi | 0x100000
        low_shift = tl.minimum(shift, 31)
        low_quotient = (significand_hi << (32 - low_shift)) | (
            (lo >> low_shift) & ((one << (32 - low_shift)) - 1)
        )
        low_remainder = (lo & ((one << low_shift) - 1)).to(tl.int32)
        low_half = (one << (low_shift - 1)).to(tl.int32)
        low_increment = (low_remainder > low_half) | (
            (low_remainder == low_half) & ((low_quotient & 1) != 0)
        )

        high_shift = tl.maximum(shift - 32, 1)
        high_quotient = significand_hi >> high_shift
        high_remainder = (significand_hi & ((one << high_shift) - 1)).to(tl.int32)
        high_half = (one << (high_shift - 1)).to(tl.int32)
        high_increment = (high_remainder > high_half) | (
            (high_remainder == high_half) & ((lo != 0) | ((high_quotient & 1) != 0))
        )
        word_increment = ((lo >> 31) != 0) & (
            ((lo & 0x7FFFFFFF) != 0) | ((significand_hi & 1) != 0)
        )
        subnormal = tl.where(
            shift < 32,
            low_quotient + low_increment.to(tl.uint32),
            tl.where(
                shift == 32,
                significand_hi + word_increment.to(tl.uint32),
                high_quotient + high_increment.to(tl.uint32),
            ),
        )
        magnitude = tl.where(exponent >= 897, normal, subnormal)
        magnitude = tl.where(exponent < 873, 0, magnitude)
        magnitude = tl.where(exponent > 1150, 0x7F800000, magnitude)
        special = tl.where(
            (fraction_hi != 0) | (lo != 0), 0x7FC00000 | mantissa, 0x7F800000
        )
        magnitude = tl.where(exponent == 0x7FF, special, magnitude)
        bits = (hi & 0x80000000) | magnitude.to(tl.uint32)
        return bits.to(tl.float32, bitcast=True)
    elif DTYPE == tl.uint16:
        signed = P.to(tl.pointer_type(tl.int16))
        value = tl.load(signed + offsets, mask, other=0).to(tl.int32) & 0xFFFF
        return value.to(tl.float32)
    elif DTYPE == tl.uint32:
        signed = P.to(tl.pointer_type(tl.int32))
        value = tl.load(signed + offsets, mask, other=0)
        half = ((value.to(tl.uint32) >> 1) & 0x7FFFFFFF) | (value & 1)
        return tl.where(
            value < 0, half.to(tl.int32).to(tl.float32) * 2.0, value.to(tl.float32)
        )
    elif DTYPE == tl.uint64:
        words = P.to(tl.pointer_type(tl.uint32))
        lo = tl.load(words + 2 * offsets, mask, other=0)
        hi = tl.load(words + 2 * offsets + 1, mask, other=0)
        leading = tl.full((), 0, tl.int32)
        remaining = (hi & 0x7FFFFFFF).to(tl.int32)
        one = tl.full((), 1, tl.int32)
        for index in tl.static_range(5):
            step = 16 >> index
            take = remaining >= (one << step)
            leading += tl.where(take, step, 0)
            remaining = tl.where(take, remaining >> step, remaining)
        leading = tl.where(hi.to(tl.int32) < 0, 31, leading)
        shift = leading + 2
        low_shift = tl.minimum(shift, 31)
        word = tl.full((), 1, tl.uint32)
        quotient_low = (hi << (32 - low_shift)) | (
            (lo >> low_shift) & ((word << (32 - low_shift)) - 1)
        )
        sticky_low = (lo & ((word << low_shift) - 1)) != 0
        quotient = tl.where(
            shift < 32, quotient_low, tl.where(shift == 32, hi, (hi >> 1) & 0x7FFFFFFF)
        )
        sticky = tl.where(
            shift < 32, sticky_low, (lo != 0) | ((shift == 33) & ((hi & 1) != 0))
        )
        quotient = quotient | sticky.to(tl.uint32)
        scale = ((shift + 127) << 23).to(tl.float32, bitcast=True)
        high_value = quotient.to(tl.int32).to(tl.float32) * scale
        low_signed = lo.to(tl.int32)
        low_half = ((lo >> 1) & 0x7FFFFFFF) | (lo & 1)
        low_value = tl.where(
            low_signed < 0,
            low_half.to(tl.int32).to(tl.float32) * 2.0,
            low_signed.to(tl.float32),
        )
        return tl.where(hi == 0, low_value, high_value)
    else:
        return tl.load(P + offsets, mask, other=0).to(tl.float32)


@triton.jit
def _store_f32(P, offsets, value, mask, DTYPE: tl.constexpr = None):
    if DTYPE is None:
        DTYPE = P.dtype.element_ty
    if DTYPE == tl.float64:
        words = P.to(tl.pointer_type(tl.uint32))
        bits = value.to(tl.float32).to(tl.uint32, bitcast=True)
        sign = bits & 0x80000000
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        hi = sign | ((exponent + 896) << 20) | (mantissa >> 3)
        lo = (mantissa & 7) << 29

        fraction_bits = mantissa.to(tl.int32).to(tl.float32).to(tl.uint32, bitcast=True)
        leading = tl.maximum(((fraction_bits >> 23) & 0xFF).to(tl.int32) - 127, 0)
        normalized = mantissa << (23 - leading)
        sub_hi = (
            sign
            | ((874 + leading).to(tl.uint32) << 20)
            | ((normalized & 0x7FFFFF) >> 3)
        )
        sub_lo = (normalized & 7) << 29
        hi = tl.where(exponent == 0, tl.where(mantissa == 0, sign, sub_hi), hi)
        lo = tl.where(exponent == 0, tl.where(mantissa == 0, 0, sub_lo), lo)
        special_hi = (
            sign | 0x7FF00000 | (mantissa >> 3) | tl.where(mantissa != 0, 0x80000, 0)
        )
        hi = tl.where(exponent == 0xFF, special_hi, hi)
        tl.store(words + 2 * offsets, lo, mask)
        tl.store(words + 2 * offsets + 1, hi, mask)
    elif DTYPE == tl.int8 or DTYPE == tl.uint8 or DTYPE == tl.int16 or DTYPE == tl.int1:
        minimum: tl.constexpr = 0.0 if DTYPE == tl.uint8 else -32768.0
        maximum: tl.constexpr = 65535.0 if DTYPE == tl.uint8 else 32767.0
        value = tl.where(value != value, 0.0, value)
        value = tl.minimum(tl.maximum(value, minimum), maximum)
        tl.store(P + offsets, value.to(tl.int32), mask)
    elif DTYPE == tl.uint16:
        signed = P.to(tl.pointer_type(tl.int16))
        value = tl.where(value != value, 0.0, value)
        value = tl.minimum(tl.maximum(value, 0.0), 65535.0)
        tl.store(signed + offsets, value.to(tl.int32).to(tl.int16), mask)
    elif DTYPE == tl.uint32:
        signed = P.to(tl.pointer_type(tl.int32))
        low = value.to(tl.int32)
        high = (value - 2147483648.0).to(tl.int32) | -2147483648
        integer = tl.where(value < 2147483648.0, low, high)
        integer = tl.where(value >= 4294967296.0, -1, integer)
        integer = tl.where((value <= 0.0) | (value != value), 0, integer)
        tl.store(signed + offsets, integer, mask)
    elif DTYPE == tl.uint64:
        words = P.to(tl.pointer_type(tl.uint32))
        bits = value.to(tl.float32).to(tl.uint32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF).to(tl.int32) - 127
        mantissa = (bits & 0x7FFFFF) | 0x800000
        right = tl.minimum(tl.maximum(23 - exponent, 0), 23)
        left = tl.minimum(tl.maximum(exponent - 23, 0), 31)
        lo = tl.where(exponent < 23, mantissa >> right, mantissa << left)
        lo = tl.where((exponent < 0) | (exponent >= 55), 0, lo)
        high_right = tl.minimum(tl.maximum(55 - exponent, 0), 31)
        high_left = tl.minimum(tl.maximum(exponent - 55, 0), 8)
        hi = tl.where(exponent < 55, mantissa >> high_right, mantissa << high_left)
        hi = tl.where(exponent < 32, 0, hi)
        all_bits = tl.full((), -1, tl.int32).to(tl.uint32)
        lo = tl.where(exponent >= 64, all_bits, lo)
        hi = tl.where(exponent >= 64, all_bits, hi)
        zero = (value <= 0.0) | (value != value)
        lo = tl.where(zero, 0, lo)
        hi = tl.where(zero, 0, hi)
        tl.store(words + 2 * offsets, lo, mask)
        tl.store(words + 2 * offsets + 1, hi, mask)
    else:
        tl.store(P + offsets, value, mask)


@triton.jit
def _round_f32(value, DTYPE: tl.constexpr):
    if DTYPE == tl.float64:
        return value
    elif DTYPE == tl.int8 or DTYPE == tl.int16:
        value = tl.where(value != value, 0.0, value)
        value = tl.minimum(tl.maximum(value, -32768.0), 32767.0)
        integer = value.to(tl.int32)
        if DTYPE == tl.int8:
            integer = ((integer + 128) & 255) - 128
        return integer.to(tl.float32)
    elif DTYPE == tl.uint8 or DTYPE == tl.uint16:
        bits: tl.constexpr = 8 if DTYPE == tl.uint8 else 16
        value = tl.where(value != value, 0.0, value)
        value = tl.minimum(tl.maximum(value, 0.0), 65535.0)
        return (value.to(tl.int32) & ((1 << bits) - 1)).to(tl.float32)
    elif DTYPE == tl.uint32 or DTYPE == tl.uint64:
        maximum: tl.constexpr = (
            4294967296.0 if DTYPE == tl.uint32 else 18446744073709551616.0
        )
        rounded = tl.where(value < 8388608.0, value.to(tl.int32).to(tl.float32), value)
        rounded = tl.minimum(tl.maximum(rounded, 0.0), maximum)
        return tl.where(value != value, 0.0, rounded)
    else:
        return value.to(DTYPE).to(tl.float32)
