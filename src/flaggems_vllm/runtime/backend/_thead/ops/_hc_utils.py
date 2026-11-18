# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import triton
import triton.language as tl


@triton.jit
def _hc_cast(value, dtype: tl.constexpr):
    if dtype == tl.float8e5:
        is_nan = value != value
        # Match NVIDIA's finite saturation before PPU's packed conversion.
        value = tl.minimum(tl.maximum(value, -57344.0), 57344.0)
        # PPU E5M2 lowering rounds exact halfway values away from zero.
        half = value.to(tl.float16, fp_downcast_rounding="rtz")
        tie = ((half.to(tl.uint16, bitcast=True) & 0x1FF) == 0x80) & (
            half.to(tl.float32) == value
        )
        previous = (value.to(tl.uint32, bitcast=True) - 1).to(tl.float32, bitcast=True)
        value = tl.where(tie, previous, value)
    result = value.to(dtype)
    if dtype == tl.float8e5:
        # The packed half-to-E5M2 conversion also loses NaNs.
        bits = tl.where(is_nan, 0x7F, result.to(tl.uint8, bitcast=True))
        result = bits.to(tl.uint8).to(dtype, bitcast=True)
    return result
