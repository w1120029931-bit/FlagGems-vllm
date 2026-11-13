# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Use native vLLM when available, otherwise a test-only CPU Torch reference."""

import importlib

import pytest
import torch
from triton.compiler.errors import CompilationError

import flaggems_vllm as gems
from baseline import hc_silu as baseline
from tests import conftest as cfg

pytestmark = [
    pytest.mark.hc_silu,
    pytest.mark.skipif(
        not gems.runtime.torch_device_fn.is_available(), reason="accelerator required"
    ),
]
_FP8_SKIP = pytest.mark.skipif(
    gems.vendor_name == "ascend"
    and gems.runtime.torch_device_fn.is_available()
    and "910B" in gems.runtime.torch_device_fn.get_device_name(gems.device).upper(),
    reason="Ascend 910B does not support FP8",
)
SUPPORTED_DTYPES = (
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "bool",
    pytest.param("float8_e4m3fn", marks=_FP8_SKIP),
    pytest.param("float8_e5m2", marks=_FP8_SKIP),
)


def _is_hopper():
    return gems.vendor_name == "nvidia" and torch.cuda.get_device_capability()[0] == 9


def _reference_cast(value, dtype):
    # Native final bool stores use signed bytes.
    storage_dtype = torch.int8 if dtype == torch.bool else dtype
    if not storage_dtype.is_floating_point:
        limits = torch.iinfo(storage_dtype)
        truncated = torch.trunc(value.double())
        representable = (
            torch.isfinite(truncated)
            & (truncated >= float(limits.min))
            & (truncated < float(limits.max + 1))
        )
        if not representable.all():
            pytest.skip(
                f"CPU Torch float-to-{storage_dtype} reference is undefined outside "
                "the finite representable range; requires a real native oracle"
            )
    result = value.to(storage_dtype)
    return result.view(dtype) if storage_dtype != dtype else result


def _to_cpu(value):
    if value.device.type == "musa" and value.dtype in (torch.uint16, torch.uint32):
        return value.detach().view(torch.uint8).cpu().view(value.dtype)
    return value.detach().cpu()


def _reference(x, hc_count):
    scaled = _to_cpu(x).float() / hc_count
    return _reference_cast(scaled * torch.sigmoid(scaled), x.dtype).to(x.device)


@pytest.fixture(scope="module")
def native():
    return baseline.native()


@pytest.fixture(scope="module")
def reference(native):
    return native if native is not None else _reference


@pytest.fixture(params=("generic", "hopper", "public"))
def implementation(request):
    if request.param == "public":
        return gems.hc_silu
    if request.param == "hopper":
        if not _is_hopper():
            pytest.skip("Hopper entry requires NVIDIA SM90")
        module = "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_silu"
    else:
        module = "flaggems_vllm.ops.hc_silu"
    return importlib.import_module(module).hc_silu


def _storage_bytes(x):
    return torch.empty(0, dtype=torch.uint8, device=x.device).set_(
        x.untyped_storage(), 0, (x.untyped_storage().nbytes(),), (1,)
    )


def _check(args, implementation, reference):
    x = args[0]
    saved = _storage_bytes(x).cpu().clone()
    expected = reference(*args)
    assert torch.equal(_storage_bytes(x).cpu(), saved), "reference mutated input"
    actual = implementation(*args)
    baseline.assert_outputs(actual, expected, args)
    assert torch.equal(_storage_bytes(x).cpu(), saved), "candidate mutated input"
    return actual, expected


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_native_dtypes(dtype, implementation, reference):
    if not hasattr(torch, dtype):
        pytest.skip(f"installed Torch does not expose {dtype}")
    args, _ = baseline.build_inputs(
        baseline.binding(3, 17, dtype, row_gap=11), device=gems.device
    )
    _check(args, implementation, reference)


@pytest.mark.parametrize("dtype", baseline.DTYPES)
@pytest.mark.parametrize(
    "case", baseline.grid(core=cfg.QUICK_MODE), ids=baseline.case_id
)
def test_workloads(case, dtype, implementation, reference):
    args, _ = baseline.build_inputs(case, dtype, device=gems.device)
    _check(args, implementation, reference)


@pytest.mark.parametrize("shape", [(0, 0), (0, 17), (3, 0), (1, 1), (7, 513)])
@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64"))
def test_boundary_shapes(shape, dtype, implementation, native, reference):
    args, _ = baseline.build_inputs(baseline.binding(*shape, dtype), device=gems.device)
    if shape[1] == 0:
        if native is not None:
            with pytest.raises(CompilationError, match="arange's end argument"):
                native(*args)
        with pytest.raises(ValueError, match="nonzero input width"):
            implementation(*args)
    else:
        _check(args, implementation, reference)


@pytest.mark.parametrize("hc_count", (0, -4, -1, 1, 3, 4, 17, 1 << 31))
def test_scalar_semantics(hc_count, implementation, reference):
    x = torch.tensor([[0.0, -0.0, -7.0, -1.0, 1.0, 7.0]], device=gems.device)
    _check((x, hc_count), implementation, reference)


@pytest.mark.parametrize("layout", ("zero_rows", "overlap_rows", "offset"))
def test_row_layouts(layout, implementation, reference):
    x = torch.randn((6, 29), device=gems.device)
    if layout == "zero_rows":
        x = x[:1, :17].expand(4, 17)
    elif layout == "overlap_rows":
        x = x.as_strided((4, 17), (7, 1))
    else:
        x = x[1:5, 3:20]
    _check((x, 3), implementation, reference)


@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
def test_nonfinite_and_signed_zero(dtype, implementation, reference):
    x = torch.tensor(
        [[float("nan"), float("inf"), -float("inf"), 0.0, -0.0, -80.0, 80.0]],
        dtype=dtype,
        device=gems.device,
    )
    actual, expected = _check((x, 4), implementation, reference)
    zero = expected == 0
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_fp32_scaling_before_output_cast(dtype, implementation, reference):
    x = torch.linspace(-8, 8, 4097, device=gems.device).to(dtype).reshape(1, -1)
    actual, expected = _check((x, 3), implementation, reference)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ("rank", "stride", "scalar_type", "scalar_overflow"))
def test_rejected_arguments(kind, implementation, native):
    x = torch.ones((2, 8), device=gems.device)
    hc_count = 4
    if kind == "rank":
        x = x.unsqueeze(0)
    elif kind == "stride":
        x = x[:, ::2]
    elif kind == "scalar_type":
        hc_count = 1.5
    else:
        hc_count = 1 << 65
    native_error, native_message = {
        "rank": (ValueError, r"too many values to unpack \(expected 2\)"),
        "stride": (AssertionError, None),
        "scalar_type": (RuntimeError, "argument 'hc_count'"),
        "scalar_overflow": (RuntimeError, "argument 'hc_count'"),
    }[kind]
    if native is not None:
        with pytest.raises(native_error, match=native_message):
            native(x, hc_count)

    candidate_error, candidate_message = {
        "rank": (ValueError, "2-D tensor with unit inner stride"),
        "stride": (ValueError, "2-D tensor with unit inner stride"),
        "scalar_type": (TypeError, "hc_count must be an integer"),
        "scalar_overflow": (ValueError, "native int64 argument"),
    }[kind]
    with pytest.raises(candidate_error, match=candidate_message):
        implementation(x, hc_count)


@pytest.mark.parametrize(
    "dtype",
    [
        "complex64",
        "complex128",
        pytest.param("float8_e4m3fnuz", marks=_FP8_SKIP),
        pytest.param("float8_e5m2fnuz", marks=_FP8_SKIP),
        pytest.param("float8_e8m0fnu", marks=_FP8_SKIP),
    ],
)
@pytest.mark.parametrize("shape", ((2, 8), (0, 8), (2, 0)))
def test_unsupported_dtypes(dtype, shape, implementation, native):
    if not hasattr(torch, dtype):
        pytest.skip(f"installed Torch does not expose {dtype}")
    args, _ = baseline.build_inputs(baseline.binding(*shape, dtype), device=gems.device)
    if native is not None:
        if dtype in ("complex64", "complex128", "float8_e8m0fnu"):
            with pytest.raises(KeyError) as caught:
                native(*args)
            assert caught.value.args == (dtype,)
        else:
            native_type = {"float8_e4m3fnuz": "fp8e4b8", "float8_e5m2fnuz": "fp8e5b16"}[
                dtype
            ]
            with pytest.raises(
                CompilationError, match=f"type {native_type} not supported"
            ):
                native(*args)
    with pytest.raises(NotImplementedError, match="dtype is unsupported"):
        implementation(*args)


def test_no_out_keyword(implementation, native):
    x = torch.ones((2, 8), device=gems.device)
    if native is not None:
        with pytest.raises(TypeError):
            native(x, 4, out=torch.empty_like(x))
    with pytest.raises(TypeError):
        implementation(x, 4, out=torch.empty_like(x))


def test_public_dispatch():
    if not _is_hopper():
        pytest.skip("Hopper dispatch requires NVIDIA SM90")
    expected = importlib.import_module(
        "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_silu"
    ).hc_silu
    assert gems.hc_silu is expected
