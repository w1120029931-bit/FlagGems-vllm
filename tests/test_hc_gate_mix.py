# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Use native vLLM when available, otherwise a test-only CPU Torch reference."""

import importlib

import pytest
import torch
from triton.compiler.errors import CompilationError

import flaggems_vllm as gems
from baseline import hc_gate_mix as baseline
from tests import conftest as cfg

pytestmark = pytest.mark.skipif(
    not gems.runtime.torch_device_fn.is_available(), reason="accelerator required"
)
_IMPLEMENTATIONS = ("generic", "hopper", "public")
_FP8_SKIP = pytest.mark.skipif(
    gems.vendor_name == "ascend"
    and gems.runtime.torch_device_fn.is_available()
    and "910B" in gems.runtime.torch_device_fn.get_device_name(gems.device).upper(),
    reason="Ascend 910B does not support FP8",
)
_DTYPES = (
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


def _reference(x, gate, hc_count):
    n, width = x.shape
    hidden = width // hc_count
    if n == 0 or hidden == 0:
        return torch.empty((n, hidden), dtype=x.dtype, device=x.device)
    values = _to_cpu(x).float().reshape(n, hc_count, hidden)
    gates = torch.sigmoid(_to_cpu(gate).float().reshape(n, hc_count, hidden))
    mixed = torch.zeros((n, hidden), dtype=torch.float32)
    for branch in range(hc_count):
        # Match sequential FP32 fused multiply-add before the final HC division.
        mixed = (
            mixed.double() + gates[:, branch].double() * values[:, branch].double()
        ).float()
    return _reference_cast(mixed / hc_count, x.dtype).to(x.device)


@pytest.fixture(scope="module")
def native():
    return baseline.native()


@pytest.fixture(scope="module")
def reference(native):
    return native if native is not None else _reference


@pytest.fixture(params=_IMPLEMENTATIONS)
def implementation(request):
    if request.param == "public":
        return gems.hc_gate_mix
    if request.param == "hopper":
        if gems.vendor_name != "nvidia" or torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("Hopper entry requires NVIDIA SM90")
        module = "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_gate_mix"
    else:
        module = "flaggems_vllm.ops.hc_gate_mix"
    return importlib.import_module(module).hc_gate_mix


def _storage_bytes(value):
    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        value.untyped_storage(), 0, (value.untyped_storage().nbytes(),), (1,)
    )


def _check(args, implementation, reference):
    snapshots = [_storage_bytes(value).cpu().clone() for value in args[:2]]
    expected = reference(*args)
    for value, saved in zip(args[:2], snapshots):
        assert torch.equal(
            _storage_bytes(value).cpu(), saved
        ), "reference mutated an input"
    actual = implementation(*args)
    baseline.assert_output(actual, expected, args)
    for value, saved in zip(args[:2], snapshots):
        assert torch.equal(
            _storage_bytes(value).cpu(), saved
        ), "candidate mutated an input"
    return actual, expected


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize(
    "case", baseline.grid(core=cfg.QUICK_MODE), ids=baseline.case_id
)
@pytest.mark.parametrize("dtype", baseline.DTYPES)
def test_workload_matrix(case, dtype, implementation, reference):
    args, _ = baseline.build_inputs(case, dtype, device=gems.device)
    _check(args, implementation, reference)


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize("dtype", _DTYPES)
def test_native_dtypes(dtype, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(3, 17, dtype, hc=3, layout="packed"), device=gems.device
    )
    _check(args, implementation, reference)


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize(
    "x_dtype,gate_dtype",
    [
        ("bfloat16", "float32"),
        ("float16", "float64"),
        ("float64", "bfloat16"),
        ("float32", "int16"),
        ("int32", "float32"),
        ("uint64", "float32"),
        ("bool", "float32"),
        pytest.param("float8_e4m3fn", "float16", marks=_FP8_SKIP),
        pytest.param("float16", "float8_e5m2", marks=_FP8_SKIP),
    ],
)
def test_mixed_dtypes(x_dtype, gate_dtype, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(3, 127, x_dtype, hc=3, layout="packed", gate_dtype=gate_dtype),
        device=gems.device,
    )
    _check(args, implementation, reference)


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize(
    "shape", [(0, 4, 17), (2, 4, 0), (1, 1, 1), (2, 3, 1), (7, 3, 513)]
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_boundary_shapes(shape, dtype, implementation, reference):
    n, hc, h = shape
    args, _ = baseline.build_inputs(
        baseline.binding(n, h, dtype, hc=hc), device=gems.device
    )
    _check(args, implementation, reference)


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize("layout", ["zero_rows", "overlap_rows", "offset"])
def test_row_layout(layout, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(4, 17, "float32", hc=3), device=gems.device
    )
    tensors = []
    for value in args[:2]:
        n, width = value.shape
        if layout == "zero_rows":
            value = value[:1].expand(n, width)
        elif layout == "overlap_rows":
            value = value.as_strided(value.shape, (width // 2, 1))
        else:
            value = torch.randn(n + 2, width + 7, device=gems.device)[
                1 : n + 1, 3 : width + 3
            ]
        tensors.append(value)
    _check((*tensors, args[2]), implementation, reference)


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize("field", [0, 1])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite(field, value, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(2, 17, "float32", hc=3), device=gems.device
    )
    args[field][0, 0] = value
    _check(args, implementation, reference)


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint64", "bool"])
def test_integer_nonfinite_gate(dtype, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(2, 17, dtype, hc=3, gate_dtype="float32"), device=gems.device
    )
    args[1][0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device=gems.device
    )
    _check(args, implementation, reference)


@pytest.mark.hc_gate_mix
def test_signed_zero(implementation, reference):
    x = torch.tensor([[0.0, -0.0, -0.0, 0.0] * 4], device=gems.device)
    gate = torch.zeros_like(x)
    actual, expected = _check((x, gate, 4), implementation, reference)
    torch.testing.assert_close(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize(
    "kind",
    [
        "hc_zero",
        "hc_negative",
        "nondivisible",
        "x_rank",
        "gate_rank",
        "shape",
        "x_stride",
        "gate_stride",
    ],
)
def test_native_rejected_metadata(kind, implementation, native):
    x = torch.randn(2, 12, device=gems.device)
    gate = torch.randn_like(x)
    hc = 3
    if kind == "hc_zero":
        hc = 0
    elif kind == "hc_negative":
        hc = -1
    elif kind == "nondivisible":
        hc = 5
    elif kind == "x_rank":
        x = x.reshape(1, 2, 12)
    elif kind == "gate_rank":
        gate = gate.reshape(1, 2, 12)
    elif kind == "shape":
        gate = gate[:1]
    elif kind == "x_stride":
        x = torch.randn(2, 24, device=gems.device)[:, ::2]
    elif kind == "gate_stride":
        gate = torch.randn(2, 24, device=gems.device)[:, ::2]
    native_error, native_message = {
        "hc_zero": (ZeroDivisionError, "integer modulo by zero"),
        "hc_negative": (RuntimeError, "negative dimension"),
        "gate_rank": (ValueError, r"too many values to unpack \(expected 2\)"),
    }.get(kind, (AssertionError, None))
    if native is not None:
        with pytest.raises(native_error, match=native_message):
            native(x, gate, hc)

    if kind in ("x_rank", "gate_rank", "shape"):
        candidate_message = "same 2-D shape"
    elif kind in ("x_stride", "gate_stride"):
        candidate_message = "unit inner strides"
    else:
        candidate_message = "hc_count must divide width"
    with pytest.raises(ValueError, match=candidate_message):
        implementation(x, gate, hc)


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize(
    "dtype_name",
    [
        "complex64",
        "complex128",
        pytest.param("float8_e4m3fnuz", marks=_FP8_SKIP),
        pytest.param("float8_e5m2fnuz", marks=_FP8_SKIP),
        pytest.param("float8_e8m0fnu", marks=_FP8_SKIP),
    ],
)
def test_other_native_dtypes(dtype_name, implementation, native):
    if not hasattr(torch, dtype_name):
        pytest.skip("dtype is absent from this PyTorch build")
    args, _ = baseline.build_inputs(
        baseline.binding(2, 17, dtype_name, hc=3), device=gems.device
    )
    if native is None:
        with pytest.raises(NotImplementedError, match="dtype"):
            implementation(*args)
        return
    try:
        expected = native(*args)
    except Exception as error:
        dtype = getattr(torch, dtype_name)
        if isinstance(error, KeyError):
            dtype_rejection = error.args in ((dtype,), (dtype_name,))
        elif isinstance(error, CompilationError):
            message = str(error).lower()
            fp8_type = any(
                token in message
                for token in (
                    "fp8e4b8",
                    "fp8e5b16",
                    "float8_e4m3fnuz",
                    "float8_e5m2fnuz",
                )
            )
            dtype_rejection = fp8_type and any(
                token in message
                for token in (
                    "not supported",
                    "unsupported",
                    "cannot cast",
                    "cannot convert",
                )
            )
        else:
            dtype_rejection = False
        if not dtype_rejection:
            raise
        with pytest.raises(NotImplementedError, match="dtype"):
            implementation(*args)
    else:
        baseline.assert_output(implementation(*args), expected, args)


@pytest.mark.hc_gate_mix
def test_no_out_parameter(implementation, native):
    args, _ = baseline.build_inputs(baseline.binding(2, 17), device=gems.device)
    out = torch.empty(2, 17, dtype=args[0].dtype, device=gems.device)
    for function in (
        (native, implementation) if native is not None else (implementation,)
    ):
        with pytest.raises(TypeError):
            function(*args, out=out)


@pytest.mark.hc_gate_mix
def test_public_hopper_dispatch():
    if gems.vendor_name != "nvidia" or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("public Hopper dispatch requires NVIDIA SM90")
    hopper = importlib.import_module(
        "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_gate_mix"
    ).hc_gate_mix
    assert gems.hc_gate_mix is hopper


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize("hc_count", [1 << 63, (1 << 65) + 1, -(1 << 63) - 1])
def test_hc_count_schema_overflow(hc_count, implementation, native):
    x = torch.empty((2, 0), device=gems.device)
    gate = torch.empty_like(x)
    if native is not None:
        with pytest.raises((RuntimeError, TypeError, OverflowError)):
            native(x, gate, hc_count)
    with pytest.raises(ValueError, match="signed-int64"):
        implementation(x, gate, hc_count)


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize("n", [0, 2])
@pytest.mark.parametrize("hc_count", [-1, -3])
def test_zero_width_negative_hc(n, hc_count, implementation, reference):
    x = torch.empty((n, 0), device=gems.device)
    gate = torch.empty_like(x)
    _check((x, gate, hc_count), implementation, reference)
