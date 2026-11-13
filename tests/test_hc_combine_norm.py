# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Use native vLLM when available, otherwise a test-only CPU Torch reference."""

import importlib
import math

import pytest
import torch
from triton.compiler.errors import CompilationError

import flaggems_vllm as gems
from baseline import hc_combine_norm as baseline
from tests import conftest as cfg

pytestmark = [
    pytest.mark.hc_combine_norm,
    pytest.mark.skipif(
        not gems.runtime.torch_device_fn.is_available(), reason="accelerator required"
    ),
]
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
    "float8_e4m3fn",
    "float8_e5m2",
)
IMPLEMENTATIONS = ("generic", "hopper", "public")


def _reference_cast(value, dtype, *, store=False):
    # Native final bool stores use signed bytes; intermediate bool stays logical.
    storage_dtype = torch.int8 if dtype == torch.bool and store else dtype
    if not storage_dtype.is_floating_point and storage_dtype != torch.bool:
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
    if (
        value.device.type == "musa"
        and value.dtype in (torch.uint16, torch.uint32)
        and not value.is_contiguous()
    ):
        return value.detach().view(torch.uint8).cpu().view(value.dtype)
    return value.detach().cpu()


def _reference_sum_sq(values, residual, block, weight, hidden):
    output_alignment = 16 if residual.shape[1] % 16 == 0 else 1
    widths = [math.gcd(hidden, 16 // residual.element_size(), output_alignment)]
    for tensor, branch_stride, row_stride in (
        (residual, hidden, residual.stride(0)),
        (block, 0, block.stride(0)),
        (weight, 0 if weight.numel() == hidden else hidden, 0),
    ):
        vector = 16 // tensor.element_size() if tensor.data_ptr() % 16 == 0 else 1
        row_alignment = 16 if row_stride % 16 == 0 else 1
        widths.append(math.gcd(vector, branch_stride, row_alignment))
    vector_width = min(4 * (values.shape[-1] // 512), max(widths))
    lane_values = max(4, vector_width)
    lanes = 512 // lane_values
    local = torch.arange(lane_values)
    positions = (
        torch.arange(lanes)[:, None] * vector_width
        + local % vector_width
        + (local // vector_width) * (lanes * vector_width)
    )
    tiles = values.reshape(*values.shape[:-1], values.shape[-1] // 512, 512)
    tiles = tiles[..., positions]
    partial = tiles[..., 1] * tiles[..., 1]
    for index in (0, *range(2, lane_values)):
        value = tiles[..., index].double()
        partial = (value * value + partial.double()).float()
    partial = partial.reshape(*partial.shape[:-1], lanes // 32, 32)
    for width in (16, 8, 4, 2, 1):
        partial = partial[..., :width] + partial[..., width : 2 * width]
    partial = partial.squeeze(-1)
    width = partial.shape[-1] // 2
    while width:
        partial = partial[..., :width] + partial[..., width : 2 * width]
        width //= 2
    tiles = partial.squeeze(-1)
    groups = max(1, vector_width // 4)
    partial = tiles[..., :groups]
    for start in range(groups, tiles.shape[-1], groups):
        partial = partial + tiles[..., start : start + groups]
    width = groups // 2
    while width:
        partial = partial[..., :width] + partial[..., width : 2 * width]
        width //= 2
    return partial


def _reference(residual, block_output, injection_logits, norm_weight, eps, hc_count):
    n, width = residual.shape
    hidden = width // hc_count
    # Preserve masked lanes through the mandatory integer/bool materialization.
    extent = 512 * (1 << ((hidden + 511) // 512 - 1).bit_length())
    values = torch.zeros((n, hc_count, extent), dtype=torch.float32)
    block = torch.zeros((n, 1, extent), dtype=torch.float32)
    values[..., :hidden] = _to_cpu(residual).float().reshape(n, hc_count, hidden)
    block[..., :hidden] = _to_cpu(block_output).float().reshape(n, 1, hidden)
    if injection_logits is None:
        combined = values + block
    else:
        logits = _to_cpu(injection_logits).float().reshape(n, hc_count, 1)
        gate = 2.0 * torch.sigmoid(logits / hc_count)
        # Widen only multiply-add to retain the single FP32 FMA rounding.
        combined = (values.double() + block.double() * gate.double()).float()
    combined = _reference_cast(combined, residual.dtype)
    rounded = combined.float()
    sum_sq = (
        _reference_sum_sq(rounded, residual, block_output, norm_weight, hidden)
        if eps < 0
        else (rounded * rounded).sum(dim=-1, keepdim=True)
    )
    scale = torch.tensor(1.0 / hidden, dtype=torch.float32)
    epsilon = torch.tensor(eps, dtype=torch.float32)
    # Match the native FP32 FMA after rounding both scalar operands.
    denominator = (sum_sq.double() * scale.double() + epsilon.double()).float()
    # CPU FP32 rsqrt can round 16 + eps to 0.25 and cross integer boundaries.
    inverse_rms = torch.rsqrt(denominator.double()).float()
    normalized = rounded[..., :hidden] * inverse_rms
    weight = _to_cpu(norm_weight).float()
    weight = weight.reshape(1, 1 if weight.numel() == hidden else hc_count, hidden)
    # The final Gemma multiply-add also rounds only once to FP32.
    normalized = (normalized.double() + normalized.double() * weight.double()).float()
    normalized = _reference_cast(normalized, residual.dtype, store=True)
    return (
        combined[..., :hidden].reshape(n, width).contiguous().to(residual.device),
        normalized.reshape(n, width).to(residual.device),
    )


@pytest.fixture(scope="module")
def native():
    return baseline.native()


@pytest.fixture(scope="module")
def reference(native):
    return native if native is not None else _reference


@pytest.fixture(params=IMPLEMENTATIONS)
def implementation(request):
    if request.param == "public":
        return gems.hc_combine_norm
    if request.param == "hopper":
        if gems.vendor_name != "nvidia" or torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("Hopper entry requires NVIDIA SM90")
        module = "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_combine_norm"
    else:
        module = "flaggems_vllm.ops.hc_combine_norm"
    return importlib.import_module(module).hc_combine_norm


def _storage_bytes(value):
    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        value.untyped_storage(), 0, (value.untyped_storage().nbytes(),), (1,)
    )


def _check(args, implementation, reference):
    # Include row padding and offsets in mutation checks, not only logical values.
    tensors = tuple(value for value in args[:4] if isinstance(value, torch.Tensor))
    snapshots = [_storage_bytes(value).cpu().clone() for value in tensors]
    expected = reference(*args)
    for value, saved in zip(tensors, snapshots):
        assert torch.equal(
            _storage_bytes(value).cpu(), saved
        ), "reference mutated an input"
    actual = implementation(*args)
    baseline.assert_outputs(actual, expected, args)
    if args[2] is None and args[0].dtype.is_floating_point:
        torch.testing.assert_close(
            actual[0].cpu().double(),
            expected[0].cpu().double(),
            atol=0,
            rtol=0,
            equal_nan=True,
        )
    for value, saved in zip(tensors, snapshots):
        assert torch.equal(
            _storage_bytes(value).cpu(), saved
        ), "candidate mutated an input"
    return actual, expected


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("has_logits", [True, False])
def test_all_native_dtypes(has_logits, dtype, implementation, reference):
    case = baseline.binding(
        3, 17, dtype, hc=3, layout="packed", shared=True, has_logits=has_logits
    )
    args, _ = baseline.build_inputs(case, device=gems.device)
    _check(args, implementation, reference)


@pytest.mark.parametrize(
    "dtypes",
    [
        ("bfloat16", "float16", "float32", "float64"),
        ("float32", "int16", "int8", "float64"),
        ("float64", "bfloat16", "float64", "float32"),
        ("float8_e4m3fn", "float16", "float32", "float32"),
        ("uint64", "float32", "int16", "float32"),
        ("bool", "float32", "float32", "float32"),
    ],
)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("has_logits", [True, False])
def test_mixed_dtypes(has_logits, dtypes, shared, implementation, reference):
    case = baseline.binding(
        3, 127, dtypes[0], hc=3, shared=shared, layout="packed", has_logits=has_logits
    )
    case.update(zip(("block_dtype", "logits_dtype", "weight_dtype"), dtypes[1:]))
    args, _ = baseline.build_inputs(case, device=gems.device)
    _check(args, implementation, reference)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("has_logits", [True, False])
def test_mixed_unsigned_representable(has_logits, shared, implementation, reference):
    case = baseline.binding(
        3, 127, "uint64", hc=3, shared=shared, layout="packed", has_logits=has_logits
    )
    case.update(block_dtype="float32", logits_dtype="int16", weight_dtype="float32")
    args, _ = baseline.build_inputs(case, device=gems.device)
    # Retain mixed dtypes and gaps while staying inside portable integer casts.
    args[1].abs_()
    args[3].abs_()
    _check(args, implementation, reference)


@pytest.mark.parametrize("dtype", baseline.DTYPES)
@pytest.mark.parametrize(
    "case", baseline.grid(core=cfg.QUICK_MODE), ids=baseline.case_id
)
def test_workload_matrix(case, dtype, implementation, reference):
    # Main workload matrix; the remaining tests cover interface edge cases.
    args, _ = baseline.build_inputs(case, dtype, device=gems.device)
    _check(args, implementation, reference)


@pytest.mark.parametrize("shape", [(0, 4, 17), (1, 1, 1), (2, 3, 1), (7, 3, 513)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("has_logits", [True, False])
def test_boundary_shapes(has_logits, shape, dtype, implementation, reference):
    n, hc, h = shape
    args, _ = baseline.build_inputs(
        baseline.binding(n, h, dtype, hc=hc, shared=True, has_logits=has_logits),
        device=gems.device,
    )
    _check(args, implementation, reference)


@pytest.mark.parametrize("layout", ["zero_rows", "overlap_rows", "offset"])
@pytest.mark.parametrize("has_logits", [True, False])
def test_row_strides_and_offsets(has_logits, layout, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(4, 17, "bfloat16", hc=3, shared=True, has_logits=has_logits),
        device=gems.device,
    )
    values = list(args)
    for index, value in enumerate(values[:3]):
        if value is None:
            continue
        n, width = value.shape
        if layout == "zero_rows":
            values[index] = value[:1].expand(n, width)
        elif layout == "overlap_rows":
            values[index] = value.as_strided(value.shape, (max(1, width // 2), 1))
        else:
            storage = torch.randn((n + 2, width + 7), device=value.device).to(
                value.dtype
            )
            values[index] = storage[1 : n + 1, 3 : width + 3]
    weight_storage = torch.randn(values[3].numel() + 5, device=gems.device).to(
        values[3].dtype
    )
    values[3] = weight_storage[3 : 3 + values[3].numel()]
    _check(tuple(values), implementation, reference)


@pytest.mark.parametrize(
    "rank,weight_dtype",
    [(0, "float32"), (2, "float32"), (3, "float32"), (0, "uint16"), (0, "uint32")],
)
@pytest.mark.parametrize("has_logits", [True, False])
def test_contiguous_weight_rank(
    has_logits, rank, weight_dtype, implementation, reference
):
    case = baseline.binding(
        3,
        1 if rank == 0 else 17,
        "float32",
        hc=3,
        shared=True,
        has_logits=has_logits,
    )
    case.update(weight_rank=rank, weight_dtype=weight_dtype)
    args, _ = baseline.build_inputs(case, device=gems.device)
    _check(args, implementation, reference)


@pytest.mark.parametrize("eps", [-0.9868139028549194, -0.9868138432502747])
def test_negative_epsilon_cancellation(eps, implementation, reference):
    case = baseline.binding(2, 2560, "float32", hc=3, shared=True, has_logits=False)
    case.update(eps=eps, seed=20261107)
    args, _ = baseline.build_inputs(case, device=gems.device)
    args[1].zero_()
    args[3].fill_(0.25)
    _check(args, implementation, reference)


@pytest.mark.parametrize(
    "hidden,dtype,layout,shared,has_logits",
    [
        (17, "float32", "dense", False, False),
        (258, "float32", "dense", False, False),
        (258, "float32", "offset2", False, False),
        (513, "float32", "packed", False, False),
        (2048, "float32", "offset", False, False),
        (4097, "float32", "dense", False, False),
        (1024, "bfloat16", "dense", True, True),
        (2560, "float16", "dense", False, False),
        (2560, "float64", "dense", True, True),
        (2560, "float64", "offset4", True, True),
    ],
)
@pytest.mark.parametrize("direction", [-float("inf"), float("inf")])
def test_negative_epsilon_boundary_layout(
    hidden, dtype, layout, shared, has_logits, direction, implementation, reference
):
    case = baseline.binding(
        2,
        hidden,
        dtype,
        hc=3,
        shared=shared,
        layout="dense" if layout.startswith("offset") else layout,
        has_logits=has_logits,
    )
    case.update(seed=20261107)
    if layout == "offset4":
        case.update(weight_dtype="bfloat16")
    args, _ = baseline.build_inputs(case, device=gems.device)
    values = list(args)
    if layout.startswith("offset"):
        offset = int(layout.removeprefix("offset") or "1")
        for index in (0, 1, 3):
            tensor = values[index]
            storage = torch.empty(
                tensor.numel() + offset, dtype=tensor.dtype, device=tensor.device
            )
            values[index] = storage[offset:].view(tensor.shape).copy_(tensor)
    values[1].zero_()
    values[3].fill_(0.25)
    center = -_to_cpu(values[0])[0, :hidden].float().square().mean()
    values[4] = torch.nextafter(center, torch.tensor(direction)).item()
    _check(tuple(values), implementation, reference)


def test_negative_epsilon_fused_denominator(implementation, reference):
    case = baseline.binding(1, 2560, "float32", hc=3, shared=True, has_logits=False)
    case.update(eps=-1.0)
    args, _ = baseline.build_inputs(case, device=gems.device)
    args[0].fill_(1.0)
    args[1].zero_()
    args[3].zero_()
    actual, _ = _check(args, implementation, reference)
    torch.testing.assert_close(
        actual[1], torch.full_like(actual[1], 8192.0), rtol=0, atol=0
    )


@pytest.mark.parametrize("eps", [0.0, -0.5, float("nan"), float("inf")])
@pytest.mark.parametrize("has_logits", [True, False])
def test_epsilon_semantics(has_logits, eps, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(2, 17, "float32", shared=True, has_logits=has_logits),
        device=gems.device,
    )
    values = list(args)
    values[0].zero_()
    values[1].zero_()
    values[4] = eps
    _check(tuple(values), implementation, reference)


@pytest.mark.parametrize(
    "field,has_logits",
    [(i, True) for i in (0, 1, 2, 3)] + [(i, False) for i in (0, 1, 3)],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs(field, has_logits, value, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(2, 17, "float32", hc=3, shared=True, has_logits=has_logits),
        device=gems.device,
    )
    args[field].reshape(-1)[0] = value
    _check(args, implementation, reference)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("has_logits", [True, False])
def test_materialized_rounding_boundary(has_logits, dtype, implementation, reference):
    case = baseline.binding(2, 4, dtype, hc=3, shared=True, has_logits=has_logits)
    case.update(block_dtype="float32", logits_dtype="float32", weight_dtype="float32")
    args, _ = baseline.build_inputs(case, device=gems.device)
    residual, block, logits, weight, _, _ = args
    residual.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0], device=gems.device).repeat(2, 3))
    block.copy_(
        torch.tensor([0.003, -0.003, 0.006, -0.006], device=gems.device).repeat(2, 1)
    )
    if logits is not None:
        logits.zero_()
    weight.fill_(0.25)
    actual, expected = _check(args, implementation, reference)
    # Injection equals one exactly; combine must materialize residual-dtype values.
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


@pytest.mark.parametrize("has_logits", [True, False])
def test_signed_zero(has_logits, implementation, reference):
    args, _ = baseline.build_inputs(
        baseline.binding(2, 4, "float32", hc=3, shared=True, has_logits=has_logits),
        device=gems.device,
    )
    args[0].copy_(torch.tensor([0.0, -0.0, 0.0, -0.0], device=gems.device).repeat(2, 3))
    args[1].copy_(torch.tensor([0.0, -0.0, -0.0, 0.0], device=gems.device).repeat(2, 1))
    if args[2] is not None:
        args[2].zero_()
    args[3].zero_()
    actual, expected = _check(args, implementation, reference)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(torch.signbit(got), torch.signbit(ref))


@pytest.mark.parametrize(
    "kind",
    [
        "hc_zero",
        "h_zero",
        "residual_rank",
        "block_shape",
        "logits_shape",
        "hc_divisibility",
        "residual_stride",
        "block_stride",
        "logits_stride",
        "weight_stride",
        "weight_size",
    ],
)
def test_native_rejected_inputs(kind, implementation, native):
    args, _ = baseline.build_inputs(
        baseline.binding(2, 4, "float32", hc=3, shared=True), device=gems.device
    )
    values = list(args)
    if kind == "hc_zero":
        values[5] = 0
    elif kind == "h_zero":
        values[0], values[1], values[3] = (
            values[0][:, :0],
            values[1][:, :0],
            values[3][:0],
        )
    elif kind == "residual_rank":
        values[0] = values[0].unsqueeze(0)
    elif kind == "block_shape":
        values[1] = values[1][:, :-1]
    elif kind == "logits_shape":
        values[2] = values[2][:, :-1]
    elif kind == "hc_divisibility":
        values[5] = 5
    elif kind.endswith("_stride") and kind != "weight_stride":
        index = {"residual_stride": 0, "block_stride": 1, "logits_stride": 2}[kind]
        value = values[index]
        values[index] = torch.empty(
            (value.shape[0], value.shape[1] * 2), device=gems.device
        )[:, ::2]
    elif kind == "weight_stride":
        values[3] = torch.empty(values[3].numel() * 2, device=gems.device)[::2]
    elif kind == "weight_size":
        values[3] = values[3][:-1]
    native_error, message = {
        "hc_zero": (ZeroDivisionError, "integer modulo by zero"),
        "h_zero": (
            CompilationError,
            "arange's end argument must be greater than the start argument",
        ),
        "residual_rank": (ValueError, r"too many values to unpack \(expected 2\)"),
    }.get(kind, (AssertionError, None))
    if native is not None:
        with pytest.raises(native_error, match=message):
            native(*values)
    with pytest.raises(ValueError):
        implementation(*values)


@pytest.mark.parametrize(
    "dtype",
    ["complex64", "complex128", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"],
)
@pytest.mark.parametrize("has_logits", [True, False])
def test_native_unsupported_dtypes(has_logits, dtype, implementation, native):
    case = baseline.binding(2, 4, dtype, hc=3, shared=True, has_logits=has_logits)
    args, _ = baseline.build_inputs(case, device=gems.device)
    if native is not None:
        if dtype in ("complex64", "complex128", "float8_e8m0fnu"):
            with pytest.raises(KeyError) as error:
                native(*args)
            assert error.value.args in ((dtype,), (getattr(torch, dtype),))
        else:
            fp8_type = {"float8_e4m3fnuz": "fp8e4b8", "float8_e5m2fnuz": "fp8e5b16"}[
                dtype
            ]
            with pytest.raises(
                CompilationError,
                match=f"type {fp8_type} not supported in this architecture",
            ):
                native(*args)
    with pytest.raises(NotImplementedError, match="input dtype is unsupported"):
        implementation(*args)


@pytest.mark.parametrize("has_logits", [True, False])
def test_no_out_keyword(has_logits, implementation, native):
    args, _ = baseline.build_inputs(
        baseline.binding(2, 4, has_logits=has_logits), device=gems.device
    )
    if native is not None:
        with pytest.raises(TypeError):
            native(*args, out=(torch.empty_like(args[0]), torch.empty_like(args[0])))
    with pytest.raises(TypeError):
        implementation(
            *args, out=(torch.empty_like(args[0]), torch.empty_like(args[0]))
        )


def test_public_dispatch():
    if gems.vendor_name != "nvidia" or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper dispatch assertion requires NVIDIA SM90")
    expected = importlib.import_module(
        "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_combine_norm"
    ).hc_combine_norm
    assert gems.hc_combine_norm is expected


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("hidden_size", [1, 128, 513])
def test_integer_nan_materialization_padding(
    dtype, hidden_size, implementation, reference
):
    # Native tiles each reduction in blocks of 512. Masked combine lanes with
    # NaN injection still cross the integer materialization boundary before the
    # FP32 norm; changing padding width can change observable integer outputs.
    case = baseline.binding(2, hidden_size, dtype, hc=3, shared=True)
    case.update(block_dtype="float32", logits_dtype="float32", weight_dtype="float32")
    args, _ = baseline.build_inputs(case, device=gems.device)
    args[0].fill_(3)
    args[1].zero_()
    args[2].copy_(
        torch.tensor(
            [[float("nan"), 0.0, 3.0], [3.0, float("nan"), 0.0]], device=gems.device
        )
    )
    args[3].fill_(3)
    _check(args, implementation, reference)


def test_injection_argument_is_required(implementation, native):
    args, _ = baseline.build_inputs(
        baseline.binding(2, 4, has_logits=False), device=gems.device
    )
    for op in (native, implementation) if native is not None else (implementation,):
        with pytest.raises(TypeError, match="injection_logits"):
            op(
                residual=args[0],
                block_output=args[1],
                norm_weight=args[3],
                eps=args[4],
                hc_count=args[5],
            )
