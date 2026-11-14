# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Installed vLLM baseline, upstream HC test, and synthetic serving workloads."""

import importlib

import torch

OP_NAME = "hc_combine_norm"
NATIVE_MODULE = (
    "vllm_metax.models.qwen4_exp.metax.ops.hc"
    if getattr(torch.version, "maca", None)
    else "vllm.models.qwen4_exp.nvidia.ops.hc"
)
NATIVE_API = NATIVE_MODULE + "." + OP_NAME
NATIVE_VERSION = None
UPSTREAM_NONE_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
NATIVE_UNAVAILABLE_REASON = None
DTYPES = (torch.bfloat16,)
IS_INPLACE = False
SEED = 20261107


def native():
    global NATIVE_VERSION, NATIVE_UNAVAILABLE_REASON
    NATIVE_VERSION = None
    NATIVE_UNAVAILABLE_REASON = None
    try:
        package = importlib.import_module(NATIVE_MODULE.split(".")[0])
        NATIVE_VERSION = getattr(package, "__version__", None)
        op = getattr(importlib.import_module(NATIVE_MODULE), OP_NAME)
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = f"Installed vLLM {OP_NAME} is not callable"
        return None
    return op


def binding(
    n,
    h,
    dtype="bfloat16",
    hc=4,
    shared=False,
    layout="dense",
    mixed=False,
    has_logits=True,
):
    return {
        "n": n,
        "hc": hc,
        "h": h,
        "dtype": dtype,
        "has_logits": has_logits,
        "block_dtype": "float16" if mixed else dtype,
        "logits_dtype": "float32" if mixed else dtype,
        "weight_dtype": "float32" if mixed else dtype,
        "shared": shared,
        "layout": layout,
        "weight_rank": 1,
        "eps": 1e-6,
        "seed": SEED,
    }


UPSTREAM_COMMIT = "98dff2a81d747d1dba01a47f939f48c3526d4206"
UPSTREAM_TEST = "tests/models/qwen4_exp/test_hc_ops.py"
UPSTREAM_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    f"{UPSTREAM_COMMIT}/{UPSTREAM_TEST}#L71-L94"
)
MODEL_CONFIG_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    f"{UPSTREAM_COMMIT}/vllm/models/qwen4_exp/config.py#L28-L67"
)
HIDDEN_CONFIG_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    f"{UPSTREAM_COMMIT}/vllm/transformers_utils/configs/qwen3_next.py#L189-L197"
)
UPSTREAM_NONE_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    f"{UPSTREAM_NONE_COMMIT}/{UPSTREAM_TEST}#L118-L136"
)
_UPSTREAM_CASE = {
    "n": 2,
    "has_logits": True,
    "hc": 4,
    "h": 2560,
    "shared": False,
    "layout": "dense",
    "weight_rank": 1,
    "eps": 1e-6,
    "seed": 0,
    "source_kind": "upstream-test",
    "scenario": "upstream",
    "source_url": UPSTREAM_URL,
}


# Synthetic token counts; H comes from model defaults (2048) or tests (2560).
_HIDDEN_SIZES = (2048, 2560)
_SCENARIO_TOKENS = {"decode": (1, 8, 32), "prefill": (128, 512, 2048)}
_CORE_TOKENS = (1, 32, 512)


def key_shape(case):
    # All modes use composite keys so residual-shape lookup cannot mix them.
    return [case["n"], case["hc"], case["h"], int(case.get("has_logits", True))]


def case_id(case):
    suffix = "" if case.get("has_logits", True) else "_unit_injection"
    return f"n{case['n']}_hc{case['hc']}_h{case['h']}{suffix}"


def grid(core=False):
    # Keep dtype outside the binding: the collector iterates grid() x DTYPES.
    cases = [dict(_UPSTREAM_CASE)]
    for h in _HIDDEN_SIZES:
        for scenario, tokens in _SCENARIO_TOKENS.items():
            for n in tokens:
                if core and n not in _CORE_TOKENS:
                    continue
                cases.append(
                    dict(
                        _UPSTREAM_CASE,
                        n=n,
                        h=h,
                        source_kind="synthetic-coverage",
                        scenario=scenario,
                        source_url=HIDDEN_CONFIG_URL if h == 2048 else UPSTREAM_URL,
                    )
                )
    cases.extend(
        dict(
            _UPSTREAM_CASE,
            n=n,
            has_logits=False,
            source_kind="upstream-test",
            scenario="unit-injection",
            source_url=UPSTREAM_NONE_URL,
        )
        for n in (1, 17, 2048)
    )
    return cases


def shape_keys(core=False):
    return [tuple(key_shape(case)) for case in grid(core)]


def _typed_case(case, dtype=None):
    """Resolve collector dtype while retaining explicit mixed-dtype regressions."""
    resolved = dict(case)
    name = (
        str(dtype).removeprefix("torch.")
        if dtype is not None
        else resolved.get("dtype", "bfloat16")
    )
    resolved["dtype"] = name
    resolved.setdefault("has_logits", True)
    for field in ("block_dtype", "logits_dtype", "weight_dtype"):
        resolved.setdefault(field, name)
    return resolved


def build_inputs(case, dtype=None, device="cuda"):
    """Generate identical CPU bits and preserve row gaps/independent dtypes."""
    case = _typed_case(case, dtype)
    n, hc, h = (case[name] for name in ("n", "hc", "h"))
    gen = torch.Generator(device="cpu").manual_seed(case.get("seed", SEED))
    packed = case.get("layout", "dense") == "packed"

    def tensor(width, dtype_name, gap=0):
        dt = getattr(torch, dtype_name)
        shape = (n, width + gap)
        if dt == torch.bool:
            host = torch.randint(0, 2, shape, generator=gen).to(dt)
        elif not dt.is_floating_point and not dt.is_complex:
            low = 0 if dtype_name.startswith("uint") else -3
            host = torch.randint(low, 4, shape, generator=gen).to(dt)
        else:
            host = torch.randn(shape, generator=gen, dtype=torch.float32).to(dt)
        host = host[:, :width]
        result = torch.empty_strided(host.shape, host.stride(), dtype=dt, device=device)
        if result.device.type == "musa" and result.dtype in (
            torch.uint16,
            torch.uint32,
        ):
            result.view(torch.uint8).copy_(host.view(torch.uint8))
        else:
            result.copy_(host)
        return result

    residual = tensor(hc * h, case["dtype"], 17 if packed else 0)
    block = tensor(h, case["block_dtype"], 11 if packed else 0)
    logits = (
        tensor(hc, case["logits_dtype"], 257 if packed else 0)
        if case["has_logits"]
        else None
    )
    count = h if case["shared"] else hc * h
    weight_dtype = getattr(torch, case["weight_dtype"])
    if weight_dtype.is_floating_point or weight_dtype.is_complex:
        weight = torch.randn(count, generator=gen).to(weight_dtype)
    else:
        weight = torch.randint(0, 3, (count,), generator=gen).to(weight_dtype)
    weight = weight.to(device)
    rank = case.get("weight_rank", 1)
    weight = (
        weight.reshape((1,) * (rank - 1) + (count,)) if rank else weight.reshape(())
    )
    return (residual, block, logits, weight, case["eps"], hc), {}


def config(case, dtype=None):
    case = _typed_case(case, dtype)
    n, hc, h = (case[name] for name in ("n", "hc", "h"))
    packed = case["layout"] == "packed"

    def tensor(width, dtype, gap=0):
        return {
            "shape": [n, width],
            "dtype": str(getattr(torch, dtype)),
            "stride": [width + gap, 1],
        }

    count = h if case["shared"] else hc * h
    return {
        "case_id": case_id(case),
        "native_api": NATIVE_API,
        "native_version": NATIVE_VERSION,
        "binding": dict(case),
        "workload_source": case.get("source_url"),
        "workload_source_kind": case.get("source_kind", "interface-regression"),
        "workload_scenario": case.get("scenario", "interface-regression"),
        "model_config_reference": MODEL_CONFIG_URL,
        "is_model_trace": False,
        "inputs": {
            "residual": tensor(hc * h, case["dtype"], 17 if packed else 0),
            "block_output": tensor(h, case["block_dtype"], 11 if packed else 0),
            "injection_logits": (
                tensor(hc, case["logits_dtype"], 257 if packed else 0)
                if case["has_logits"]
                else None
            ),
            "norm_weight": {
                "shape": (
                    ([1] * (case["weight_rank"] - 1) + [count])
                    if case["weight_rank"]
                    else []
                ),
                "dtype": str(getattr(torch, case["weight_dtype"])),
            },
            "eps": {"scalar": case["eps"]},
            "hc_count": {"scalar": hc},
        },
        "outputs": {
            "combined": tensor(hc * h, case["dtype"]),
            "normalized": tensor(hc * h, case["dtype"]),
        },
        "graph": False,
        "compute_dtype": "float32",
        "rounding_boundary": "combine -> residual dtype -> float32 normalization",
        "mutated_inputs": [],
        "alias": "two fresh independent outputs; neither aliases an input",
    }


def assert_outputs(actual, expected, args):
    """Check native agreement and the complete public output contract."""
    assert isinstance(actual, tuple) and len(actual) == 2
    residual = args[0]
    pointers = [
        x.untyped_storage().data_ptr() for x in args[:4] if isinstance(x, torch.Tensor)
    ]
    output_pointers = []
    dtype = residual.dtype
    for value, reference in zip(actual, expected):
        assert isinstance(value, torch.Tensor)
        assert value.shape == reference.shape == residual.shape
        assert value.dtype == reference.dtype == dtype
        assert value.device == reference.device == residual.device
        assert value.is_contiguous()
        if value.numel():
            pointer = value.untyped_storage().data_ptr()
            assert pointer not in pointers and pointer not in output_pointers
            output_pointers.append(pointer)
        if not dtype.is_floating_point:
            assert torch.equal(value.detach().cpu(), reference.detach().cpu())
            continue
        # CPU FP64 also permits comparisons of FP8 containers.
        got, ref = value.detach().cpu().double(), reference.detach().cpu().double()
        tolerance = (
            max(2e-6, 2 * torch.finfo(dtype).eps) if dtype.is_floating_point else 0
        )
        torch.testing.assert_close(torch.isnan(got), torch.isnan(ref))
        torch.testing.assert_close(torch.isposinf(got), torch.isposinf(ref))
        torch.testing.assert_close(torch.isneginf(got), torch.isneginf(ref))
        torch.testing.assert_close(
            got, ref, atol=tolerance, rtol=tolerance, equal_nan=True
        )
