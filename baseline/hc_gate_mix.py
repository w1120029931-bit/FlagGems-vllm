# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Installed vLLM HC gate mix with sourced and synthetic workloads."""

import importlib

import torch

OP_NAME = "hc_gate_mix"
NATIVE_MODULE = "vllm.models.qwen4_exp.nvidia.ops.hc"
NATIVE_API = NATIVE_MODULE + "." + OP_NAME
NATIVE_VERSION = None
NATIVE_UNAVAILABLE_REASON = None
DTYPES = (torch.bfloat16,)
IS_INPLACE = False
UPSTREAM_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
UPSTREAM_TEST = "tests/models/qwen4_exp/test_hc_ops.py"
UPSTREAM_URL = (
    f"https://github.com/vllm-project/vllm/blob/{UPSTREAM_COMMIT}/"
    f"{UPSTREAM_TEST}#L50-L61"
)
# Preserve the verified historical default as a synthetic-width source.
HIDDEN_CONFIG_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    "98dff2a81d747d1dba01a47f939f48c3526d4206/"
    "vllm/transformers_utils/configs/qwen3_next.py#L189-L197"
)
_HIDDEN_SIZES = (2048, 2560)
_SCENARIO_TOKENS = {"decode": (1, 8, 32), "prefill": (128, 512, 2048)}
_CORE_TOKENS = (1, 32, 512)


def native():
    global NATIVE_VERSION, NATIVE_UNAVAILABLE_REASON
    NATIVE_VERSION = None
    NATIVE_UNAVAILABLE_REASON = None
    try:
        import vllm

        NATIVE_VERSION = getattr(vllm, "__version__", None)
        op = getattr(importlib.import_module(NATIVE_MODULE), OP_NAME)
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = f"Installed vLLM {OP_NAME} is not callable"
        return None
    return op


def binding(n, h, dtype="bfloat16", hc=4, layout="dense", gate_dtype=None):
    return {
        "n": n,
        "h": h,
        "hc": hc,
        "dtype": dtype,
        "gate_dtype": dtype if gate_dtype is None else gate_dtype,
        "layout": layout,
        "seed": 0,
    }


def grid(core=False):
    upstream = {
        "n": 2,
        "hc": 4,
        "h": 2560,
        "layout": "dense",
        "seed": 0,
        "source_kind": "upstream-test",
        "scenario": "upstream",
        "source_url": UPSTREAM_URL,
    }
    cases = [upstream]
    for h in _HIDDEN_SIZES:
        for scenario, tokens in _SCENARIO_TOKENS.items():
            for n in tokens:
                if core and n not in _CORE_TOKENS:
                    continue
                cases.append(
                    dict(
                        upstream,
                        n=n,
                        h=h,
                        scenario=scenario,
                        source_kind="synthetic-coverage",
                        source_url=HIDDEN_CONFIG_URL if h == 2048 else UPSTREAM_URL,
                    )
                )
    return cases


def shape_keys(core=False):
    return [(case["n"], case["hc"], case["h"]) for case in grid(core)]


def key_shape(case):
    return [case["n"], case["hc"] * case["h"]]


def case_id(case):
    return f"n{case['n']}_hc{case['hc']}_h{case['h']}"


def _typed_case(case, dtype):
    result = dict(case)
    result["dtype"] = (
        str(dtype).removeprefix("torch.")
        if dtype is not None
        else result.get("dtype", "bfloat16")
    )
    result.setdefault("gate_dtype", result["dtype"])
    return result


def build_inputs(case, dtype=None, device="cuda"):
    case = _typed_case(case, dtype)
    n, width = case["n"], case["hc"] * case["h"]
    gen = torch.Generator(device="cpu").manual_seed(case.get("seed", 0))

    def make(dtype_name, gap):
        dt = getattr(torch, dtype_name)
        size = (n, width + gap)
        if dt == torch.bool:
            host = torch.randint(0, 2, size, generator=gen).to(dt)
        elif not dt.is_floating_point and not dt.is_complex:
            low = 0 if dtype_name.startswith("uint") else -3
            host = torch.randint(low, 4, size, generator=gen).to(dt)
        else:
            host = torch.randn(size, generator=gen).to(dt)
        host = host[:, :width]
        output = torch.empty_strided(host.shape, host.stride(), dtype=dt, device=device)
        if output.device.type == "musa" and output.dtype in (
            torch.uint16,
            torch.uint32,
        ):
            output.view(torch.uint8).copy_(host.view(torch.uint8))
        else:
            output.copy_(host)
        return output

    packed = case["layout"] == "packed"
    return (
        make(case["dtype"], 17 if packed else 0),
        make(case["gate_dtype"], 29 if packed else 0),
        case["hc"],
    ), {}


def config(case, dtype=None):
    case = _typed_case(case, dtype)
    n, h, hc = (case[key] for key in ("n", "h", "hc"))
    width = h * hc
    packed = case["layout"] == "packed"
    return {
        "case_id": case_id(case),
        "native_api": NATIVE_API,
        "native_version": NATIVE_VERSION,
        "binding": dict(case),
        "workload_source": case.get("source_url", UPSTREAM_URL),
        "inputs": {
            "x": {
                "shape": [n, width],
                "dtype": str(getattr(torch, case["dtype"])),
                "stride": [width + (17 if packed else 0), 1],
            },
            "gate": {
                "shape": [n, width],
                "dtype": str(getattr(torch, case["gate_dtype"])),
                "stride": [width + (29 if packed else 0), 1],
            },
            "hc_count": {"scalar": hc},
        },
        "outputs": {
            "output": {
                "shape": [n, h],
                "dtype": str(getattr(torch, case["dtype"])),
                "stride": [h, 1],
            }
        },
        "compute_dtype": "float32",
        "reduction_order": "sequential HC accumulation, then divide by HC",
        "graph": False,
        "mutated_inputs": [],
        "alias": "fresh output, no input alias",
    }


def assert_output(actual, expected, args):
    x, gate, hc = args
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape == (x.shape[0], x.shape[1] // hc)
    assert actual.dtype == expected.dtype == x.dtype
    assert actual.device == expected.device == x.device
    assert actual.is_contiguous()
    if actual.numel():
        assert actual.untyped_storage().data_ptr() not in (
            x.untyped_storage().data_ptr(),
            gate.untyped_storage().data_ptr(),
        )
    if not x.dtype.is_floating_point:
        assert torch.equal(actual.cpu(), expected.cpu())
        return
    got, ref = actual.detach().cpu().double(), expected.detach().cpu().double()
    torch.testing.assert_close(torch.isnan(got), torch.isnan(ref))
    torch.testing.assert_close(torch.isposinf(got), torch.isposinf(ref))
    torch.testing.assert_close(torch.isneginf(got), torch.isneginf(ref))
    tolerance = max(2e-6, 2 * torch.finfo(x.dtype).eps)
    torch.testing.assert_close(got, ref, atol=tolerance, rtol=tolerance, equal_nan=True)
