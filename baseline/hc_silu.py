# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Installed vLLM HC SiLU with sourced and synthetic workloads."""

import importlib

import torch

OP_NAME = "hc_silu"
NATIVE_MODULE = "vllm.models.qwen4_exp.nvidia.ops.hc"
NATIVE_API = NATIVE_MODULE + "." + OP_NAME
NATIVE_VERSION = None
NATIVE_UNAVAILABLE_REASON = None
DTYPES = (torch.bfloat16,)
IS_INPLACE = False
SEED = 20261111
UPSTREAM_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
SOURCE_ROOT = "https://github.com/vllm-project/vllm/blob/" + UPSTREAM_COMMIT
MODEL_CONFIG_URL = SOURCE_ROOT + "/vllm/models/qwen4_exp/nvidia/model.py"
MODEL_CALL_URL = (
    SOURCE_ROOT + "/vllm/models/qwen4_exp/nvidia/hyperconnection.py#L137-L159"
)
FUSED_REFERENCE_URL = SOURCE_ROOT + "/tests/models/qwen4_exp/test_hc_ops.py#L139-L157"
FUSED_REFERENCE_TOKENS = (1, 3, 5, 17, 48)
LEGACY_SOURCE_ROOT = (
    "https://github.com/vllm-project/vllm/blob/"
    "98dff2a81d747d1dba01a47f939f48c3526d4206"
)
HIDDEN_CONFIG_URL = (
    LEGACY_SOURCE_ROOT + "/vllm/transformers_utils/configs/qwen3_next.py#L189-L197"
)
HC_TEST_URL = LEGACY_SOURCE_ROOT + "/tests/models/qwen4_exp/test_hc_ops.py#L20-L24"


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


def binding(n, h, dtype="bfloat16", hc=4, row_gap=0):
    return {"n": n, "h": h, "hc": hc, "dtype": dtype, "row_gap": row_gap, "seed": SEED}


def grid(core=False):
    cases = []
    for width in (320, 2048, 2560):
        for scenario, tokens in (("decode", (1, 8, 32)), ("prefill", (128, 512, 2048))):
            for n in tokens:
                if core and n not in (1, 32, 512):
                    continue
                reference = width == 320 and n in FUSED_REFERENCE_TOKENS
                case = {
                    "n": n,
                    "h": width,
                    "hc": 4,
                    "row_gap": 16 if width == 320 else 0,
                    "seed": SEED,
                    "source_kind": (
                        "upstream-fused-test-reference"
                        if reference
                        else (
                            "model-config-derived"
                            if width == 320
                            else "synthetic-width-coverage"
                        )
                    ),
                    "scenario": scenario,
                    "source_url": (
                        FUSED_REFERENCE_URL
                        if reference
                        else (
                            MODEL_CALL_URL
                            if width == 320
                            else HIDDEN_CONFIG_URL if width == 2048 else HC_TEST_URL
                        )
                    ),
                }
                cases.append(case)
    for n in FUSED_REFERENCE_TOKENS[1:]:
        if not core or n == 17:
            cases.append(
                dict(
                    binding(n, 320, row_gap=16),
                    source_kind="upstream-fused-test-reference",
                    scenario="fused-down-silu-reference",
                    source_url=FUSED_REFERENCE_URL,
                )
            )
    return cases


def shape_keys(core=False):
    return [(c["n"], c["hc"], c["h"]) for c in grid(core)]


def key_shape(case):
    return [case["n"], case["h"]]


def case_id(case):
    return f"n{case['n']}_hc{case['hc']}_dim{case['h']}"


def build_inputs(case, dtype=None, device="cuda"):
    dtype = dtype or getattr(torch, case.get("dtype", "bfloat16"))
    n, width = case["n"], case["h"]
    stride = width + case.get("row_gap", 0)
    gen = torch.Generator(device="cpu").manual_seed(case.get("seed", SEED))
    if dtype == torch.bool:
        host = torch.randint(0, 2, (n, stride), generator=gen).to(dtype)
    elif not dtype.is_floating_point and not dtype.is_complex:
        low = 0 if str(dtype).startswith("torch.uint") else -8
        host = torch.randint(low, 9, (n, stride), generator=gen).to(dtype)
    else:
        host = torch.randn((n, stride), generator=gen).to(dtype)
    host = host[:, :width]
    x = torch.empty_strided(host.shape, host.stride(), dtype=dtype, device=device)
    if x.device.type == "musa" and x.dtype in (torch.uint16, torch.uint32):
        x.view(torch.uint8).copy_(host.view(torch.uint8))
    else:
        x.copy_(host)
    return (x, case["hc"]), {}


def config(case, dtype=None):
    dtype = dtype or getattr(torch, case.get("dtype", "bfloat16"))
    n, width = case["n"], case["h"]
    resolved = dict(case, dtype=str(dtype).removeprefix("torch."))
    return {
        "case_id": case_id(case),
        "native_api": NATIVE_API,
        "native_version": NATIVE_VERSION,
        "binding": resolved,
        "workload_source": case.get("source_url"),
        "workload_source_kind": case.get("source_kind", "interface-regression"),
        "workload_scenario": case.get("scenario", "interface-regression"),
        "model_config_reference": MODEL_CONFIG_URL,
        "is_model_trace": False,
        "workload_note": (
            "N=1/3/5/17/48 with DIM=320/stride=336 follows the independent "
            "hc_silu reference call inside upstream test_hc_down_silu_fused; "
            "this is not a standalone upstream hc_silu test. Values are generated "
            "operator inputs, not replayed GEMM outputs. Other token counts are "
            "synthetic. DIM=320/stride=336 also matches the unfused projection "
            "slice; DIM=2048/2560 retain synthetic width coverage based on the "
            "prior 0.29 source hints, not observed hc_silu model inputs."
        ),
        "inputs": {
            "x": {
                "shape": [n, width],
                "dtype": str(dtype),
                "stride": [width + case.get("row_gap", 0), 1],
            },
            "hc_count": {"scalar": case["hc"]},
        },
        "outputs": {
            "y": {"shape": [n, width], "dtype": str(dtype), "stride": [width, 1]}
        },
        "graph": False,
        "compute_dtype": "float32",
        "rounding_boundary": "FP32 scaled SiLU -> input dtype once at output",
        "mutated_inputs": [],
        "alias": "fresh output; no input alias",
    }


def assert_outputs(actual, expected, args):
    x = args[0]
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape == x.shape
    assert actual.dtype == expected.dtype == x.dtype
    assert actual.device == expected.device == x.device
    assert actual.is_contiguous()
    if actual.numel():
        assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    if not x.dtype.is_floating_point:
        assert torch.equal(actual.cpu(), expected.cpu())
        return
    got, ref = actual.detach().cpu().double(), expected.detach().cpu().double()
    tolerance = max(2e-6, 2 * torch.finfo(x.dtype).eps)
    assert torch.equal(torch.isnan(got), torch.isnan(ref))
    assert torch.equal(torch.isposinf(got), torch.isposinf(ref))
    assert torch.equal(torch.isneginf(got), torch.isneginf(ref))
    torch.testing.assert_close(got, ref, atol=tolerance, rtol=tolerance, equal_nan=True)
