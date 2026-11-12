# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""HC gated mixing measured by the repository's common non-graph timers."""

import importlib

import pytest
import torch

import flaggems_vllm as gems
from baseline import hc_gate_mix as baseline
from benchmark import base, conftest, consts


class HCGateMixBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, HC, H; upstream test + synthetic decode/prefill"

    def set_more_shapes(self):
        return baseline.shape_keys()

    def init_user_config(self):
        super().init_user_config()
        if self.mode == consts.BenchMode.CUDAGRAPH:
            raise ValueError(
                "hc_gate_mix baseline measurements must not use CUDA Graph"
            )
        if not set(self.shapes) <= set(baseline.shape_keys()):
            raise ValueError("unknown workload; use baseline.hc_gate_mix.grid() shapes")

    def get_input_iter(self, dtype):
        for case in baseline.grid():
            if (case["n"], case["hc"], case["h"]) not in self.shapes:
                continue
            self.binding, self.binding_dtype = case, dtype
            args, kwargs = baseline.build_inputs(case, dtype, device=gems.device)
            yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        return {
            "args": super().record_shapes(*args, **kwargs),
            "config": baseline.config(self.binding, self.binding_dtype),
            "entry": self.entry,
        }


@pytest.mark.hc_gate_mix
@pytest.mark.parametrize("entry", ["public", "generic"])
def test_hc_gate_mix(entry):
    native = baseline.native()
    if native is None and conftest.Config.base_data is None:
        pytest.skip(f"{baseline.NATIVE_UNAVAILABLE_REASON}; supply --base-data")
    if entry == "generic":
        candidate = importlib.import_module("flaggems_vllm.ops.hc_gate_mix").hc_gate_mix
    else:
        candidate = gems.hc_gate_mix
        if gems.vendor_name == "nvidia" and torch.cuda.get_device_capability()[0] == 9:
            hopper = importlib.import_module(
                "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_gate_mix"
            ).hc_gate_mix
            assert candidate is hopper, "public dispatch did not select Hopper"
    assert candidate is not native
    bench = HCGateMixBenchmark(
        op_name=baseline.OP_NAME,
        torch_op=native,
        gems_op=candidate,
        dtypes=baseline.DTYPES,
        is_inplace=baseline.IS_INPLACE,
    )
    bench.entry = entry
    bench.run()
