# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""HC SiLU timings use the repository's unchanged non-graph benchmark."""

import importlib

import pytest
import torch

import flaggems_vllm as gems
from baseline import hc_silu as baseline
from benchmark import base, conftest, consts


class HCSiluBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, HC, DIM; fused-test reference and synthetic coverage"

    def set_more_shapes(self):
        return baseline.shape_keys()

    def init_user_config(self):
        super().init_user_config()
        if self.mode == consts.BenchMode.CUDAGRAPH:
            raise ValueError("hc_silu measurements must not use CUDA Graph")
        if not set(self.shapes) <= set(baseline.shape_keys()):
            raise ValueError("unknown workload; use baseline.hc_silu.grid() shapes")

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


@pytest.mark.hc_silu
@pytest.mark.parametrize("entry", ("public", "generic"))
def test_hc_silu(entry):
    native = baseline.native()
    if native is None and conftest.Config.base_data is None:
        pytest.skip(f"{baseline.NATIVE_UNAVAILABLE_REASON}; supply --base-data")
    if entry == "generic":
        candidate = importlib.import_module("flaggems_vllm.ops.hc_silu").hc_silu
    else:
        candidate = gems.hc_silu
        if gems.vendor_name == "nvidia" and torch.cuda.get_device_capability()[0] == 9:
            hopper = importlib.import_module(
                "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_silu"
            ).hc_silu
            assert candidate is hopper, "public dispatch did not select Hopper"
    assert candidate is not native
    bench = HCSiluBenchmark(
        op_name=baseline.OP_NAME,
        torch_op=native,
        gems_op=candidate,
        dtypes=baseline.DTYPES,
        is_inplace=baseline.IS_INPLACE,
    )
    bench.entry = entry
    bench.run()
