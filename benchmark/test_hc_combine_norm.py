# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark upstream and synthetic HC workloads with repository non-graph timers."""

import importlib

import pytest
import torch

import flaggems_vllm as gems
from baseline import hc_combine_norm as baseline
from benchmark import base, conftest, consts


class HCCombineNormBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, HC, H, has_logits; upstream + synthetic HC workloads"

    def set_more_shapes(self):
        return baseline.shape_keys()

    def init_user_config(self):
        super().init_user_config()
        if self.mode == consts.BenchMode.CUDAGRAPH:
            raise ValueError(
                "hc_combine_norm baseline measurements must not use CUDA Graph"
            )
        if conftest.Config.base_data is not None:
            shapes = conftest.Config.base_data.get(baseline.OP_NAME, {}).get(
                "shapes", {}
            )
            if any(
                str([case["n"], case["hc"] * case["h"]]) in shapes
                for case in baseline.grid()
            ):
                raise ValueError(
                    "hc_combine_norm legacy residual-shape baseline keys are unsafe; "
                    "recollect with [N, HC, H, has_logits] keys"
                )
        known = set(baseline.shape_keys())
        if not set(self.shapes) <= known:
            raise ValueError(
                "unknown workload; use baseline.hc_combine_norm.grid() shapes"
            )

    def get_input_iter(self, dtype):
        for case in baseline.grid():
            if tuple(baseline.key_shape(case)) not in self.shapes:
                continue
            self.binding = case
            self.binding_dtype = dtype
            args, kwargs = baseline.build_inputs(case, dtype, gems.device)
            yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        return {
            "args": super().record_shapes(*args, **kwargs),
            "config": baseline.config(self.binding, self.binding_dtype),
            "entry": self.entry,
        }


@pytest.mark.hc_combine_norm
@pytest.mark.parametrize("entry", ["public", "generic"])
def test_hc_combine_norm(entry):
    native = baseline.native()
    if native is None and conftest.Config.base_data is None:
        pytest.skip(f"{baseline.NATIVE_UNAVAILABLE_REASON}; supply --base-data")
    if entry == "generic":
        candidate = importlib.import_module(
            "flaggems_vllm.ops.hc_combine_norm"
        ).hc_combine_norm
    else:
        candidate = gems.hc_combine_norm
        if gems.vendor_name == "nvidia" and torch.cuda.get_device_capability()[0] == 9:
            hopper = importlib.import_module(
                "flaggems_vllm.runtime.backend._nvidia.hopper.ops.hc_combine_norm"
            ).hc_combine_norm
            assert candidate is hopper, "public dispatch did not select Hopper"
    assert native is not candidate
    bench = HCCombineNormBenchmark(
        op_name=baseline.OP_NAME,
        torch_op=native,
        gems_op=candidate,
        dtypes=baseline.DTYPES,
        is_inplace=baseline.IS_INPLACE,
    )
    bench.entry = entry
    bench.run()
