# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8 as vllm_per_token_quant_int8,
)

import flaggems_vllm

from . import base


class PerTokenQuantInt8Benchmark(base.GenericBenchmark):
    # FlagGems-sglang/tools/op_sglang_perf_baseline.json (blob ae582b855d9f).
    DEFAULT_SHAPES = [(m, n) for m in (1, 8, 64, 512, 4096) for n in (2048, 4096, 8192)]
    DEFAULT_SHAPE_DESC = "M, N"
    DEFAULT_SHAPE_FILES = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")
    MORE_SHAPES = [
        (1, 128),
        (1, 4096),
        (7, 513),
        (64, 1024),
        (512, 4096),
        (2048, 5120),
        (2048, 8192),
        (4096, 13824),
    ]

    def set_more_shapes(self):
        return self.MORE_SHAPES

    def set_dtypes(self, user_desired_dtypes):
        if (
            not user_desired_dtypes
            and base.Config.bench_level == base.consts.BenchLevel.CORE
        ):
            user_desired_dtypes = [torch.bfloat16]
        super().set_dtypes(user_desired_dtypes)


def _input_fn(shape, dtype, device):
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    yield ((x * 2 - 1).to(dtype),)


@pytest.mark.per_token_quant_int8
def test_per_token_quant_int8():
    bench = PerTokenQuantInt8Benchmark(
        op_name="per_token_quant_int8",
        torch_op=vllm_per_token_quant_int8,
        input_fn=_input_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.set_gems(flaggems_vllm.per_token_quant_int8)
    bench.run()
