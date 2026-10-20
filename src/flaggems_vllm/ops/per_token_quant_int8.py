# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from FlagGems-sglang's per_token_quant_int8 for the vLLM contract.

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("per_token_quant_int8"),
    key=["M", "N", "stride_xm", "stride_xn"],
    rep=20,
)
@triton.jit
def _per_token_quant_int8_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    offsets = row * stride_xm + cols * stride_xn
    if N == BLOCK_N:
        x = tl.load(x_ptr + offsets).to(tl.float32)
    else:
        x = tl.load(x_ptr + offsets, mask=cols < N, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-10)
    scale = absmax / 127.0
    # vLLM rounds halfway cases away from zero instead of truncating the cast.
    xq = libdevice.round(x * (127.0 / absmax)).to(tl.int8)
    if N == BLOCK_N:
        tl.store(xq_ptr + row * N + cols, xq, cache_modifier=".cs")
    else:
        tl.store(xq_ptr + row * N + cols, xq, mask=cols < N, cache_modifier=".cs")
    tl.store(scale_ptr + row, scale, cache_modifier=".cs")


def per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each last-dimension row with vLLM's symmetric INT8 semantics."""
    logger.debug("GEMS PER TOKEN QUANT INT8")
    shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, shape[-1])
    N = x.shape[-1]
    M = x.numel() // N
    xq = torch.empty(shape, device=x.device, dtype=torch.int8)
    scales = torch.empty(shape[:-1] + (1,), device=x.device, dtype=torch.float32)
    if M == 0:
        return xq, scales

    with torch_device_fn.device(x.device):
        _per_token_quant_int8_kernel[(M,)](
            x,
            xq,
            scales,
            M,
            N,
            x.stride(-2),
            x.stride(-1),
            triton.next_power_of_2(N),
        )
    return xq, scales
