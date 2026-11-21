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

"""Native policies for measured streaming pointwise layouts."""

from inspect import signature

import torch
import triton.language as tl

try:
    from triton.experimental.tle.language import gpu
except ImportError:
    gpu = None

_HAS_POLICY = all(
    "cache_policy" in signature(op).parameters for op in (tl.load, tl.store)
)


def _policy(backend, name, *args, **kwargs):
    namespace = gpu if backend is None else getattr(gpu, backend, None)
    constructor = getattr(namespace, name, None)
    return (
        constructor(*args, **kwargs)
        if _HAS_POLICY and constructor is not None
        else None
    )


METAX_CG = _policy("metax", "CachePolicy", "cg")
PPU_LU = _policy("ppu", "CachePolicy", ".lu")
MUSA_PERSIST = _policy(
    "mthreads", "LSUCachePolicy", inner_persistence=3, outer_persistence=3, l2_policy=1
)

POLICIES = {
    "metax_read_cg": ("METAX_CG", None),
    "ppu_read_lu": ("PPU_LU", None),
    "musa_write_persist": (None, "MUSA_PERSIST"),
}

_VENDOR_FAMILIES = {
    "mthreads": {
        "silu_and_mul_kernel",
        "gelu_none_and_mul_kernel",
    },
}


def pointwise_family(fn):
    if not _HAS_POLICY:
        return None
    # Selected scalar kernels only; backward and complex kernels keep their defaults.
    name = fn.__name__
    if any(name in families for families in _VENDOR_FAMILIES.values()):
        return name
    return None


def _contiguous(tensor):
    stride = 1
    for size, actual in zip(reversed(tensor.shape), reversed(tensor.stride())):
        if size != 1 and actual != stride:
            return False
        stride *= size
    return True


def select_policy(family, args, outputs):
    import flaggems_vllm

    if len(outputs) != 1 or outputs["out0"].numel() < 2**22:
        return None
    vendor = flaggems_vllm.vendor_name
    if family not in _VENDOR_FAMILIES.get(vendor, ()):
        return None
    tensors = [arg for arg in args if hasattr(arg, "stride")]
    if len(tensors) != 2:
        return None
    out = outputs["out0"]
    if any(t.dtype != torch.float32 for t in (*tensors, out)) or not _contiguous(out):
        return None
    if any(
        t.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
        for t in tensors
    ):
        return None
    count = out.numel()
    if vendor == "metax" and METAX_CG is not None and 2**23 <= count <= 2**25:
        strides = tuple(v for t in tensors for v in t.stride())
        if strides and 8 <= min(strides) <= 16:
            return "metax_read_cg"
    if vendor == "thead" and PPU_LU is not None and 2**23 <= count <= 2**24:
        if all(_contiguous(t) for t in tensors):
            return "ppu_read_lu"
    if vendor == "mthreads" and MUSA_PERSIST is not None and 2**24 <= count <= 2**25:
        if all(_contiguous(t) for t in tensors):
            return "musa_write_persist"
    return None
