# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixCase:
    group: str
    b: int
    h: int
    t: int
    d: int
    e: int
    layout: str = "contiguous"
    initial: str = "random"
    state_layout: str = "contiguous"
    dtype: torch.dtype = torch.float32
    decay: str = "heads"
    signal: str = "random"
    state_dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class DecodeCase:
    group: str
    b: int
    h: int
    d: int
    slots: str = "random"
    layout: str = "contiguous"
    state_layout: str = "contiguous"
    dtype: torch.dtype = torch.float32
    block: int = 32
    decay: str = "heads"
    metadata_strided: bool = False
    signal: str = "random"
    state_dtype: torch.dtype = torch.float32
    cache_rows: int | None = None


# Native MiniMax-Text-01 on H20, vLLM 0.23, TP8. Select frequent calls,
# chunk boundaries and the measured batch sizes around Hopper dispatch thresholds.
MINIMAX_PREFIX_LENGTHS = (
    2,
    16,
    31,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    512,
    1023,
    1024,
    1025,
    1536,
    2048,
    4095,
    4096,
    4097,
    8191,
    8192,
    12185,
    16383,
    16384,
)
MINIMAX_DECODE_BATCHES = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    24,
    25,
    31,
    32,
    33,
    48,
    63,
    64,
    65,
    96,
    127,
    128,
)
PREFIX_CASES = [
    PrefixCase("minimax-tp8", 1, 8, t, 128, 128, state_dtype=torch.bfloat16)
    for t in MINIMAX_PREFIX_LENGTHS
]
DECODE_CASES = [
    DecodeCase(
        "minimax-tp8",
        b,
        8,
        128,
        slots="linear",
        state_dtype=torch.bfloat16,
        cache_rows=10749,
    )
    for b in MINIMAX_DECODE_BATCHES
]


def case_id(case):
    tag = "p" if isinstance(case, PrefixCase) else "d"
    t = getattr(case, "t", 1)
    e = getattr(case, "e", case.d)
    return f"{tag}-{case.group}-b{case.b}h{case.h}t{t}d{case.d}e{e}"


def make_qkv(case, device):
    b, h, t, d, e = (
        case.b,
        case.h,
        getattr(case, "t", 1),
        case.d,
        getattr(case, "e", case.d),
    )

    def random(*shape):
        return torch.randn(*shape, device=device, dtype=case.dtype).mul_(0.125)

    if case.layout == "packed":
        raw = random(b, t, h * (2 * d + e))
        qkv = tuple(
            x.view(b, t, h, n).transpose(1, 2)
            for x, n in zip(raw.split([h * d, h * d, h * e], -1), (d, d, e))
        )
    else:

        def tensor(width, layout):
            if layout == "token-major":
                return random(b, t, h, width).transpose(1, 2)
            if layout == "offset":
                return random(b, h, t, width + 3)[..., 1 : 1 + width]
            if layout == "stride2":
                return random(b, t, h, width * 2)[..., ::2].transpose(1, 2)
            return random(b, h, t, width)

        layouts = (
            ("stride2", "contiguous", "token-major")
            if case.layout == "mixed"
            else (case.layout,) * 3
        )
        qkv = tuple(tensor(width, layout) for width, layout in zip((d, d, e), layouts))
    if case.signal.startswith("zero-"):
        qkv[{"zero-q": 0, "zero-k": 1, "zero-v": 2}[case.signal]].zero_()
    elif case.signal == "alternating":
        for x in qkv:
            x[..., 1::2].neg_()
        qkv[1][..., ::2].copy_(qkv[1][..., 1::2])
        qkv[0][..., 1::2].copy_(-0.9 * qkv[0][..., ::2])
    return qkv


def make_slope(case, device):
    h = case.h
    if case.group == "minimax-tp8":
        # TP rank 0's measured slopes. All ranks share this tensor layout.
        return torch.tensor(
            [
                0.91701323,
                0.84090483,
                0.77111316,
                0.70711386,
                0.64842629,
                0.59460950,
                0.54525936,
                0.50000501,
            ],
            dtype=torch.float32,
            device=device,
        ).view(h, 1, 1)
    if case.decay == "heads":
        return torch.linspace(0.002, 0.12, h, device=device)
    if case.decay == "steep":
        return torch.linspace(1.0, 8.0, h, device=device)
    if case.decay == "mixed":
        rates = torch.tensor([0.0, 1e-6, 0.125, 3.0], device=device)
        return rates[torch.arange(h, device=device) % 4]
    return torch.full(
        (h,),
        {"zero": 0.0, "tiny": 1e-6, "half": 0.6931471805599453}[case.decay],
        device=device,
    )


def make_state(case, rows, device):
    d, e, h = case.d, getattr(case, "e", case.d), case.h
    layout = case.state_layout
    options = {"device": device, "dtype": case.state_dtype}
    if layout == "head-major":
        return torch.empty(h, rows, d, e, **options).transpose(0, 1)
    if layout == "transposed":
        return torch.empty(rows, h, e, d, **options).transpose(-1, -2)
    if layout == "interleaved":
        return torch.empty(rows, h, d, 2 * e, **options)[..., ::2]
    if layout == "pitched":
        return torch.empty(rows, h, d, e + 7, **options)[..., 2 : 2 + e]
    return torch.empty(rows, h, d, e, **options)


def make_initial(case, device):
    if case.initial == "none":
        return None
    state = make_state(case, case.b, device)
    return state.zero_() if case.initial == "zero" else state.normal_(std=0.125)


def make_decode_inputs(case, device):
    b = case.b
    rows = case.cache_rows or (2 * b + 3 if case.slots == "sparse" else b + 3)
    seq = torch.arange(b, device=device)
    if case.slots == "linear":
        # Exercise the far end of the model-sized allocation (above 2 GiB).
        slots = seq + (rows - b) if case.cache_rows is not None else seq
    elif case.slots == "reverse":
        slots = seq.flip(0)
    elif case.slots == "sparse":
        slots = 2 * seq + 1
    else:
        slots = torch.randperm(rows, device=device)[:b]
        if case.slots == "half-pad":
            slots[1::2] = -1
        elif case.slots == "tail-pad":
            slots[(b + 1) // 2 :] = -1
        elif case.slots == "one-live":
            slots[:-1] = -1
        elif case.slots == "all-pad":
            slots.fill_(-1)
    slots = slots.to(
        torch.int32 if case.cache_rows is not None or b % 2 else torch.int64
    )
    slope = make_slope(case, device)
    if case.metadata_strided:
        slot_storage = torch.full((b, 2), -1, dtype=slots.dtype, device=device)
        slot_storage[:, 0] = slots
        slots = slot_storage[:, 0]
        rate_storage = torch.zeros(case.h, 2, device=device)
        rate_storage[:, 0] = slope
        slope = rate_storage[:, 0]
    return slope, slots, make_state(case, rows, device).normal_(std=0.125)


def native_prefix_reason(case):
    if case.t == 0:
        return "empty sequence"
    if case.e < 16 or case.e & (case.e - 1):
        return "native value tile needs a power of two >=16"
    if case.dtype == torch.bfloat16:
        return "native prefix needs FP32 V with FP32 decay"
    return None


def native_decode_reason(qkv, slope, slots):
    if not all(x.is_contiguous() for x in qkv):
        return "native decode assumes shared QKV strides and unit feature stride"
    if slope.stride(0) != 1 or slots.stride(0) != 1:
        return "native decode ignores slope/slot strides"
    return None
