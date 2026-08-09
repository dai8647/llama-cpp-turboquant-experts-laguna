#!/usr/bin/env python3
"""
mixed_quant.py - build a mixed-precision GGUF (ds4-style asymmetric quantization).

The ds4 / DwarfStar approach (antirez): quantize ONLY the routed MoE experts
down to ~2 bits and keep the quality-critical parts (router, shared experts,
attention, projections) at higher precision. On memory-bandwidth-bound decode
(CPU-offloaded experts) this roughly halves the bytes read per generated token.

This script applies that idea using this fork's own fast 2-bit type TQ2_0
(2.06 bpw, same bitrate as IQ2_XXS) with a Q8_0 base. TQ2_0 has optimized
decode kernels in this fork, so it is usually faster than IQ2_XXS/Q2_K.

Default mapping (deepseek2/deepseek4/qwen3moe-style MoE models):

    blk.*.ffn_{gate,up,down}_exps.weight   -> expert-type (default TQ2_0)   [routed experts]
    blk.*.ffn_{gate,up,down}_shexp.weight  -> shared-type (default F16)     [shared experts]
    blk.*.ffn_gate_inp.weight              -> router-type (default F16)     [router]
    everything else                        -> base-type   (default Q8_0)

Usage:
    python3 scripts/mixed_quant.py -h
    python3 scripts/mixed_quant.py model-f16.gguf model-tq2.gguf
    python3 scripts/mixed_quant.py --expert-type TQ1_0 model-f16.gguf model-tq1.gguf
    python3 scripts/mixed_quant.py --print-plan model-f16.gguf /dev/null

Type selection is limited to what this fork's gguf-py can produce: Q4_0,
Q4_1, Q5_0, Q5_1, Q8_0, Q2_K..Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, BF16, TQ1_0, TQ2_0. The IQ
quantizers are unweighted (no imatrix) ports, so for --imatrix parity use
llama-quantize instead.

Requires this fork's gguf-py (auto-added to sys.path) and numpy.
Tensors whose last dim is not a multiple of the target block size fall back to
base-type, then F16, then stay at the source type.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

# Make the fork-local gguf package importable (same trick as gguf-py scripts).
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).resolve().parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

import numpy as np  # noqa: E402

from gguf import (  # noqa: E402
    GGMLQuantizationType,
    GGUFReader,
    GGUFValueType,
    GGUFWriter,
    QuantError,
    dequantize,
    quantize,
)

logger = logging.getLogger("mixed-quant")

# Regex patterns evaluated in order; first match wins.
EXPERT_RE = re.compile(r"(^|\.)blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")
ROUTER_RE = re.compile(r"(^|\.)blk\.\d+\.ffn_gate_inp\.weight$")
SHARED_RE = re.compile(r"(^|\.)blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$")

# Types this fork's gguf-py can actually PRODUCE (quantize implemented).
# K-quants (Q2_K..Q6_K) and the IQ family (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
# IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS) are bit-exact ports of the C
# reference in ggml-quants.c. The IQ ports take no imatrix weights, so the
# output can differ from llama-quantize with --imatrix.
_QUANTIZABLE = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q4_1,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q5_1,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    GGMLQuantizationType.TQ1_0,
    GGMLQuantizationType.TQ2_0,
}


def _valid_type(name: str) -> GGMLQuantizationType:
    try:
        t = GGMLQuantizationType[name]
    except KeyError:
        raise argparse.ArgumentTypeError(f"unknown GGML type: {name}")
    if t not in _QUANTIZABLE:
        raise argparse.ArgumentTypeError(
            f"type {name} cannot be produced by this gguf-py (quantize not implemented); "
            f"choose from {sorted(t_.name for t_ in _QUANTIZABLE)}"
        )
    return t


def _type_name(t: GGMLQuantizationType) -> str:
    return t.name


def resolve_type(name: str, args: argparse.Namespace) -> GGMLQuantizationType:
    """Return the requested target type for a tensor name."""
    if any(re.search(p, name) for p in args.keep_pattern):
        return GGMLQuantizationType.F16  # keep high precision, best effort
    if EXPERT_RE.search(name):
        return args.expert_type
    if ROUTER_RE.search(name):
        return args.router_type
    if SHARED_RE.search(name):
        return args.shared_type
    return args.base_type


def _as_f32(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype in (GGMLQuantizationType.F16, GGMLQuantizationType.F32, GGMLQuantizationType.BF16):
        return dequantize(data, qtype) if qtype != GGMLQuantizationType.F32 else data.astype(np.float32, copy=False)
    return dequantize(data, qtype)


def _fallback_chain(qtype: GGMLQuantizationType, base: GGMLQuantizationType, source: GGMLQuantizationType) -> list[GGMLQuantizationType]:
    """Type ladder: requested -> base -> F16 -> source (pass-through)."""
    chain = [qtype]
    if base != qtype:
        chain.append(base)
    if GGMLQuantizationType.F16 not in chain:
        chain.append(GGMLQuantizationType.F16)
    if source not in chain:
        chain.append(source)
    return chain


def _check_quantizable(qtype: GGMLQuantizationType, label: str) -> None:
    """Fail fast if the requested type cannot be produced by this gguf-py."""
    try:
        quantize(np.zeros((256,), dtype=np.float32), qtype)
    except NotImplementedError:
        sys.exit(f"error: {label} quantization ({qtype.name}) is not implemented in this gguf-py")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build a ds4-style mixed-precision GGUF")
    parser.add_argument("input", type=str, help="input GGUF (F16/BF16 source recommended)")
    parser.add_argument("output", type=str, help="output GGUF path")
    parser.add_argument("--expert-type", type=_valid_type, default=GGMLQuantizationType.TQ2_0,
                        help="routed expert type (default TQ2_0; TQ1_0 and Q2_K are the other 2-bit-class options)")
    parser.add_argument("--base-type", type=_valid_type, default=GGMLQuantizationType.Q8_0,
                        help="type for everything else (default Q8_0; use F16 for max quality)")
    parser.add_argument("--router-type", type=_valid_type, default=GGMLQuantizationType.F16,
                        help="router (ffn_gate_inp) type (default F16)")
    parser.add_argument("--shared-type", type=_valid_type, default=GGMLQuantizationType.F16,
                        help="shared expert type (default F16)")
    parser.add_argument("--keep-pattern", action="append", default=[],
                        metavar="REGEX", help="tensor names matching this regex stay at F16 (repeatable)")
    parser.add_argument("--print-plan", action="store_true", help="print the per-tensor type plan and exit")
    args = parser.parse_args()

    _check_quantizable(args.expert_type, "expert")
    _check_quantizable(args.base_type, "base")
    if args.router_type != GGMLQuantizationType.F16:
        _check_quantizable(args.router_type, "router")
    if args.shared_type != GGMLQuantizationType.F16:
        _check_quantizable(args.shared_type, "shared")

    reader = GGUFReader(args.input)
    # NOTE: the fork's GGUFReader is single-file only; sharded GGUFs must be
    # merged first (e.g. with gguf-split).

    arch = "llama"
    arch_field = reader.fields.get("general.architecture")
    if arch_field is not None:
        arch = str(arch_field.contents())

    # --- plan ---
    plan: list[tuple[str, GGMLQuantizationType, GGMLQuantizationType, tuple[int, ...]]] = []
    for t in reader.tensors:
        plan.append((t.name, t.tensor_type, resolve_type(t.name, args), tuple(int(d) for d in t.shape)))

    if args.print_plan:
        for name, src, dst, _shape in plan:
            mark = " <-" if dst != src else ""
            logger.info("%-60s %-8s -> %-8s%s", name, src.name, dst.name, mark)
        logger.info("total tensors: %d", len(plan))
        return

    # --- write ---
    writer = GGUFWriter(args.output, arch, use_temp_file=False)
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue  # internal pseudo-fields (version/counts); writer emits its own
        main_type = field.types[0]
        sub_type = field.types[-1] if main_type == GGUFValueType.ARRAY and len(field.types) > 1 else None
        writer.add_key_value(key, field.contents(), main_type, sub_type)

    stats: dict[str, int] = {}
    n_fallback = 0
    for idx, (name, src, dst, _shape) in enumerate(plan):
        t = reader.tensors[idx]
        data = t.data
        if src not in _QUANTIZABLE or dst == src:
            # pass through raw bytes (F64/int tensors and same-type hits); the
            # writer derives the logical shape from the byte/array shape
            writer.add_tensor(name, data, raw_shape=data.shape, raw_dtype=src)
            stats[_type_name(src)] = stats.get(_type_name(src), 0) + int(t.n_bytes)
            continue

        f32 = _as_f32(data, src)
        chosen = None
        for cand in _fallback_chain(dst, args.base_type, src):
            try:
                q = quantize(f32, cand)
                chosen = cand
                break
            except (ValueError, QuantError, NotImplementedError) as e:
                logger.debug("tensor %s: %s not applicable (%s), trying next", name, cand.name, e)
        if chosen is None:
            chosen = src
            q = data
            logger.warning("tensor %s: no quant type fits, passing through as %s", name, src.name)
        if chosen != dst:
            n_fallback += 1
        del f32
        writer.add_tensor(name, q, raw_shape=q.shape, raw_dtype=chosen)
        stats[_type_name(chosen)] = stats.get(_type_name(chosen), 0) + int(q.nbytes)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    in_bytes = sum(int(t.n_bytes) for t in reader.tensors)
    out_bytes = sum(stats.values())
    logger.info("output: %s", args.output)
    for tname, nb in sorted(stats.items()):
        logger.info("  %-8s %10.1f MiB", tname, nb / (1024.0 * 1024.0))
    logger.info("  total   %10.1f MiB  (input %10.1f MiB, ratio %.3f)",
                out_bytes / (1024.0 * 1024.0), in_bytes / (1024.0 * 1024.0), in_bytes / max(out_bytes, 1))
    if n_fallback:
        logger.warning("%d tensor(s) fell back to a different type than requested", n_fallback)


if __name__ == "__main__":
    main()
