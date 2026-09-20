#!/usr/bin/env python3
"""Make the thin qwen4exp MTP draft GGUF standalone-loadable.

Input : mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
        32 tensors, all blk.48.*; trunk tensors (token_embd / output / output_hc_*) absent.
Output: mtp-Qwen3.8-Flash-Next-draft-q8_0.gguf
        same draft block, plus token_embd copied from the trunk shard so the
        qwen-next graph_mtp (tok_embd REQUIRED) and the hc-head mixer can run.

Transformations:
  1. rename blk.48.nextn.hc_head_{norm,down,up} -> output_hc_{norm,down,up}
     (the loader creates hc_head_* under the output_* names; the draft GGUF
      nests them under nextn., so flatten)
  2. copy token_embd.weight from trunk shard 1 (Q8_0, [2560, 248320] = 776 MB)
  3. drop the per-block compress_ratios entry for the MTP block? No - kept as is:
     the draft has 49 ratios with ratio 0 at blk.48 (dense), which matches
     graph_mtp's dense attention.

Run:  PYTHONPATH=<repo>/gguf-py python tools/prep-mtp-draft-qwen38.py
"""
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))

import numpy as np  # noqa: E402
from gguf import GGUFReader, GGUFWriter, ReaderTensor  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = r"C:\Users\dai86\.lmstudio\models\ISTA-DASLab\Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
DRAFT = os.path.join(SRC_DIR, "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
TRUNK = os.path.join(SRC_DIR, "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf")
OUT = r"C:\Users\dai86\Downloads\mtp-draft\mtp-Qwen3.8-Flash-Next-draft-q8_0.gguf"

# KV names worth carrying over verbatim (scalar or array, all python-friendly)
KV_KEEP = [
    "general.architecture",
    "general.name",
    "qwen4exp.block_count",
    "qwen4exp.context_length",
    "qwen4exp.embedding_length",
    "qwen4exp.embedding_length_per_layer_input",
    "qwen4exp.attention.head_count",
    "qwen4exp.attention.head_count_kv",
    "qwen4exp.attention.layer_norm_rms_epsilon",
    "qwen4exp.attention.key_length",
    "qwen4exp.attention.value_length",
    "qwen4exp.rope.dimension_count",
    "qwen4exp.rope.dimension_sections",
    "qwen4exp.rope.freq_base",
    "qwen4exp.expert_count",
    "qwen4exp.expert_used_count",
    "qwen4exp.expert_feed_forward_length",
    "qwen4exp.expert_shared_feed_forward_length",
    "qwen4exp.nextn_predict_layers",
    "qwen4exp.ssm.conv_kernel",
    "qwen4exp.ssm.state_size",
    "qwen4exp.ssm.group_count",
    "qwen4exp.ssm.time_step_rank",
    "qwen4exp.ssm.inner_size",
    "qwen4exp.full_attention_interval",
    "qwen4exp.hyper_connection.count",
    "qwen4exp.hyper_connection.low_rank",
    "qwen4exp.attention.indexer.head_count",
    "qwen4exp.attention.indexer.key_length",
    "qwen4exp.attention.indexer.top_k",
    "qwen4exp.attention.compress_ratios",
    "general.type",
    "general.sampling.temp",
    "general.sampling.top_k",
    "general.sampling.top_p",
]
# tokenizer keys copied from the draft itself (it already carries the full vocab)
TOK_KEEP = [k for k in GGUFReader(DRAFT).fields if k.startswith("tokenizer.") or k == "general.file_type"]


def kv_array(reader: GGUFReader, name: str):
    """Return a python list for an array KV: strings as str, scalars as a flat list."""
    kv = reader.fields[name]
    if len(kv.types) > 1 and kv.types[1] == 8:  # array of strings
        return [bytes(kv.parts[i]).decode("utf-8", "replace") for i in kv.data]
    arr = np.concatenate([kv.parts[i] for i in kv.data])
    # flatten one level of nesting (rope_sections is written as [[11],[11],[10],[0]])
    if arr.dtype == object:
        arr = np.concatenate([np.asarray(x).ravel() for x in arr])
    return arr.tolist()


def add_kv(w: GGUFWriter, reader: GGUFReader, name: str) -> None:
    kv = reader.fields.get(name)
    if kv is None:
        return
    if kv.types[0] == 8:  # string
        w.add_string(name, bytes(kv.parts[kv.data[0]]).decode("utf-8", "replace"))
    elif kv.types[0] == 9:  # array (of scalars or strings)
        w.add_array(name, kv_array(reader, name))
    else:
        val = kv.parts[kv.data[0]]
        # GGUF v3 scalar types: 0=uint8 1=int8 2=uint16 3=int16 4=uint32 5=int32
        #                       6=float32 7=bool 10=uint64 11=int64 12=float64
        fn = {
            0: w.add_uint8, 1: w.add_int8,
            2: w.add_uint16, 3: w.add_int16,
            4: w.add_uint32, 5: w.add_int32,
            6: w.add_float32, 7: w.add_bool,
            10: w.add_uint64, 11: w.add_int64, 12: w.add_float64,
        }.get(kv.types[0])
        if fn is None:
            raise RuntimeError(f"unsupported kv type {kv.types[0]} for {name}")
        fn(name, val.item() if hasattr(val, "item") else val)


def main() -> None:
    print(f"reading {DRAFT}")
    dr = GGUFReader(DRAFT)
    print(f"reading {TRUNK}")
    tr = GGUFReader(TRUNK)

    tok_embd = next(t for t in tr.tensors if t.name == "token_embd.weight")
    print(f"trunk token_embd: {tok_embd.tensor_type} {tok_embd.shape}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    w = GGUFWriter(OUT, arch=dr.fields["general.architecture"].parts[
        dr.fields["general.architecture"].data[0]].tobytes().decode().rstrip("\x00"))

    # --- metadata ---
    # The thin draft GGUF carries nearly all qwen4exp hyperparameters itself (verified:
    # context_length, embedding dims, rope, ssm, hc, indexer, compress_ratios, block_count=49
    # including the MTP block). The original bug was only that KV_KEEP omitted
    # context_length / embedding_length_per_layer_input. Read from the draft (dr), falling
    # back to the trunk (tr) for the few keys the thin draft does not carry.
    for name in KV_KEEP:
        if name in dr.fields:
            add_kv(w, dr, name)
        else:
            add_kv(w, tr, name)
    for name in TOK_KEEP:
        add_kv(w, dr, name)
    w.add_string("general.name", "Qwen3.8-Flash-Next-MTP-draft")

    # --- tensors: draft block (renamed) + token_embd ---
    n_copied = 0
    for t in dr.tensors:  # type: ReaderTensor
        name = t.name
        if name in ("blk.48.nextn.hc_head_norm.weight",
                    "blk.48.nextn.hc_head_down.weight",
                    "blk.48.nextn.hc_head_up.weight"):
            # loader expects output_hc_{norm,down,up}; the draft nests them under nextn.hc_head_
            name = name.replace("blk.48.nextn.hc_head_", "output_hc_")
        data = bytes(t.data)
        tensor_bytes = np.frombuffer(data, dtype=np.uint8)
        if len(t.shape) == 3:
            row_count = math.prod(t.shape[1:])
            raw_shape = (t.shape[2], t.shape[1], len(data) // row_count)
            w.add_tensor(name, tensor_bytes, raw_shape=raw_shape, raw_dtype=t.tensor_type)
        elif len(t.shape) == 2:
            w.add_tensor(name, tensor_bytes,
                         raw_shape=(t.shape[1], len(data) // t.shape[1]),
                         raw_dtype=t.tensor_type)
        else:
            w.add_tensor(name, tensor_bytes, raw_dtype=t.tensor_type)
        n_copied += 1
        if n_copied % 8 == 0:
            print(f"  {n_copied}/{len(dr.tensors) + 1} tensors")

    # token_embd: raw Q8_0 bytes straight from the trunk shard
    te_data = bytes(tok_embd.data)
    w.add_tensor("token_embd.weight",
                 np.frombuffer(te_data, dtype=np.uint8),
                 raw_shape=(tok_embd.shape[1], len(te_data) // tok_embd.shape[1]),
                 raw_dtype=tok_embd.tensor_type)
    n_copied += 1
    print(f"  {n_copied}/{len(dr.tensors) + 1} tensors")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    size = os.path.getsize(OUT)
    print(f"wrote {OUT} ({size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
