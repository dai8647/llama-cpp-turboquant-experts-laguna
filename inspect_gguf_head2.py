#!/usr/bin/env python3
"""List GGUF tensor names from a truncated head download or first shard."""
import sys

sys.path.insert(0, "gguf-py")
import numpy as np
from gguf import GGUFReader

_orig_get = GGUFReader._get


def _safe_get(self, offset, type_or_arr, count=None, **kw):
    try:
        if count is None:
            res = _orig_get(self, offset, type_or_arr, **kw)
        else:
            res = _orig_get(self, offset, type_or_arr, count, **kw)
        if count is not None and getattr(res, "size", 0) != count:
            return np.zeros(count, dtype=type_or_arr)
        return res
    except Exception:
        if count is not None:
            return np.zeros(count, dtype=type_or_arr)
        return np.zeros(1, dtype=type_or_arr)


GGUFReader._get = _safe_get

r = GGUFReader(sys.argv[1])
names = [t.name for t in r.tensors]
print("total tensors:", len(names))
hits = [n for n in names if any(k in n for k in ("nextn", "mtp", "dspark"))]
print("nextn/mtp/dspark hits:", len(hits))
for h in hits:
    print(h)
print("--- first 8 names ---")
for n in names[:8]:
    print(n)
