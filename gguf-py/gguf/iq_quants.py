# IQ-family quantization.
#
# Bit-exact ports of the reference implementations in ggml/src/ggml-quants.c:
#   IQ2_XXS  quantize_iq2_xxs (imatrix weights required in C; ones here by default)
#   IQ2_XS   quantize_iq2_xs  (imatrix weights required in C; ones here by default)
#   IQ2_S    quantize_row_iq2_s_ref (weight = 0.25*sigma2 + xb^2 when no weights)
#   IQ3_XXS  quantize_row_iq3_xxs_ref (weight = xb^2 when no weights)
#   IQ3_S    quantize_row_iq3_s_ref  (weight = xb^2 when no weights)
#   IQ1_S    quantize_iq1_s  (imatrix weights required in C; ones here by default)
#   IQ1_M    quantize_iq1_m  (weight = xb^2 when no weights)
#   IQ4_NL   quantize_row_iq4_nl_ref (ntry = -1, weight = xb^2)
#   IQ4_XS   quantize_row_iq4_xs_ref (ntry = 7, weight = xb^2)
#
# The grid search tables (grid positions, map, neighbour lists) are rebuilt the
# same way as iq2xs_init_impl / iq3xs_init_impl in ggml-quants.c, but derived
# from the grid_hex payloads already decoded for dequantization.
#
# This module lives separately from quants.py (the dequantize-only IQ classes
# are defined there); importing it patches quantize_blocks onto those classes.
# gguf/__init__.py imports it right after quants.
from __future__ import annotations

from math import ceil, log2

import numpy as np

from .constants import QK_K
from .quants import IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS


def _iq_seq_sum(a: np.ndarray) -> np.ndarray:
    # sequential float32 accumulation like a scalar C loop (np.sum is pairwise)
    return np.cumsum(a, axis=-1, dtype=np.float32)[..., -1]


def _iq_pack_bits(cls) -> tuple[np.ndarray, np.ndarray, int, int]:
    # decode grid_hex into the raw per-position values of the grid
    bits = ceil(log2(len(cls.grid_map)))
    elems_per_byte = 8 // bits
    raw = np.frombuffer(cls.grid_hex, dtype=np.uint8)
    raw = raw.reshape((-1, 2))
    raw = (np.where(raw > 0x40, raw + 9, raw) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
    raw = (raw[..., 0] | raw[..., 1]).reshape((-1, 1))
    vals = raw >> np.array(list(range(0, 8, 8 // elems_per_byte)), dtype=np.uint8).reshape((1, -1))
    vals = (vals & ((1 << bits) - 1)).reshape(cls.grid_shape).astype(np.uint8)
    npos = cls.grid_shape[-1]
    u = np.zeros(cls.grid_shape[0], dtype=np.int64)
    for i in range(npos):
        u |= vals[:, i].astype(np.int64) << (bits * i)
    return vals, u, bits, npos


_iq_search_tables: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _iq_search_tables_for(cls, kmap_size: int, nwant: int):
    # replicate iq2xs_init_impl / iq3xs_init_impl: grid positions, map, neighbour lists
    # keyed by the grid payload so classes sharing a grid (IQ1_S/IQ1_M) build once
    key = (cls.grid_hex, kmap_size, nwant)
    if key in _iq_search_tables:
        return _iq_search_tables[key]
    vals, u, bits, npos = _iq_pack_bits(cls)
    pos = (2 * vals.astype(np.int16) + 1)                    # int8 positions, 2l+1
    kmap = np.full(kmap_size, -1, dtype=np.int32)
    kmap[u] = np.arange(len(u), dtype=np.int32)
    missing = np.nonzero(kmap < 0)[0]
    rows = []
    for s in range(0, len(missing), 1024):
        us = missing[s:s + 1024]
        digs = np.zeros((len(us), npos), dtype=np.int16)
        for i in range(npos):
            digs[:, i] = (us >> (bits * i)) & ((1 << bits) - 1)
        ppos = 2 * digs + 1
        d2 = ((pos[None, :, :].astype(np.int32) - ppos[:, None, :]) ** 2).sum(axis=-1)
        order = np.argsort(d2, axis=1, kind="stable")
        sd2 = np.take_along_axis(d2, order, axis=1)
        bound = np.concatenate([np.ones((len(us), 1), dtype=bool), sd2[:, 1:] != sd2[:, :-1]], axis=1)
        keep = np.cumsum(bound, axis=1) - 1 < nwant
        rows.append(np.where(keep, order, -1))
    nb_idx = np.concatenate(rows, axis=0)
    max_n = int((nb_idx >= 0).sum(axis=1).max())
    nb_idx = nb_idx[:, :max_n]
    kmap[missing] = -(np.arange(len(missing), dtype=np.int32) + 1)
    t = (pos, kmap, nb_idx, (nb_idx >= 0).sum(axis=1))
    _iq_search_tables[key] = t
    return t


def _iq_pack_code(L: np.ndarray, bits: int) -> np.ndarray:
    npos = L.shape[-1]
    out = np.zeros(L.shape[:-1], dtype=np.int64)
    for i in range(npos):
        out |= L[..., i].astype(np.int64) << (bits * i)
    return out


def _iq_neighbour_L(pos, kmap, nb_idx, u, xval, waux, scale, npos=None):
    # iq2_find_best_neighbour / iq3_find_best_neighbour vectorized over rows.
    # u: (R,) int; xval, waux: (R, npos) float32; scale: (R,) float32
    # returns (L, bad): L is (R, npos) int32 (0 for on-grid rows)
    if npos is None:
        npos = xval.shape[-1]
    m = kmap[u]
    bad = m < 0
    L = np.zeros((len(u), npos), dtype=np.int32)
    if bad.any():
        r = -m[bad] - 1
        nb = nb_idx[r]
        ok = nb >= 0
        gv = pos[np.clip(nb, 0, None)].astype(np.float32)
        diff = scale[bad][:, None, None] * gv - xval[bad][:, None, :]
        d2 = _iq_seq_sum(waux[bad][:, None, :] * diff * diff)
        d2 = np.where(ok, d2, np.float32(np.inf))
        j = np.argmin(d2, axis=1)
        g = gv[np.arange(gv.shape[0]), j].astype(np.int32)
        L[bad] = (g - 1) // 2
    return L, bad


def _iq_neighbour_idx(pos, kmap, nb_idx, u, xval, weight, scale, xx, npos):
    # iq1_find_best_neighbour2 vectorized over rows (d2 over the grid values of
    # xx = x_p/x_m with the full weight); returns the best grid index per row.
    # xx: (R, 3) float32
    m = kmap[u]
    bad = m < 0
    idx = m.copy()
    if bad.any():
        r = -m[bad] - 1
        nb = nb_idx[r]
        ok = nb >= 0
        gv = pos[np.clip(nb, 0, None)]                          # (B, max_n, npos)
        q = np.take_along_axis(xx[bad][:, None, :], (gv - 1) // 2, axis=-1).astype(np.float32)
        diff = scale[bad][:, None, None] * q - xval[bad][:, None, :]
        d2 = _iq_seq_sum(weight[bad][:, None, :] * diff * diff)
        d2 = np.where(ok, d2, np.float32(np.inf))
        j = np.argmin(d2, axis=1)
        idx[bad] = nb[np.arange(nb.shape[0]), j]
    return idx


def _iq_signs(xb: np.ndarray, weight: np.ndarray, parity: bool, mask7: bool):
    # sign flips of the 8-vectors, returns (xval, block_signs)
    # xb, weight: (G, m, 8) float32; block_signs: (G, m) uint8
    neg = xb < 0
    xval = np.where(neg, -xb, xb)
    s = np.zeros(xb.shape[:-1], dtype=np.uint8)
    for i in range(8):
        s |= neg[..., i].astype(np.uint8) << np.uint8(i)
    if parity:
        # odd number of flips: flip the element with the smallest weight*xb^2
        wxb = (weight * xb) * xb
        imin = np.argmin(wxb, axis=-1)                          # first occurrence
        odd = (neg.sum(axis=-1) % 2) == 1
        flip = np.arange(8).reshape(1, 1, 8) == imin[..., None]
        xval = np.where(odd[..., None] & flip, -xval, xval)
        s ^= np.where(odd, np.uint8(1) << imin.astype(np.uint8), np.uint8(0))
    if mask7:
        s = s & np.uint8(127)
    return xval, s


def _iq_make_qp_quants(x: np.ndarray, nmax: int, w: np.ndarray) -> np.ndarray:
    # port of make_qp_quants in ggml-quants.c, vectorized over rows
    # x: (G, n) float32 (non-negative); w: (G, n) float32; returns scale per row
    n = x.shape[-1]
    maxv = x.max(axis=-1)
    zero = maxv < np.float32(1e-15)
    denom = np.where(zero, np.float32(1), maxv)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        iscale = np.float32(nmax) / denom
        scale = np.where(zero, np.float32(0), np.float32(1) / iscale)
        Lf = np.rint(iscale[:, None] * x).astype(np.float32)
        best_mse = _iq_seq_sum(w * (x - scale[:, None] * Lf) ** 2)
        for is_ in range(-4, 5):
            if is_ == 0:
                continue
            iscale_is = (np.float32(0.1) * is_ + np.float32(nmax)) / denom
            scale_is = np.float32(1) / iscale_is
            l = np.minimum(nmax, np.rint(iscale_is[:, None] * x).astype(np.int32)).astype(np.float32)
            mse = _iq_seq_sum(w * (x - scale_is[:, None] * l) ** 2)
            better = mse < best_mse
            best_mse = np.where(better, mse, best_mse)
            iscale = np.where(better, iscale_is, iscale)
        L = np.minimum(nmax, np.rint(iscale[:, None] * x).astype(np.int32)).astype(np.float32)
        sumlx = _iq_seq_sum(w * x * L)
        suml2 = _iq_seq_sum(w * L * L)
        for itry in range(5):
            n_changed = 0
            for i in range(n):
                wi = w[:, i]
                xi = x[:, i]
                Li = L[:, i]
                slx = sumlx - wi * xi * Li
                sl2 = suml2 - wi * Li * Li
                pos = (slx > 0) & (sl2 > 0)
                new_l = np.minimum(nmax, np.rint(xi * sl2 / slx)).astype(np.int32)
                changed = pos & (new_l != L[:, i].astype(np.int32))
                nlf = new_l.astype(np.float32)
                slx2 = slx + wi * xi * nlf
                sl22 = sl2 + wi * nlf * nlf
                acc = changed & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
                L[:, i] = np.where(acc, nlf, L[:, i])
                sumlx = np.where(acc, slx2, sumlx)
                suml2 = np.where(acc, sl22, suml2)
                n_changed += int(acc.sum())
            if n_changed == 0:
                break
        out = np.where((suml2 > 0) & ~zero, sumlx / suml2, np.float32(0))
    return out


def _iq_best_index_int8(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    # port of best_index_int8 in ggml-quants.c (values: 16 int8 sorted ascending)
    v = values.astype(np.float32)
    out = np.zeros(x.shape, dtype=np.int32)
    in_range = (x > v[0]) & (x < v[-1])
    mid = np.searchsorted(v, x, side="right")
    ml = np.maximum(mid - 1, 0)
    mu = np.minimum(mid, 15)
    res = np.where(x - v[ml] < v[mu] - x, ml, mu)
    return np.where(in_range, res, np.where(x >= v[-1], 15, 0))


# ===================================================================================
# IQ2_XXS (2.0625 bpw) - 8 groups of 32 per block, 4 grid codes of 8 per group
# ===================================================================================

def _quantize_iq2_xxs(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    n_groups = QK_K // 32
    G = n_blocks * n_groups
    xg = x.reshape(G, 32)
    sigma2 = _iq_seq_sum(x * x) / np.float32(QK_K)
    sig2 = np.repeat(sigma2, n_groups)
    wq = np.ones_like(xg) if weights is None else weights.reshape(G, 32)
    weight = wq * np.sqrt(sig2[:, None] + xg * xg)
    waux = np.sqrt(weight)
    xval, block_signs = _iq_signs(xg.reshape(G, 4, 8), weight.reshape(G, 4, 8), True, True)
    maxv = xval.max(axis=(1, 2))
    zero = maxv < np.float32(1e-15)
    pos, kmap, nb_idx, _ = _iq_search_tables_for(cls, 43692, 2)

    scale = _iq_make_qp_quants(xg, 4, weight)
    eff_max = scale * np.float32(3)
    best = np.zeros(G, dtype=np.float32)
    L = np.zeros((G, 32), dtype=np.int32)
    denom = np.where(zero, np.float32(1), eff_max)
    xv = xval.reshape(G, 4, 8)
    wv = waux.reshape(G, 4, 8)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for is_ in range(-6, 7):
            id = (np.float32(5) + np.float32(0.1) * is_) / denom
            this_scale = np.float32(1) / id
            lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
            lu = np.clip(lu, 0, 2).astype(np.int32)
            uk = _iq_pack_code(lu, 2)
            Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 8), wv.reshape(-1, 8), np.repeat(this_scale, 4))
            Ln = Ln.reshape(G, 4, 8)
            bad = bad.reshape(G, 4)
            Laux = np.where(bad[..., None], Ln, lu)
            Lauxf = Laux.reshape(G, 32).astype(np.float32)
            q = np.float32(2) * Lauxf + np.float32(1)
            sumqx = _iq_seq_sum(weight * xg * q)
            sumq2 = _iq_seq_sum(weight * q * q)
            better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            scale = np.where(better, sumqx / sumq2, scale)
            best = np.where(better, scale * sumqx, best)
            L = np.where(better[:, None], Laux.reshape(G, 32), L)

        # final refinement pass: re-quantize with the winning scale, snap to the grid
        pos_scale = scale > 0
        id = np.float32(1) / np.where(pos_scale, scale, np.float32(1))
        lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
        lu = np.clip(lu, 0, 2).astype(np.int32)
        uk = _iq_pack_code(lu, 2)
        Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 8), wv.reshape(-1, 8), np.repeat(scale, 4))
        Ln = Ln.reshape(G, 4, 8)
        bad = bad.reshape(G, 4)
        Lf = np.where(bad[..., None], Ln, lu).reshape(G, 32)
        L = np.where(pos_scale[:, None], Lf, L)
        Lf2 = L.astype(np.float32)
        q = np.float32(2) * Lf2 + np.float32(1)
        sumqx = _iq_seq_sum(weight * xg * q)
        sumq2 = _iq_seq_sum(weight * q * q)
        scale = np.where(pos_scale & (sumq2 > 0), sumqx / sumq2, scale)

        # scale sign flip (should never happen) and packing
        neg = scale < 0
        scale = np.abs(scale)
        block_signs = np.where(neg[:, None], (~block_signs) & np.uint8(127), block_signs)

        uk2 = _iq_pack_code(L.reshape(G, 4, 8), 2)
        grid_index = kmap[uk2]
        assert (grid_index >= 0).all()
        q2a = (grid_index.astype(np.int64) << (8 * np.arange(4)).reshape(1, 4)).sum(axis=-1)
        q2b = (block_signs.astype(np.int64) << (7 * np.arange(4)).reshape(1, 4)).sum(axis=-1)

        scales_g = scale.reshape(n_blocks, n_groups)
        max_scale = scales_g.max(axis=-1)
        d = max_scale / np.float32(31)
        dh = d[:, None].astype(np.float16).view(np.uint8)
        idb = np.float32(1) / np.where(d > 0, d, np.float32(1))
        l = np.rint(np.float32(0.5) * (idb[:, None] * scales_g - np.float32(1)))
        l = np.clip(l, 0, 15).astype(np.uint32)
        q2b = q2b.reshape(n_blocks, n_groups) | (l << np.uint32(28))
        q2a = q2a.reshape(n_blocks, n_groups)
        q2 = np.stack([q2a, q2b], axis=-1).reshape(n_blocks, 16).astype(np.uint32)
        qs = q2.view(np.uint8).reshape(n_blocks, 64)
        return np.concatenate([dh, qs], axis=-1)


# ===================================================================================
# IQ2_XS (2.3125 bpw) / IQ2_S (3.0625 bpw) - 16 groups of 16, 2 grid codes of 8
# ===================================================================================

def _quantize_iq2_xs(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    n_groups = QK_K // 16
    G = n_blocks * n_groups
    xg = x.reshape(G, 16)
    sigma2 = _iq_seq_sum(x * x) / np.float32(QK_K)
    sig2 = np.repeat(sigma2, n_groups)
    wq = np.ones_like(xg) if weights is None else weights.reshape(G, 16)
    weight = wq * np.sqrt(sig2[:, None] + xg * xg)
    waux = np.sqrt(weight)
    xval, block_signs = _iq_signs(xg.reshape(G, 2, 8), weight.reshape(G, 2, 8), True, True)
    maxv = xval.max(axis=(1, 2))
    zero = maxv < np.float32(1e-15)
    pos, kmap, nb_idx, _ = _iq_search_tables_for(cls, 43692, 2)

    best = np.zeros(G, dtype=np.float32)
    scale = maxv / np.float32(5)
    L = np.zeros((G, 16), dtype=np.int32)
    is_on_grid = np.ones((G, 2), dtype=bool)
    denom = np.where(zero, np.float32(1), maxv)
    xv = xval.reshape(G, 2, 8)
    wv = waux.reshape(G, 2, 8)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for is_ in range(-9, 10):
            id = (np.float32(5) + np.float32(0.1) * is_) / denom
            this_scale = np.float32(1) / id
            lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
            lu = np.clip(lu, 0, 2).astype(np.int32)
            uk = _iq_pack_code(lu, 2)
            Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 8), wv.reshape(-1, 8), np.repeat(this_scale, 2))
            Ln = Ln.reshape(G, 2, 8)
            bad = bad.reshape(G, 2)
            Laux = np.where(bad[..., None], Ln, lu)
            Lauxf = Laux.reshape(G, 16).astype(np.float32)
            q = np.float32(2) * Lauxf + np.float32(1)
            sumqx = _iq_seq_sum(weight * xg * q)
            sumq2 = _iq_seq_sum(weight * q * q)
            better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            scale = np.where(better, sumqx / sumq2, scale)
            best = np.where(better, scale * sumqx, best)
            L = np.where(better[:, None], Laux.reshape(G, 16), L)
            is_on_grid = np.where(better[:, None], ~bad, is_on_grid)

        # second pass: re-quantize the groups that were not on the grid
        do = (~is_on_grid).any(axis=1) & (scale > 0)
        id = np.float32(1) / np.where(do, scale, np.float32(1))
        lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
        lu = np.clip(lu, 0, 2).astype(np.int32)
        L = np.where((~is_on_grid)[..., None] & do[:, None, None], lu, L.reshape(G, 2, 8))
        uk = _iq_pack_code(lu, 2)
        Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 8), wv.reshape(-1, 8), np.repeat(scale, 2))
        Ln = Ln.reshape(G, 2, 8)
        bad = bad.reshape(G, 2)
        L = np.where((~is_on_grid)[..., None] & bad[..., None] & do[:, None, None], Ln, L)
        Lf = L.reshape(G, 16).astype(np.float32)
        q = np.float32(2) * Lf + np.float32(1)
        sumqx = _iq_seq_sum(weight * xg * q)
        sumq2 = _iq_seq_sum(weight * q * q)
        scale = np.where(do & (sumq2 > 0), sumqx / sumq2, scale)

        neg = scale < 0
        scale = np.abs(scale)
        block_signs = np.where(neg[:, None], (~block_signs) & np.uint8(127), block_signs)

        uk2 = _iq_pack_code(L.reshape(G, 2, 8), 2)
        grid_index = kmap[uk2]
        assert (grid_index >= 0).all()
        q2 = (grid_index.astype(np.int64) | (block_signs.astype(np.int64) << np.int64(9))).astype(np.uint16)
        qs = q2.reshape(n_blocks, 32).view(np.uint8).reshape(n_blocks, 64)

        scales_g = scale.reshape(n_blocks, n_groups)
        max_scale = scales_g.max(axis=-1)
        d = max_scale / np.float32(31)
        dh = d[:, None].astype(np.float16).view(np.uint8)
        idb = np.float32(1) / np.where(d > 0, d, np.float32(1))
        l = np.rint(np.float32(0.5) * (idb[:, None] * scales_g - np.float32(1)))
        l = np.clip(l, 0, 15).astype(np.uint8)
        sc = (l[:, 0::2] | (l[:, 1::2] << np.uint8(4))).astype(np.uint8)
        return np.concatenate([dh, qs, sc], axis=-1)


def _quantize_iq2_s(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    n_groups = QK_K // 16
    G = n_blocks * n_groups
    xg = x.reshape(G, 16)
    sigma2 = np.float32(2) * _iq_seq_sum(x * x) / np.float32(QK_K)
    sig2 = np.repeat(sigma2, n_groups)
    if weights is None:
        weight = np.float32(0.25) * sig2[:, None] + xg * xg
    else:
        weight = weights.reshape(G, 16) * np.sqrt(sig2[:, None] + xg * xg)
    waux = np.sqrt(weight)
    xval, block_signs = _iq_signs(xg.reshape(G, 2, 8), weight.reshape(G, 2, 8), False, False)
    maxv = xval.max(axis=(1, 2))
    zero = maxv < np.float32(1e-8)
    pos, kmap, nb_idx, _ = _iq_search_tables_for(cls, 43692, 2)

    best = np.zeros(G, dtype=np.float32)
    scale = maxv / np.float32(5)
    L = np.zeros((G, 16), dtype=np.int32)
    is_on_grid = np.ones((G, 2), dtype=bool)
    denom = np.where(zero, np.float32(1), maxv)
    xv = xval.reshape(G, 2, 8)
    wv = waux.reshape(G, 2, 8)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for is_ in range(-9, 10):
            id = (np.float32(5) + np.float32(0.1) * is_) / denom
            this_scale = np.float32(1) / id
            lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
            lu = np.clip(lu, 0, 2).astype(np.int32)
            uk = _iq_pack_code(lu, 2)
            Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 8), wv.reshape(-1, 8), np.repeat(this_scale, 2))
            Ln = Ln.reshape(G, 2, 8)
            bad = bad.reshape(G, 2)
            Laux = np.where(bad[..., None], Ln, lu)
            Lauxf = Laux.reshape(G, 16).astype(np.float32)
            q = np.float32(2) * Lauxf + np.float32(1)
            sumqx = _iq_seq_sum(weight * xg * q)
            sumq2 = _iq_seq_sum(weight * q * q)
            better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            scale = np.where(better, sumqx / sumq2, scale)
            best = np.where(better, scale * sumqx, best)
            L = np.where(better[:, None], Laux.reshape(G, 16), L)
            is_on_grid = np.where(better[:, None], ~bad, is_on_grid)

        do = (~is_on_grid).any(axis=1) & (scale > 0)
        id = np.float32(1) / np.where(do, scale, np.float32(1))
        lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
        lu = np.clip(lu, 0, 2).astype(np.int32)
        L = np.where((~is_on_grid)[..., None] & do[:, None, None], lu, L.reshape(G, 2, 8))
        uk = _iq_pack_code(lu, 2)
        Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 8), wv.reshape(-1, 8), np.repeat(scale, 2))
        Ln = Ln.reshape(G, 2, 8)
        bad = bad.reshape(G, 2)
        L = np.where((~is_on_grid)[..., None] & bad[..., None] & do[:, None, None], Ln, L)
        Lf = L.reshape(G, 16).astype(np.float32)
        q = np.float32(2) * Lf + np.float32(1)
        sumqx = _iq_seq_sum(weight * xg * q)
        sumq2 = _iq_seq_sum(weight * q * q)
        scale = np.where(do & (sumq2 > 0), sumqx / sumq2, scale)

        neg = scale < 0
        scale = np.abs(scale)
        block_signs = np.where(neg[:, None], ~block_signs, block_signs)

        uk2 = _iq_pack_code(L.reshape(G, 2, 8), 2)
        grid_index = kmap[uk2]
        assert (grid_index >= 0).all()
        gi = grid_index.reshape(n_blocks, 16, 2)
        i8 = 2 * np.arange(16)[:, None] + np.arange(2)[None, :]
        qs = np.zeros((n_blocks, 64), dtype=np.uint8)
        qs[:, :32] = (gi & 255).astype(np.uint8).reshape(n_blocks, 32)
        qs[:, 32:] = block_signs.reshape(n_blocks, 32)
        qh = np.zeros((n_blocks, 8), dtype=np.uint8)
        val = ((gi >> 8).astype(np.uint8) << (2 * (i8 % 4)).astype(np.uint8)[None, :, :]).reshape(n_blocks, 32)
        r = np.tile(np.arange(n_blocks)[:, None], (1, 32))
        c = np.tile((i8 // 4).reshape(-1)[None, :], (n_blocks, 1))
        np.add.at(qh, (r, c), val)

        scales_g = scale.reshape(n_blocks, n_groups)
        max_scale = scales_g.max(axis=-1)
        d = (max_scale / np.float32(31)) * np.float32(0.9875)
        dh = d[:, None].astype(np.float16).view(np.uint8)
        idb = np.float32(1) / np.where(d > 0, d, np.float32(1))
        l = np.rint(np.float32(0.5) * (idb[:, None] * scales_g - np.float32(1)))
        l = np.clip(l, 0, 15).astype(np.uint8)
        sc = (l[:, 0::2] | (l[:, 1::2] << np.uint8(4))).astype(np.uint8)
        return np.concatenate([dh, qs, qh, sc], axis=-1)


# ===================================================================================
# IQ3_XXS (3.4375 bpw) / IQ3_S (4.4375 bpw) - 8 groups of 32, 8 grid codes of 4
# ===================================================================================

def _quantize_iq3_xxs(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    n_groups = QK_K // 32
    G = n_blocks * n_groups
    xg = x.reshape(G, 32)
    sigma2 = np.float32(2) * _iq_seq_sum(x * x) / np.float32(QK_K)
    sig2 = np.repeat(sigma2, n_groups)
    if weights is None:
        weight = xg * xg
    else:
        weight = weights.reshape(G, 32) * np.sqrt(sig2[:, None] + xg * xg)
    waux = np.sqrt(weight)
    # only the first 4 k-groups get sign bits (the 8 k-groups pair up)
    xval, block_signs = _iq_signs(xg.reshape(G, 4, 8), weight.reshape(G, 4, 8), True, True)
    maxv = xval.max(axis=(1, 2))
    zero = maxv < np.float32(1e-8)
    pos, kmap, nb_idx, _ = _iq_search_tables_for(cls, 4096, 2)

    best = np.zeros(G, dtype=np.float32)
    scale = maxv / np.float32(15)
    L = np.zeros((G, 32), dtype=np.int32)
    is_on_grid = np.ones((G, 8), dtype=bool)
    denom = np.where(zero, np.float32(1), maxv)
    xv = xval.reshape(G, 8, 4)
    wv = waux.reshape(G, 8, 4)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for is_ in range(-15, 16):
            id = (np.float32(15) + np.float32(0.2) * is_) / denom
            this_scale = np.float32(1) / id
            lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
            lu = np.clip(lu, 0, 7).astype(np.int32)
            uk = _iq_pack_code(lu, 3)
            Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 4), wv.reshape(-1, 4), np.repeat(this_scale, 8))
            Ln = Ln.reshape(G, 8, 4)
            bad = bad.reshape(G, 8)
            Laux = np.where(bad[..., None], Ln, lu)
            Lauxf = Laux.reshape(G, 32).astype(np.float32)
            q = np.float32(2) * Lauxf + np.float32(1)
            sumqx = _iq_seq_sum(weight * xg * q)
            sumq2 = _iq_seq_sum(weight * q * q)
            better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            scale = np.where(better, sumqx / sumq2, scale)
            best = np.where(better, scale * sumqx, best)
            L = np.where(better[:, None], Laux.reshape(G, 32), L)
            is_on_grid = np.where(better[:, None], ~bad, is_on_grid)

        do = (~is_on_grid).any(axis=1) & (scale > 0)
        id = np.float32(1) / np.where(do, scale, np.float32(1))
        lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
        lu = np.clip(lu, 0, 7).astype(np.int32)
        L = np.where((~is_on_grid)[..., None] & do[:, None, None], lu, L.reshape(G, 8, 4))
        uk = _iq_pack_code(lu, 3)
        Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 4), wv.reshape(-1, 4), np.repeat(scale, 8))
        Ln = Ln.reshape(G, 8, 4)
        bad = bad.reshape(G, 8)
        L = np.where((~is_on_grid)[..., None] & bad[..., None] & do[:, None, None], Ln, L)
        Lf = L.reshape(G, 32).astype(np.float32)
        q = np.float32(2) * Lf + np.float32(1)
        sumqx = _iq_seq_sum(weight * xg * q)
        sumq2 = _iq_seq_sum(weight * q * q)
        scale = np.where(do & (sumq2 > 0), sumqx / sumq2, scale)

        neg = scale < 0
        scale = np.abs(scale)
        block_signs = np.where(neg[:, None], (~block_signs) & np.uint8(127), block_signs)

        uk2 = _iq_pack_code(L.reshape(G, 8, 4), 3)
        grid_index = kmap[uk2]
        assert (grid_index >= 0).all()
        q3 = grid_index.astype(np.uint8).reshape(n_blocks, 64)
        bs = block_signs.astype(np.uint32)
        ss = bs[:, 0] | (bs[:, 1] << np.uint32(7)) | (bs[:, 2] << np.uint32(14)) | (bs[:, 3] << np.uint32(21))

        scales_g = scale.reshape(n_blocks, n_groups)
        max_scale = scales_g.max(axis=-1)
        d = max_scale / np.float32(31)
        dh = (d * np.float32(1.0125))[:, None].astype(np.float16).view(np.uint8)
        idb = np.float32(1) / np.where(d > 0, d, np.float32(1))
        l = np.rint(np.float32(0.5) * (idb[:, None] * scales_g - np.float32(1)))
        l = np.clip(l, 0, 15).astype(np.uint32)
        ss = ss.reshape(n_blocks, n_groups) | (l << np.uint32(28))
        qs = np.concatenate([q3, ss.view(np.uint8).reshape(n_blocks, 32)], axis=-1)
        return np.concatenate([dh, qs], axis=-1)


def _quantize_iq3_s(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    n_groups = QK_K // 32
    G = n_blocks * n_groups
    xg = x.reshape(G, 32)
    sigma2 = np.float32(2) * _iq_seq_sum(x * x) / np.float32(QK_K)
    sig2 = np.repeat(sigma2, n_groups)
    if weights is None:
        weight = xg * xg
    else:
        weight = weights.reshape(G, 32) * np.sqrt(sig2[:, None] + xg * xg)
    waux = np.sqrt(weight)
    xval, block_signs = _iq_signs(xg.reshape(G, 4, 8), weight.reshape(G, 4, 8), False, False)
    maxv = xval.max(axis=(1, 2))
    zero = maxv == 0
    pos, kmap, nb_idx, _ = _iq_search_tables_for(cls, 4096, 3)

    best = np.zeros(G, dtype=np.float32)
    scale = maxv / np.float32(15)
    L = np.zeros((G, 32), dtype=np.int32)
    is_on_grid = np.zeros((G, 8), dtype=bool)
    denom = np.where(zero, np.float32(1), maxv)
    xv = xval.reshape(G, 8, 4)
    wv = waux.reshape(G, 8, 4)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for is_ in range(-9, 10):
            id = (np.float32(15) + np.float32(0.2) * is_) / denom
            this_scale = np.float32(1) / id
            lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
            lu = np.clip(lu, 0, 7).astype(np.int32)
            uk = _iq_pack_code(lu, 3)
            Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 4), wv.reshape(-1, 4), np.repeat(this_scale, 8))
            Ln = Ln.reshape(G, 8, 4)
            bad = bad.reshape(G, 8)
            Laux = np.where(bad[..., None], Ln, lu)
            Lauxf = Laux.reshape(G, 32).astype(np.float32)
            q = np.float32(2) * Lauxf + np.float32(1)
            sumqx = _iq_seq_sum(weight * xg * q)
            sumq2 = _iq_seq_sum(weight * q * q)
            better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            scale = np.where(better, sumqx / sumq2, scale)
            best = np.where(better, scale * sumqx, best)
            L = np.where(better[:, None], Laux.reshape(G, 32), L)
            is_on_grid = np.where(better[:, None], ~bad, is_on_grid)

        # second pass over all k (the C has the is_on_grid skip commented out)
        do = scale > 0
        id = np.float32(1) / np.where(do, scale, np.float32(1))
        lu = np.rint(np.float32(0.5) * (id[:, None, None] * xv - np.float32(1)))
        lu = np.clip(lu, 0, 7).astype(np.int32)
        L = np.where(do[:, None, None], lu, L.reshape(G, 8, 4))
        uk = _iq_pack_code(lu, 3)
        Ln, bad = _iq_neighbour_L(pos, kmap, nb_idx, uk.reshape(-1), xv.reshape(-1, 4), wv.reshape(-1, 4), np.repeat(scale, 8))
        Ln = Ln.reshape(G, 8, 4)
        bad = bad.reshape(G, 8)
        L = np.where(do[:, None, None] & bad[..., None], Ln, L)
        Lf = L.reshape(G, 32).astype(np.float32)
        q = np.float32(2) * Lf + np.float32(1)
        sumqx = _iq_seq_sum(weight * xg * q)
        sumq2 = _iq_seq_sum(weight * q * q)
        scale = np.where(do & (sumq2 > 0), sumqx / sumq2, scale)

        neg = scale < 0
        scale = np.abs(scale)
        block_signs = np.where(neg[:, None], ~block_signs, block_signs)

        uk2 = _iq_pack_code(L.reshape(G, 8, 4), 3)
        grid_index = kmap[uk2]
        assert (grid_index >= 0).all()
        gi = grid_index.reshape(n_blocks, 8, 8)
        qs = (gi & 255).astype(np.uint8).reshape(n_blocks, 64)
        qh = np.zeros((n_blocks, 8), dtype=np.uint8)
        pos64 = np.arange(64)
        val = ((gi >> 8).astype(np.uint8) << (pos64 % 8).astype(np.uint8).reshape(1, 8, 8)).reshape(n_blocks, 64)
        r = np.tile(np.arange(n_blocks)[:, None], (1, 64))
        c = np.tile((pos64 // 8).reshape(-1)[None, :], (n_blocks, 1))
        np.add.at(qh, (r, c), val)
        signs = block_signs.reshape(n_blocks, 32)

        scales_g = scale.reshape(n_blocks, n_groups)
        max_scale = scales_g.max(axis=-1)
        d = max_scale / np.float32(31)
        dh = (d * np.float32(1.033))[:, None].astype(np.float16).view(np.uint8)
        idb = np.float32(1) / np.where(d > 0, d, np.float32(1))
        l = np.rint(np.float32(0.5) * (idb[:, None] * scales_g - np.float32(1)))
        l = np.clip(l, 0, 15).astype(np.uint8)
        sc = (l[:, 0::2] | (l[:, 1::2] << np.uint8(4))).astype(np.uint8)
        return np.concatenate([dh, qs, qh, signs, sc], axis=-1)


# ===================================================================================
# IQ1_S (1.5625 bpw) / IQ1_M (1.75 bpw) - exact weighted SSD boundary search
# ===================================================================================

_IQ1_XP = np.array([-1 + 0.125, 0.125, 1 + 0.125], dtype=np.float32)
_IQ1_XM = np.array([-1 - 0.125, -0.125, 1 - 0.125], dtype=np.float32)


def _quantize_iq1_s(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    block_size = 32
    n_groups = QK_K // block_size
    G = n_blocks * n_groups
    xg = x.reshape(G, 32)
    sigma2 = np.float32(2) * _iq_seq_sum(x * x) / np.float32(QK_K)
    sig2 = np.repeat(sigma2, n_groups)
    wq = np.ones_like(xg) if weights is None else weights.reshape(G, 32)
    weight = wq * np.sqrt(sig2[:, None] + xg * xg)
    maxv = np.abs(xg).max(axis=-1)
    zero = maxv < np.float32(1e-12)
    pos, kmap, nb_idx, _ = _iq_search_tables_for(cls, 43692, 3)

    # sort by xb; prefix sums over the sorted order
    order = np.argsort(xg, axis=-1, kind="stable")
    xs = np.take_along_axis(xg, order, axis=-1)
    ws = np.take_along_axis(weight, order, axis=-1)
    Px = np.zeros((G, 33), dtype=np.float32)
    Pw = np.zeros((G, 33), dtype=np.float32)
    Px[:, 1:] = np.cumsum(ws * xs, axis=-1, dtype=np.float32)
    Pw[:, 1:] = np.cumsum(ws, axis=-1, dtype=np.float32)

    best_score = np.full(G, np.finfo(np.float32).min, dtype=np.float32)
    scale = maxv.copy()
    besti1 = np.full(G, -1, dtype=np.int32)
    besti2 = np.full(G, -1, dtype=np.int32)
    best_shift = np.zeros(G, dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i1 in range(33):
            for i2 in range(i1, 33):
                sqxp = (Px[:, i1] - Px[:, 0]) * _IQ1_XP[0] + (Px[:, i2] - Px[:, i1]) * _IQ1_XP[1] + (Px[:, 32] - Px[:, i2]) * _IQ1_XP[2]
                sq2p = (Pw[:, i1] - Pw[:, 0]) * (_IQ1_XP[0] * _IQ1_XP[0]) + (Pw[:, i2] - Pw[:, i1]) * (_IQ1_XP[1] * _IQ1_XP[1]) + (Pw[:, 32] - Pw[:, i2]) * (_IQ1_XP[2] * _IQ1_XP[2])
                bet = (sq2p > 0) & (sqxp * sqxp > best_score * sq2p)
                scale = np.where(bet, sqxp / sq2p, scale)
                best_score = np.where(bet, scale * sqxp, best_score)
                besti1 = np.where(bet, i1, besti1)
                besti2 = np.where(bet, i2, besti2)
                best_shift = np.where(bet, 1, best_shift)
                sqxm = (Px[:, i1] - Px[:, 0]) * _IQ1_XM[0] + (Px[:, i2] - Px[:, i1]) * _IQ1_XM[1] + (Px[:, 32] - Px[:, i2]) * _IQ1_XM[2]
                sq2m = (Pw[:, i1] - Pw[:, 0]) * (_IQ1_XM[0] * _IQ1_XM[0]) + (Pw[:, i2] - Pw[:, i1]) * (_IQ1_XM[1] * _IQ1_XM[1]) + (Pw[:, 32] - Pw[:, i2]) * (_IQ1_XM[2] * _IQ1_XM[2])
                bet = (sq2m > 0) & (sqxm * sqxm > best_score * sq2m)
                scale = np.where(bet, sqxm / sq2m, scale)
                best_score = np.where(bet, scale * sqxm, best_score)
                besti1 = np.where(bet, i1, besti1)
                besti2 = np.where(bet, i2, besti2)
                best_shift = np.where(bet, -1, best_shift)

    ok = besti1 >= 0
    sp = np.arange(32).reshape(1, -1)
    lb = np.where(sp < besti1[:, None], 0, np.where(sp < besti2[:, None], 1, 2)).astype(np.int32)
    L = np.empty((G, 32), dtype=np.int32)
    L[np.arange(G)[:, None], order] = lb
    neg = scale < 0
    scale = np.abs(scale)
    L = np.where(neg[:, None], 2 - L, L)
    best_shift = np.where(neg, -best_shift, best_shift)
    scale = np.where(ok, scale, 0)
    best_shift = np.where(ok, best_shift, 1)
    L = np.where(ok[:, None], L, 1)

    xx = np.where(best_shift[:, None] == 1, _IQ1_XP[None, :], _IQ1_XM[None, :])
    uk = _iq_pack_code(L.reshape(G, 4, 8), 2)
    idx = _iq_neighbour_idx(pos, kmap, nb_idx, uk.reshape(-1), xg.reshape(-1, 8), weight.reshape(-1, 8), np.repeat(scale, 4), np.repeat(xx, 4, axis=0), 8)
    bad4 = idx < 0
    all_on_grid = ~bad4.reshape(G, 4).any(axis=1)
    gval = pos[idx.reshape(G, 4)]
    qv = np.take_along_axis(xx[:, None, :], (gval - 1) // 2, axis=-1).astype(np.float32).reshape(G, 32)
    sumqx = _iq_seq_sum(weight * qv * xg)
    sumq2 = _iq_seq_sum(weight * qv * qv)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = np.where(~all_on_grid & (sumqx > 0) & (sumq2 > 0), sumqx / sumq2, scale)

    # pack
    ok4 = ok[:, None]
    gi = np.where(ok4, idx.reshape(G, 4), 0)
    qs = np.zeros((n_blocks, 32), dtype=np.uint8)
    qs.reshape(n_blocks, 8, 4)[...] = (gi & 255).astype(np.uint8).reshape(n_blocks, 8, 4)
    h = (((gi >> 8).astype(np.uint16) << (3 * np.arange(4)).astype(np.uint16).reshape(1, 4)).sum(axis=-1)).astype(np.uint16)
    qh = np.zeros((n_blocks, 8), dtype=np.uint16)
    qh[...] = h.reshape(n_blocks, 8)

    scales_g = scale.reshape(n_blocks, n_groups)
    max_scale = scales_g.max(axis=-1)
    d = max_scale / np.float32(15)
    dh = (d * np.float32(1.125))[:, None].astype(np.float16).view(np.uint8)
    idb = np.float32(1) / np.where(d > 0, d, np.float32(1))
    l = np.rint(np.float32(0.5) * (idb[:, None] * scales_g - np.float32(1)))
    l = np.clip(l, 0, 7).astype(np.uint16)
    sh = best_shift.reshape(n_blocks, n_groups)
    qh2 = qh | (l << np.uint16(12))
    qh2 = np.where(sh == -1, qh2 | np.uint16(8 << 12), qh2)
    return np.concatenate([dh, qs, qh2.view(np.uint8).reshape(n_blocks, 16)], axis=-1)


def _quantize_iq1_m(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    block_size = 16
    n_groups = QK_K // block_size
    G = n_blocks * n_groups
    xg = x.reshape(G, 16)
    sigma2 = np.float32(2) * _iq_seq_sum(x * x) / np.float32(QK_K)
    sig2 = np.repeat(sigma2, n_groups)
    if weights is None:
        weight = xg * xg
    else:
        weight = weights.reshape(G, 16) * np.sqrt(sig2[:, None] + xg * xg)
    maxv = np.abs(xg).max(axis=-1)
    zero = maxv < np.float32(1e-7)
    pos, kmap, nb_idx, _ = _iq_search_tables_for(cls, 43692, 3)

    order = np.argsort(xg, axis=-1, kind="stable")
    xs = np.take_along_axis(xg, order, axis=-1)
    ws = np.take_along_axis(weight, order, axis=-1)
    half = order >= 8
    # k: 0 = +,+  1 = +,-  2 = -,+  3 = -,-
    kp = np.array([[True, True], [True, False], [False, True], [False, False]])
    P = np.zeros((4, 3, G, 17), dtype=np.float32)
    Q = np.zeros((4, 3, G, 17), dtype=np.float32)
    for k in range(4):
        for m in range(3):
            mul = np.where(half, _IQ1_XP[m] if kp[k][1] else _IQ1_XM[m], _IQ1_XP[m] if kp[k][0] else _IQ1_XM[m])
            P[k, m][:, 1:] = np.cumsum((ws * mul) * xs, axis=-1, dtype=np.float32)
            Q[k, m][:, 1:] = np.cumsum((ws * mul) * mul, axis=-1, dtype=np.float32)

    best_score = np.full(G, np.finfo(np.float32).min, dtype=np.float32)
    scale = maxv.copy()
    besti1 = np.full(G, -1, dtype=np.int32)
    besti2 = np.full(G, -1, dtype=np.int32)
    best_k = np.full(G, -1, dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i1 in range(17):
            for i2 in range(i1, 17):
                for k in range(4):
                    s0 = P[k, 0, :, i1] + (P[k, 1, :, i2] - P[k, 1, :, i1]) + (P[k, 2, :, 16] - P[k, 2, :, i2])
                    s2 = Q[k, 0, :, i1] + (Q[k, 1, :, i2] - Q[k, 1, :, i1]) + (Q[k, 2, :, 16] - Q[k, 2, :, i2])
                    bet = (s2 > 0) & (s0 * s0 > best_score * s2)
                    scale = np.where(bet, s0 / s2, scale)
                    best_score = np.where(bet, scale * s0, best_score)
                    besti1 = np.where(bet, i1, besti1)
                    besti2 = np.where(bet, i2, besti2)
                    best_k = np.where(bet, k, best_k)

    ok = besti1 >= 0
    sp = np.arange(16).reshape(1, -1)
    lb = np.where(sp < besti1[:, None], 0, np.where(sp < besti2[:, None], 1, 2)).astype(np.int32)
    L = np.empty((G, 16), dtype=np.int32)
    L[np.arange(G)[:, None], order] = lb
    neg = scale < 0
    scale = np.abs(scale)
    L = np.where(neg[:, None], 2 - L, L)
    best_k = np.where(neg, 3 - best_k, best_k)
    scale = np.where(ok, scale, 0)
    best_k = np.where(ok, best_k, 0)
    L = np.where(ok[:, None], L, 1)

    xxk0 = np.where(best_k[:, None] < 2, _IQ1_XP[None, :], _IQ1_XM[None, :])
    xxk1 = np.where((best_k[:, None] % 2) == 0, _IQ1_XP[None, :], _IQ1_XM[None, :])
    xx_all = np.stack([xxk0, xxk1], axis=1)                      # (G, 2, 3)
    uk = _iq_pack_code(L.reshape(G, 2, 8), 2)
    idx = _iq_neighbour_idx(pos, kmap, nb_idx, uk.reshape(-1), xg.reshape(-1, 8), weight.reshape(-1, 8), np.repeat(scale, 2), xx_all.reshape(-1, 3), 8)
    bad2 = idx < 0
    all_on_grid = ~bad2.reshape(G, 2).any(axis=1)
    gval = pos[idx]
    qv = np.take_along_axis(xx_all.reshape(G, 2, 1, 3), ((gval - 1) // 2).reshape(G, 2, 8, 1), axis=-1).astype(np.float32).reshape(G, 16)
    sumqx = _iq_seq_sum(weight * qv * xg)
    sumq2 = _iq_seq_sum(weight * qv * qv)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = np.where(~all_on_grid & (sumqx > 0) & (sumq2 > 0), sumqx / sumq2, scale)

    # pack qs/qh
    ok2 = ok[:, None]
    gi = np.where(ok2, idx.reshape(G, 2), 0)
    qs = np.zeros((n_blocks, 32), dtype=np.uint8)
    qs.reshape(n_blocks, 16, 2)[...] = (gi & 255).astype(np.uint8).reshape(n_blocks, 16, 2)
    qh = np.zeros((n_blocks, 16), dtype=np.uint8)
    qh.reshape(n_blocks, 16)[...] = (((gi >> 8).astype(np.uint8) << np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)).sum(axis=-1)).astype(np.uint8).reshape(n_blocks, 16)

    # block scales: 3-bit per group in the scales words, then the f16 d packed on top
    scales_g = scale.reshape(n_blocks, n_groups)
    max_scale = scales_g.max(axis=-1)
    d = max_scale / np.float32(15)
    idb = np.float32(1) / np.where(d > 0, d, np.float32(1))
    l = np.rint(np.float32(0.5) * (idb[:, None] * scales_g - np.float32(1)))
    l = np.clip(l, 0, 7).astype(np.int32)
    sc = np.zeros((n_blocks, 4), dtype=np.uint16)
    lg = l.reshape(n_blocks, 4, 4)
    sc |= (lg.astype(np.uint16) << np.array([0, 3, 6, 9], dtype=np.uint16).reshape(1, 1, 4)).sum(axis=-1)
    masks = np.array([0x00, 0x80, 0x08, 0x88], dtype=np.uint8).reshape(1, -1)
    qh = qh | np.take_along_axis(masks, best_k.reshape(n_blocks, n_groups), axis=-1)

    # final scale from the packed grid values
    qs16 = qs.reshape(n_blocks, 16, 2).astype(np.uint16)
    qh16 = qh.astype(np.uint16).reshape(n_blocks, 16)
    gi2 = np.stack([qs16[..., 0] | ((qh16 << np.uint16(8)) & np.uint16(0x700)),
                    qs16[..., 1] | ((qh16 << np.uint16(4)) & np.uint16(0x700))], axis=-1)
    gval2 = pos[gi2.reshape(-1)].reshape(n_blocks, 16, 2, 8)
    qv2 = np.take_along_axis(xx_all.reshape(n_blocks, 16, 2, 1, 3), ((gval2 - 1) // 2).reshape(n_blocks, 16, 2, 8, 1), axis=-1).astype(np.float32)
    qf = qv2 * (np.float32(2) * l.reshape(n_blocks, 16, 1, 1, 1) + np.float32(1))
    xbf = x.reshape(n_blocks, 16, 16)
    w2 = weight.reshape(n_blocks, 16, 16)
    qf16 = qf.reshape(n_blocks, 16, 16)
    sumqx_f = _iq_seq_sum((w2 * qf16 * xbf).reshape(n_blocks, n_groups * 16))
    sumq2_f = _iq_seq_sum((w2 * qf16 * qf16).reshape(n_blocks, n_groups * 16))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d = np.where(sumq2_f > 0, sumqx_f / sumq2_f, d)
    s16 = (d * np.float32(1.1125)).astype(np.float16).view(np.uint16)
    sc[..., 0] |= ((s16 & np.uint16(0x000f)) << np.uint16(12))
    sc[..., 1] |= ((s16 & np.uint16(0x00f0)) << np.uint16(8))
    sc[..., 2] |= ((s16 & np.uint16(0x0f00)) << np.uint16(4))
    sc[..., 3] |= ((s16 & np.uint16(0xf000)) << np.uint16(0))
    return np.concatenate([qs, qh, sc.view(np.uint8).reshape(n_blocks, 8)], axis=-1)


# ===================================================================================
# IQ4_NL (4.5 bpw) / IQ4_XS (4.25 bpw) - non-linear 4-bit with a value lookup
# ===================================================================================

def _quantize_iq4_nl(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    values = np.array(cls.kvalues, dtype=np.int8)
    sigma2 = np.float32(2) * _iq_seq_sum(x * x) / np.float32(32)
    if weights is None:
        weight = x * x
    else:
        weight = weights.reshape(n_blocks, 32) * np.sqrt(sigma2[:, None] + x * x)

    ax = np.abs(x)
    imax = ax.argmax(axis=-1)
    maxv = np.take_along_axis(x, imax[:, None], axis=-1)[..., 0]
    amax = ax.max(axis=-1)
    zero = amax < np.float32(1e-15)

    # reference path (ntry = -1): d = max/values[0], no refinement loop, no re-quant
    d0 = maxv / np.float32(values[0])
    id0 = np.float32(1) / np.where(zero, np.float32(1), d0)
    L = _iq_best_index_int8(id0[:, None] * x, values)
    q = values[L].astype(np.float32)
    sumqx = _iq_seq_sum(weight * q * x)
    sumq2 = _iq_seq_sum(weight * q * q)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d2 = np.where(sumq2 > 0, sumqx / sumq2, np.float32(0))
    scales = np.where(zero, np.float32(0), d2)

    # single super-block: dh = fp16(scales); zero blocks leave the stale L buffer
    # (previous non-zero block, or zeros for a leading zero block) in the C
    nz = ~zero
    last = np.maximum.accumulate(np.where(nz, np.arange(n_blocks), -1))
    Lp = L[np.maximum(last, 0)]
    L = np.where((last >= 0)[:, None], Lp, np.zeros_like(L))

    dh = scales[:, None].astype(np.float16).view(np.uint8)
    Lg = L.astype(np.uint8).reshape(n_blocks, 2, 16)
    qs = (Lg[:, 0, :] | (Lg[:, 1, :] << np.uint8(4)))
    return np.concatenate([dh, qs], axis=-1)


def _quantize_iq4_xs(cls, blocks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    x = blocks.astype(np.float32)
    n_groups = QK_K // 32
    G = n_blocks * n_groups
    xg = x.reshape(G, 32)
    values = np.array(cls.kvalues, dtype=np.int8)
    sigma2 = np.float32(2) * _iq_seq_sum(x * x) / np.float32(QK_K)
    sig2 = np.repeat(sigma2, n_groups)
    if weights is None:
        weight = xg * xg
    else:
        weight = weights.reshape(G, 32) * np.sqrt(sig2[:, None] + xg * xg)

    ax = np.abs(xg)
    imax = ax.argmax(axis=-1)
    maxv = np.take_along_axis(xg, imax[:, None], axis=-1)[..., 0]
    amax = ax.max(axis=-1)
    zero = amax < np.float32(1e-15)

    # ntry = 7: initial d = -max/values[0] = max/127, then the itry refinement loop
    d = -maxv / np.float32(values[0])
    id = np.float32(1) / np.where(zero, np.float32(1), d)
    L = _iq_best_index_int8(id[:, None] * xg, values)
    q = values[L].astype(np.float32)
    sumqx = _iq_seq_sum(weight * q * xg)
    sumq2 = _iq_seq_sum(weight * q * q)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d = np.where(sumq2 > 0, sumqx / sumq2, np.float32(0))
    best = d * sumqx
    for itry in range(-7, 8):
        idt = (np.float32(itry) + np.float32(values[0])) / np.where(zero, np.float32(1), maxv)
        Lt = _iq_best_index_int8(idt[:, None] * xg, values)
        qt = values[Lt].astype(np.float32)
        sumqx = _iq_seq_sum(weight * qt * xg)
        sumq2 = _iq_seq_sum(weight * qt * qt)
        bet = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d = np.where(bet, sumqx / sumq2, d)
        best = np.where(bet, d * sumqx, best)
        L = np.where(bet[:, None], Lt, L)
    scales = np.where(zero, np.float32(0), d).reshape(n_blocks, n_groups)

    abs_d = np.abs(scales)
    am = abs_d.max(axis=-1)
    im = abs_d.argmax(axis=-1)
    max_scale = np.take_along_axis(scales, im[:, None], axis=-1)[..., 0]
    max_scale = np.where(am > 0, max_scale, np.float32(0))
    dblk = -max_scale / np.float32(32)
    dh = dblk[:, None].astype(np.float16).view(np.uint8)
    idb = np.float32(1) / np.where(dblk != 0, dblk, np.float32(1))
    l = np.rint(idb[:, None] * scales)
    l = np.clip(l, -32, 31).astype(np.int32)
    dl = dblk[:, None] * l
    idl = np.float32(1) / np.where(dl != 0, dl, np.float32(1))
    L = _iq_best_index_int8(idl[:, :, None] * xg.reshape(n_blocks, n_groups, 32), values)
    lp = l + 32
    l_l = (lp & 15).astype(np.uint8)
    l_h = (lp >> 4).astype(np.uint8)
    scales_l = np.zeros((n_blocks, 4), dtype=np.uint8)
    lg2 = l_l.reshape(n_blocks, 8, 1)
    scales_l[:, 0] = (lg2[:, 0, 0] | (lg2[:, 1, 0] << 4))
    scales_l[:, 1] = (lg2[:, 2, 0] | (lg2[:, 3, 0] << 4))
    scales_l[:, 2] = (lg2[:, 4, 0] | (lg2[:, 5, 0] << 4))
    scales_l[:, 3] = (lg2[:, 6, 0] | (lg2[:, 7, 0] << 4))
    scales_h = np.zeros((n_blocks,), dtype=np.uint16)
    lh2 = l_h.reshape(n_blocks, 8)
    scales_h = (lh2.astype(np.uint16) << (2 * np.arange(8)).astype(np.uint16).reshape(1, 8)).sum(axis=-1).astype(np.uint16)

    Lg = L.astype(np.uint8).reshape(n_blocks, 8, 2, 16)
    qs = (Lg[..., 0, :] | (Lg[..., 1, :] << np.uint8(4))).reshape(n_blocks, 128)
    return np.concatenate([dh, scales_h.view(np.uint8).reshape(n_blocks, 2), scales_l, qs], axis=-1)


def _apply() -> None:
    IQ2_XXS.quantize_blocks = classmethod(_quantize_iq2_xxs)
    IQ2_XS.quantize_blocks = classmethod(_quantize_iq2_xs)
    IQ2_S.quantize_blocks = classmethod(_quantize_iq2_s)
    IQ3_XXS.quantize_blocks = classmethod(_quantize_iq3_xxs)
    IQ3_S.quantize_blocks = classmethod(_quantize_iq3_s)
    IQ1_S.quantize_blocks = classmethod(_quantize_iq1_s)
    IQ1_M.quantize_blocks = classmethod(_quantize_iq1_m)
    IQ4_NL.quantize_blocks = classmethod(_quantize_iq4_nl)
    IQ4_XS.quantize_blocks = classmethod(_quantize_iq4_xs)


_apply()
