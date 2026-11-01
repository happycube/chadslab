"""NumPy references of the stereo records (the Python interpreter's steps and
the unit tests): ENC_RESIZE, ENC_DIBR and ENC_PACK_U8, the same float32
operations in the same order as interp.c and encoders.cu (NumPy's float32
products and sums are not fused), so all three give the same bits."""
from __future__ import annotations

import numpy as np

f32 = np.float32


def lerp_src(n_in, n_out, mode):
    """The source pixels and weights of each output of a bilinear resize
    (torch's upsample_bilinear2d in float32; mode 1 align_corners)."""
    d = np.arange(n_out, dtype=np.float32)
    if mode == 1:
        scale = f32(n_in - 1) / f32(n_out - 1) if n_out > 1 else f32(0)
        src = f32(scale) * d
    else:
        scale = f32(n_in) / f32(n_out)
        src = f32(scale) * (d + f32(0.5)) - f32(0.5)
        src = np.where(src < 0, f32(0), src).astype(np.float32)
    i0 = np.minimum(src.astype(np.int64), n_in - 1)
    i1 = np.where(i0 < n_in - 1, i0 + 1, i0)
    l1 = (src - i0.astype(np.float32)).astype(np.float32)
    return i0, i1, (f32(1) - l1).astype(np.float32), l1


def resize(x, H, W, mode):
    """x (n, h, w, C) float32 -> (n, H, W, C): mode 0 bilinear (align_corners
    False), 1 bilinear (align_corners True), 2 the mean of integer blocks."""
    x = np.asarray(x, np.float32)
    n, h, w, C = x.shape
    if mode == 2:
        fy, fx = h // H, w // W
        acc = np.zeros((n, H, W, C), np.float32)
        for a in range(fy):
            for b in range(fx):
                acc = acc + x[:, a::fy, b::fx][:, :H, :W]
        return (acc / f32(fy * fx)).astype(np.float32)
    y0, y1, ly0, ly1 = lerp_src(h, H, mode)
    x0, x1, lx0, lx1 = lerp_src(w, W, mode)
    lx0, lx1 = lx0[None, None, :, None], lx1[None, None, :, None]
    top = lx0 * x[:, y0][:, :, x0] + lx1 * x[:, y0][:, :, x1]
    bot = lx0 * x[:, y1][:, :, x0] + lx1 * x[:, y1][:, :, x1]
    return np.ascontiguousarray(ly0[None, :, None, None] * top + ly1[None, :, None, None] * bot, dtype=np.float32)


def resize_taps(x, iy, wy, ix, wx):
    """x (n, h, w, C) by the taps of each axis (ENC_RESIZE_TAPS): the columns'
    sums first (in order), then the rows'"""
    x = np.asarray(x, np.float32)
    wy, wx = np.asarray(wy, np.float32), np.asarray(wx, np.float32)
    out = np.zeros((x.shape[0], iy.shape[0], ix.shape[0], x.shape[3]), np.float32)
    for k in range(iy.shape[1]):
        rows = x[:, iy[:, k]]                                   # (n, H, w, C)
        hs = np.zeros(out.shape, np.float32)
        for j in range(ix.shape[1]):
            hs = hs + wx[None, None, :, j, None] * rows[:, :, ix[:, j]]
        out = out + wy[None, :, k, None, None] * hs
    return out


def orderable(f):
    """float32 -> uint32 whose unsigned order is the float order"""
    u = np.asarray(f, np.float32).view(np.uint32)
    return np.where(u & np.uint32(0x80000000), ~u, u | np.uint32(0x80000000)).astype(np.uint32)


def unorder(u):
    u = np.asarray(u, np.uint32)
    b = np.where(u & np.uint32(0x80000000), u & np.uint32(0x7FFFFFFF), ~u).astype(np.uint32)
    return b.view(np.float32)


def _expand(lo, hi):
    """(rows, j, X) of every integer X in [lo, hi) of each (row, j)"""
    cnt = np.maximum(hi - lo, 0)
    rows, js = np.nonzero(cnt)
    c = cnt[rows, js]
    r_ = np.repeat(rows, c)
    j_ = np.repeat(js, c)
    start = np.repeat(np.cumsum(c) - c, c)
    X = np.repeat(lo[rows, js], c) + (np.arange(c.sum()) - start)
    return r_, j_, X


def dibr(img, s, sign, gap=1.0, mode=1, iters=8, tol=0.5):
    """One eye (ENC_DIBR): img (H, W, 3), s (H, W) float32 (the shift of the
    left eye; the right eye's is -s: sign -1). Returns (the eye (H, W, 3),
    the hole mask, the keys)."""
    img = np.asarray(img, np.float32)
    s = np.asarray(s, np.float32)
    sign = f32(sign)
    H, W = s.shape
    keys = np.zeros((H, W), np.uint64)
    jj = np.arange(W, dtype=np.float32)
    p = jj[None, :] + sign * s
    # boxes
    r_, j_, X = _expand(np.ceil(p - f32(0.5)).astype(np.int64), np.ceil(p + f32(0.5)).astype(np.int64))
    ok = (X >= 0) & (X < W)
    r_, j_, X = r_[ok], j_[ok], X[ok]
    k = (orderable(s[r_, j_]).astype(np.uint64) << np.uint64(32)) | (j_.astype(np.uint64) << np.uint64(1))
    np.maximum.at(keys, (r_, X), k)
    # segments
    p0, p1 = p[:, :-1], p[:, 1:]
    span = (p1 - p0).astype(np.float32)
    draw = (span > 0) & (span <= f32(1) + f32(gap))
    lo = np.where(draw, np.ceil(p0).astype(np.int64), 0)
    hi = np.where(draw, np.ceil(p1).astype(np.int64), 0)
    r_, j_, X = _expand(lo, hi)
    ok = (X >= 0) & (X < W)
    r_, j_, X = r_[ok], j_[ok], X[ok]
    a0, sp = p0[r_, j_], span[r_, j_]
    t = (X.astype(np.float32) - a0) / sp
    sa, sb = s[r_, j_], s[r_, j_ + 1]
    sx = sa + t * (sb - sa)
    k = (orderable(sx).astype(np.uint64) << np.uint64(32)) | (j_.astype(np.uint64) << np.uint64(1)) | np.uint64(1)
    np.maximum.at(keys, (r_, X), k)
    # resolve
    hole = keys == 0
    tmp = np.zeros((H, W, 3), np.float32)
    zb = np.zeros((H, W), np.float32)
    rr, XX = np.nonzero(~hole)
    kk = keys[rr, XX]
    zb[rr, XX] = unorder((kk >> np.uint64(32)).astype(np.uint32))
    j = ((kk >> np.uint64(1)) & np.uint64(0x7FFFFFFF)).astype(np.int64)
    seg = (kk & np.uint64(1)).astype(bool)
    tmp[rr[~seg], XX[~seg]] = img[rr[~seg], j[~seg]]
    r2, X2, j2 = rr[seg], XX[seg], j[seg]
    q0 = jj[j2] + sign * s[r2, j2]
    q1 = jj[j2 + 1] + sign * s[r2, j2 + 1]
    t = ((X2.astype(np.float32) - q0) / (q1 - q0))[:, None]
    ca, cb = img[r2, j2], img[r2, j2 + 1]
    tmp[r2, X2] = ca + t * (cb - ca)
    # stretch: each hole pixel from its farther nearest drawn neighbour
    hr, hx = np.nonzero(hole)
    if len(hr):
        idx = np.where(~hole, np.arange(W)[None, :], -1)
        left = np.maximum.accumulate(idx, axis=1)
        idx = np.where(~hole, np.arange(W)[None, :], W)
        right = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
        # the nearest drawn pixel strictly left / right of each hole pixel
        L = np.where(hx > 0, left[hr, np.maximum(hx - 1, 0)], -1)
        R = np.where(hx < W - 1, right[hr, np.minimum(hx + 1, W - 1)], W)
        kL = np.where(L >= 0, keys[hr, np.clip(L, 0, W - 1)], np.uint64(0))
        kR = np.where(R < W, keys[hr, np.clip(R, 0, W - 1)], np.uint64(0))
        useL = (L >= 0) & ((R >= W) | (kL <= kR))
        src = np.where(useL, L, R)
        none = (L < 0) & (R >= W)
        srcc = np.clip(src, 0, W - 1)
        tmp[hr, hx] = np.where(none[:, None], f32(0), tmp[hr, srcc])
        zb[hr, hx] = np.where(none, f32(-np.inf), zb[hr, srcc])
    out = tmp.copy()
    if len(hr):
        if mode == 0:
            acc = np.zeros((len(hr), 3), np.float32)
            for d in (-2, -1, 0, 1, 2):
                acc = acc + tmp[hr, np.clip(hx + d, 0, W - 1)]
            out[hr, hx] = acc / f32(5)
        else:
            lim = zb[hr, hx] + f32(tol)
            for _ in range(iters):
                acc = np.zeros((len(hr), 3), np.float32)
                cnt = np.zeros(len(hr), np.float32)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = hr + dy, hx + dx
                        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                        yc, xc = np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)
                        use = inside & (hole[yc, xc] | (zb[yc, xc] <= lim))
                        acc = acc + np.where(use[:, None], tmp[yc, xc], f32(0))
                        cnt = cnt + np.where(use, f32(1), f32(0))
                out[hr, hx] = acc / cnt[:, None]
                tmp[hr, hx] = out[hr, hx]
    return out, hole, keys


def pack_u8(x):
    v = np.rint(np.asarray(x, np.float32))
    return np.clip(v, 0, 255).astype(np.uint8)
