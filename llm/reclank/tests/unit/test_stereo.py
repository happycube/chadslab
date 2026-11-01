"""The stereo records (ENC_RESIZE, ENC_RESIZE_TAPS, ENC_DIBR, ENC_PACK_U8, ENC_UNPACK_U8) on the
CPU against their NumPy references (the same bits), the Python interpreter's steps, the Warper,
VDA's window alignment streamed against the authors' loop, and the st3d box of an MP4.

The authors' loop written from Video Depth Anything (DepthAnything/Video-Depth-Anything,
video_depth_anything/video_depth.py, infer_video_depth; Apache License 2.0; Copyright (2025)
Bytedance Ltd. and/or its affiliates). See THIRD_PARTY_NOTICES.md.

tests/unit/data/stereo_real.npz holds only public-domain content: "image" a 96 x 160 crop (at row
176, column 240) of NASA's photo AS11-40-5903 (Apollo 11, Buzz Aldrin by Neil Armstrong; public
domain, from Wikimedia Commons' Aldrin_Apollo_11_original.jpg) downscaled to 640 x 644; "shift" the
left eye's shift of that crop (DA3MONO's disparity of the photo, 6 % divergence, the convergence on
the centre); "vda" 60 maps of 27 x 36 (the same disparity at 60 offsets), real values for the
alignment's arithmetic. The checks of the models on real photos and video (outside the repository)
are scripts/check_stereo.py's."""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from reclank.encoders.runtime import _EncCompiler
from reclank.models.depth import ref
from reclank.models.depth import stereo as S

DATA = os.path.join(os.path.dirname(__file__), "data", "stereo_real.npz")


@pytest.fixture(scope="module")
def real():
    return np.load(DATA)


def bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def program(img, s):
    H, W = s.shape
    c = _EncCompiler(1e-6, pack=True)
    c.env.update(img=img.astype(np.float32).copy(), s=s.copy(), keys=np.zeros(2 * H * W, np.float32),
                 zb=np.zeros(H * W, np.float32), tmp=np.zeros(H * W * 3, np.float32),
                 e0=np.zeros((H, 2 * W, 3), np.float32), e1=np.zeros((2 * H, W, 3), np.float32),
                 u8=np.zeros(H * 2 * W * 3 // 4, np.float32), small=np.zeros((1, 37, 61, 3), np.float32),
                 back=np.zeros((1, H, W, 3), np.float32), half=np.zeros((1, H, W, 3), np.float32))
    c.compile(("seq",
               ("enc_dibr", "img", "s", "keys", "zb", "tmp", "e0", H, W, 0, 6 * W, 1, 1.0, 0, 8, 0.5),
               ("enc_dibr", "img", "s", "keys", "zb", "tmp", "e0", H, W, 3 * W, 6 * W, -1, 1.0, 0, 8, 0.5),
               ("enc_dibr", "img", "s", "keys", "zb", "tmp", "e1", H, W, 0, 3 * W, 1, 1.0, 1, 8, 0.5),
               ("enc_dibr", "img", "s", "keys", "zb", "tmp", "e1", H, W, 3 * H * W, 3 * W, -1, 1.0, 1, 8, 0.5),
               ("enc_pack_u8", "e0", "u8"),
               ("enc_resize", "img", "small", 1, H, W, 37, 61, 3, 0),
               ("enc_resize", "small", "back", 1, 37, 61, H, W, 3, 1),
               ("enc_resize", "e0", "half", 1, H, 2 * W, H, W, 3, 2)))
    p = c.p.finish()
    p.threads = p.buf[3] = 4
    return c.env, p


def expected(img, s, env):
    W = s.shape[1]
    H = s.shape[0]
    e0 = np.concatenate([ref.dibr(img, s, 1, 1.0, 0)[0], ref.dibr(img, s, -1, 1.0, 0)[0]], 1)
    e1 = np.concatenate([ref.dibr(img, s, 1, 1.0, 1)[0], ref.dibr(img, s, -1, 1.0, 1)[0]], 0)
    small = ref.resize(img[None].astype(np.float32), 37, 61, 0)
    return dict(e0=e0, e1=e1, small=small, back=ref.resize(small, H, W, 1),
                half=ref.resize(e0[None], H, W, 2), u8=ref.pack_u8(e0))


def test_records_cpu_same_bits(real):
    img, s = real["image"], real["shift"]
    env, p = program(img, s)
    p.run()
    want = expected(img, s, env)
    for name in ("e0", "e1", "small", "back", "half"):
        assert (bits(env[name]) == bits(want[name]).reshape(env[name].shape)).all(), name
    H, W = s.shape
    assert (env["u8"].view(np.uint8).reshape(H, 2 * W, 3) == want["u8"]).all()


def test_records_python_steps(real):
    img, s = real["image"], real["shift"]
    env, p = program(img, s)
    p.run_py()
    want = expected(img, s, env)
    for name in ("e0", "e1", "half"):
        assert (bits(env[name]) == bits(want[name]).reshape(env[name].shape)).all(), name


def test_dibr_holes_and_order(real):
    """the holes are where the background shows (behind the nearer pixels), and
    each drawn pixel comes from the nearest source"""
    img, s = real["image"], real["shift"]
    out, hole, keys = ref.dibr(img, s, 1, 1.0, 1)
    assert 0.01 < hole.mean() < 0.5
    H, W = s.shape
    zb = ref.unorder((keys >> np.uint64(32)).astype(np.uint32))
    j = ((keys >> np.uint64(1)) & np.uint64(0x7FFFFFFF)).astype(np.int64)
    rows = np.nonzero(~hole)
    # a box winner's s is its source pixel's own; nothing drawn is farther than a pixel landing on it
    box = (keys[rows] & np.uint64(1)) == 0
    assert (zb[rows][box] == s[rows[0][box], j[rows][box]]).all()
    p = np.arange(W)[None, :] + s
    for y in range(0, H, 7):
        for x in np.nonzero(~hole[y])[0][::5]:
            land = np.nonzero(np.abs(p[y] - x) < 0.5)[0]
            if len(land):
                assert zb[y, x] >= s[y, land].max() - 1e-6


def test_vda_alignment_streamed(real, monkeypatch):
    """VDASmall.video (streamed, windows of 32, step 22) gives the maps of the
    authors' infer_video_depth loop over the same window outputs"""
    from reclank.models.depth import vda as V
    maps = real["vda"]
    n_frames = 57
    h, w = maps.shape[1:]

    class Fake(V.VDASmall):
        def __init__(self):
            self.size = 518
            self.calls = []

        def prepare(self, frame):
            return frame                          # the frame's index

        def window(self, patches, ph, pw):
            self.calls.append([int(p) for p in patches])
            # a window's output: real maps, a function of the frame and of the window
            k = len(self.calls)
            return np.stack([maps[int(p) % len(maps)] * (1 + 0.1 * k) + 0.01 * k for p in patches])
    m = Fake()
    monkeypatch.setattr(V, "input_size", lambda hh, ww, size=518: (14 * 2, 14 * 2))
    class Frame(int):
        pass
    frames = [Frame(i) for i in range(n_frames)]
    for f in frames:
        f.shape = (h, w, 3)
    got = list(m.video(iter(frames), size=(h, w)))
    assert [i for i, _ in got] == list(range(n_frames))
    # the authors' loop over the same windows
    step = V.INFER_LEN - V.OVERLAP
    org = n_frames
    flist = list(range(org)) + [org - 1] * ((step - org % step) % step + V.INFER_LEN - step)
    m2 = Fake()
    depth_list, pre = [], None
    for fid in range(0, org, step):
        cur = [flist[fid + i] for i in range(V.INFER_LEN)]
        if pre is not None:
            cur[:V.OVERLAP] = [pre[i] for i in V.KEYFRAMES]
        depth_list += list(m2.window(cur, 2, 2))
        pre = cur
    assert m.calls == m2.calls
    al = V.OVERLAP - V.INTERP_LEN
    aligned, ref_al = [], []
    for fid in range(0, len(depth_list), V.INFER_LEN):
        if not aligned:
            aligned += depth_list[:V.INFER_LEN]
            ref_al = [depth_list[fid + k] for k in V.KEYFRAMES[:al]]
            continue
        sc, sh = V.fit_scale_shift(np.concatenate([depth_list[fid + i] for i in range(al)]), np.concatenate(ref_al))
        fix = lambda d: np.maximum(d * sc + sh, 0).astype(np.float32)  # noqa: E731
        pre_d = aligned[-V.INTERP_LEN:]
        post = [fix(depth_list[fid + i]) for i in range(al, V.OVERLAP)]
        wl = [0.0] + [i * (1.0 / (V.INTERP_LEN - 1)) for i in range(1, V.INTERP_LEN - 1)] + [1.0]
        aligned[-V.INTERP_LEN:] = [pre_d[i] * (1 - wl[i]) + post[i] * wl[i] for i in range(V.INTERP_LEN)]
        aligned += [fix(depth_list[fid + i]) for i in range(V.OVERLAP, V.INFER_LEN)]
        ref_al = ref_al[:1] + [fix(depth_list[fid + k]) for k in V.KEYFRAMES[1:al]]
    for i, d in got:
        assert (bits(d) == bits(aligned[i])).all(), i


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_st3d_box(real, tmp_path):
    img = np.concatenate([real["image"], real["image"]], 1)
    for flags in ([], ["-movflags", "+faststart"]):
        out = str(tmp_path / ("a%d.mp4" % len(flags)))
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", "%dx%d" % (img.shape[1], img.shape[0]), "-r", "24", "-i", "-",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p"] + flags + [out],
                           input=img.tobytes() * 4, capture_output=True)
        assert p.returncode == 0, p.stderr
        before = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "md5", "-"], capture_output=True).stdout
        assert S.inject_st3d(out, "sbs")
        assert not S.inject_st3d(out, "sbs")              # once
        after = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "md5", "-"], capture_output=True).stdout
        assert before == after
        data = open(out, "rb").read()
        i = data.find(b"st3d")
        assert i > 0 and data[i + 8] == 2
        if shutil.which("ffprobe"):
            pr = subprocess.run(["ffprobe", "-v", "error", "-show_streams", out], capture_output=True, text=True).stdout
            assert "side by side" in pr


def test_resize_taps_cpu_same_bits(real):
    """ENC_RESIZE_TAPS (OpenCV's cubic, area, torch's antialiased bilinear by the
    host's taps) on the CPU and in Python against ref.resize_taps"""
    from reclank.models.depth.common import resize_taps_of
    img = real["image"].astype(np.float32) / np.float32(255)
    H, W = img.shape[:2]
    for kind, (oh, ow) in (("cubic", (70, 118)), ("cubic", (130, 200)), ("area", (40, 64)), ("aa", (50, 77))):
        iy, wy, ix, wx = resize_taps_of(kind, H, W, oh, ow)
        c = _EncCompiler(1e-6, pack=True)
        c.env.update(x=img.reshape(-1, 3).copy(), y=np.zeros((oh * ow, 3), np.float32))
        c.compile(("seq", ("enc_resize_taps", "x", "y", 1, H, W, oh, ow, 3, iy, wy, ix, wx)))
        p = c.p.finish()
        p.threads = p.buf[3] = 4
        want = ref.resize_taps(img[None], iy, wy, ix, wx)[0]
        p.run()
        assert (bits(c.env["y"]) == bits(want).reshape(-1, 3)).all(), kind
        c.env["y"][:] = 0
        p.run_py()
        assert (bits(c.env["y"]) == bits(want).reshape(-1, 3)).all(), kind


def test_warper_cpu(real):
    """the Warper's program (uint8 upload, ENC_RESIZE of the shift, both eyes, the
    packing) on the CPU against the NumPy references"""
    img, s = real["image"], real["shift"]
    H, W = s.shape
    small = ref.resize(s[None, :, :, None], H // 2, W // 2, 0)[0, :, :, 0]
    full = ref.resize(small[None, :, :, None], H, W, 0)[0, :, :, 0]
    w = S.Warper(gpu=False, cpu_threads=4)
    for layout in S.LAYOUTS:
        got = w.warp(img, small, layout, "edge")
        L, R = ref.dibr(img, full, 1)[0], ref.dibr(img, full, -1)[0]
        both = np.concatenate([L, R], 1 if layout.startswith("sbs") else 0)
        want = both if not layout.endswith("half") else ref.resize(both[None], H, W, 2)[0]
        assert (got == ref.pack_u8(want)).all(), layout
