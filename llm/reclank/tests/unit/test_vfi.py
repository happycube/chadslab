"""The frame interpolation records (ENC_WARP, ENC_PRELU, ENC_MIX, and ENC_RESIZE's row strides and
column offsets) on the CPU and in the Python interpreter against their NumPy references (the same
bits), the output schedule, the scene-cut rule, the video pipeline's plumbing (ffmpeg in and out,
the frame counts, the rate, the audio), the GPU byte path's host halves (PIL's BOX passes of
probe_plan, thumbs_of, mean_abs, the deconv scatter), and Practical-RIFE 4.25 on the CPU against its
torch reference and on the GPU against its previous programs (with RECLANK_RIFE=the folder of
flownet.safetensors; skipped without it; the GPU's test marked gpu).

tests/unit/data/vfi_real.npz holds only our own content: "frames" two frames of a clip that Wan 2.2
TI2V-5B made here (a fox in snow, frames 2 and 10 of 17 at 24 fps, Lanczos-downscaled from 832 x 480
to 208 x 120); "out_05", "out_025" scripts/rife_reference.py's output at t 0.5 and 0.25 (pad 64);
"flow" and "mask" its last flow (4 channels) and mask over rows 24:88, columns 40:168 of the padded
frame (the real values for the records); "probe" the scene-cut test's probe frames (long side 320) of
frames 2 and 10 of the fox and of a frame of a Wan 2.2 T2V-A14B clip made here (waves at sunset,
1280 x 720, resized to 832 x 480), "probe_rd" RIFE's residual and raw difference of the two pairs,
"th32", "thgrey" the thumbnails of the three frames. The checks on real footage (outside the
repository) are scripts/check_rife.py's."""
from __future__ import annotations

import os
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from reclank.encoders.runtime import _EncCompiler
from reclank.models.depth import ref as dref
from reclank.models.vfi import ref, video as V

DATA = os.path.join(os.path.dirname(__file__), "data", "vfi_real.npz")
f32 = np.float32


@pytest.fixture(scope="module")
def real():
    return np.load(DATA)


def bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def inputs(real):
    """a 64 x 128 image (the first frame's crop / 255) and the real flow and mask there"""
    y0, x0 = real["crop"]
    img = (real["frames"][0, y0:y0 + 64, x0:x0 + 128].astype(np.float32) / f32(255))
    return img, real["flow"], real["mask"]


def program(real):
    img, fl, mask = inputs(real)
    h, w = 64, 128
    gx, gy = ref.grid(w), ref.grid(h)
    c = _EncCompiler(1e-6, pack=True)
    c.env.update(img=img.reshape(-1, 3).copy(), fl=fl.reshape(-1, 4).copy(), A=np.zeros((h * w, 10), np.float32),
                 pre=np.zeros((h * w, 4), np.float32), mix=np.zeros((h * w, 3), np.float32),
                 small=np.zeros((h // 4 * w // 4, 10), np.float32), back=np.zeros((h * w, 12), np.float32),
                 plain=np.zeros((h // 4 * w // 4, 3), np.float32))
    c.env["A"][:, 9] = mask.reshape(-1)
    slope = np.array([0.2, 0.2, 0.1, 0.25], np.float32)
    c.compile(("seq",
               ("enc_warp", "img", "fl", "A", gx, gy, h, w, 3, 3, 0, 4, 0, 10, 0),
               ("enc_warp", "img", "fl", "A", gx, gy, h, w, 3, 3, 0, 4, 2, 10, 3),
               ("enc_warp", "img", "fl", "A", gx, gy, h, w, 3, 3, 0, 4, 0, 10, 6),
               ("set", "pre", ("enc_prelu", "fl", slope)),
               ("enc_mix", "A", "mix", h * w, 10, 0, 3, 9, 3),
               ("enc_resize", "A", "small", 1, h, w, h // 4, w // 4, 10, 0),
               ("enc_resize", "small", "back", 1, h // 4, w // 4, h, w, 4, 0, 10, 3, 12, 5),
               ("enc_resize", "img", "plain", 1, h, w, h // 4, w // 4, 3, 2)))
    p = c.p.finish()
    p.threads = p.buf[3] = 4
    return c.env, p, slope


def expected(real, env, slope):
    img, fl, mask = inputs(real)
    h, w = 64, 128
    gx, gy = ref.grid(w), ref.grid(h)
    dx, dy = f32((w - 1.0) / 2.0), f32((h - 1.0) / 2.0)
    w0 = ref.warp(img, fl[..., 0:2], gx, gy, dx, dy)
    w1 = ref.warp(img, fl[..., 2:4], gx, gy, dx, dy)
    A = np.concatenate([w0, w1, w0, mask[..., None]], 2)
    small = dref.resize(A[None], h // 4, w // 4, 0)[0]
    back = np.zeros((h, w, 12), np.float32)
    back[..., 5:9] = dref.resize(small[None, ..., 3:7], h, w, 0)[0]
    return dict(A=A, pre=ref.prelu(fl, slope), mix=ref.mix(w0.reshape(-1, 3), w1.reshape(-1, 3), mask.reshape(-1)),
                small=small, back=back, plain=dref.resize(img[None], h // 4, w // 4, 2)[0])


def test_grid_is_torch_linspace():
    """the grid: torch.linspace(-1, 1, n)'s float32 values (its CPU kernel's two halves)"""
    for n in (2, 3, 64, 1088, 1920):
        g = ref.grid(n)
        assert g[0] == -1 and g[-1] == 1 and (np.diff(g) > 0).all()
        assert np.abs(g - np.linspace(-1, 1, n)).max() < 2e-7


def test_records_cpu_same_bits(real):
    env, p, slope = program(real)
    p.run()
    want = expected(real, env, slope)
    for name in ("A", "pre", "small", "back", "plain"):
        assert (bits(env[name]) == bits(want[name]).reshape(env[name].shape)).all(), name
    # ENC_MIX: the same operations; the C library's expf and NumPy's exp may differ in the last bit
    assert np.abs(env["mix"] - want["mix"]).max() <= 2.5e-7


def test_records_python_steps(real):
    env, p, slope = program(real)
    p.run_py()
    want = expected(real, env, slope)
    for name in ("A", "pre", "small", "back", "plain"):
        assert (bits(env[name]) == bits(want[name]).reshape(env[name].shape)).all(), name
    assert np.abs(env["mix"] - want["mix"]).max() <= 2.5e-7


def test_resize_without_strides_unchanged(real):
    """a record of ENC_RESIZE's 9 operands (as before its strides) and one of 13 with the defaults
    give the same bits, both ways and in all three modes"""
    from reclank.ir.opcodes import ENC_RESIZE
    img, _, _ = inputs(real)
    h, w = img.shape[:2]
    for mode, (H, W) in ((0, (37, 61)), (1, (90, 170)), (2, (16, 32))):
        outs = []
        for nops in (9, 13):
            c = _EncCompiler(1e-6, pack=True)
            c.env.update(x=img.reshape(-1, 3).copy(), y=np.zeros((H * W, 3), np.float32))
            c.compile(("seq",))
            args = [c.env["x"], c.env["y"], 1, h, w, H, W, 3, mode] + ([3, 0, 3, 0] if nops == 13 else [])
            c.p.emit(ENC_RESIZE, *args)
            p = c.p.finish()
            p.threads = p.buf[3] = 2
            p.run()
            outs.append(c.env["y"].copy())
            c.env["y"][:] = 0
            p.run_py()
            outs.append(c.env["y"].copy())
        want = dref.resize(img[None], H, W, mode)[0].reshape(-1, 3)
        for o in outs:
            assert (bits(o) == bits(want)).all(), mode


def test_schedule_factor_and_fps():
    s = V.schedule(5, Fraction(30), factor=2)
    assert len(s) == 10 and s[:4] == [(0, 0), (0, Fraction(1, 2)), (1, 0), (1, Fraction(1, 2))]
    assert s[-2:] == [(4, 0), (4, 0)]                        # the last frame repeated to the length
    s = V.schedule(4, Fraction(15), factor=4)
    assert len(s) == 16 and [t for i, t in s[:4]] == [0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]
    s = V.schedule(5, Fraction(24000, 1001), fps=Fraction(60000, 1001))   # 23.976 -> 59.94: t in steps of 0.4
    assert [t for i, t in s[:6]] == [0, Fraction(2, 5), Fraction(4, 5), Fraction(1, 5), Fraction(3, 5), 0]
    assert [i for i, t in s[:6]] == [0, 0, 0, 1, 1, 2]
    assert len(s) == 13                                      # 5 frames / 23.976 fps of 59.94
    with pytest.raises(ValueError):
        V.schedule(5, Fraction(30), fps=Fraction(24))


def test_schedule_variable_rate():
    pts = [0.0, 0.04, 0.12, 0.16]                            # a frame missing at 0.08
    s = V.schedule(4, Fraction(25), fps=Fraction(50), pts=pts)
    ts = [(i, round(float(t), 6)) for i, t in s]
    assert ts[:6] == [(0, 0.0), (0, 0.5), (1, 0.0), (1, 0.25), (1, 0.5), (1, 0.75)]
    assert ts[6:] == [(2, 0.0), (2, 0.5), (3, 0.0), (3, 0.0)]


def test_cut_detector(real):
    """two frames of one clip are no cut; a frame of another clip after them is one (with RIFE's
    residual and with the thumbnails alone); a fire right after a fire is held back"""
    th = [(real["th32"][i], real["thgrey"][i]) for i in range(3)]
    rd = real["probe_rd"]
    det = V.CutDetector()
    cut, s, _ = det(th[0], th[1], tuple(rd[0]))
    assert not cut and s > 0.45
    cut, s, _ = det(th[1], th[2], tuple(rd[1]))
    assert cut and s < 0.2
    assert not det(th[2], th[1], tuple(rd[1]))[0]          # (a fire one pair after a fire)
    old = V.CutDetector()
    assert not old(th[0], th[1])[0] and old(th[1], th[2])[0]
    assert V.ssim(th[0][0], th[0][0]) == pytest.approx(1.0)


@pytest.mark.skipif(not os.environ.get("RECLANK_RIFE"), reason="RECLANK_RIFE: the folder of flownet.safetensors")
def test_probe_residual_cpu(real):
    """RIFE's residual of the probe frames on the CPU: the values the cut test was given"""
    from reclank.models.vfi.rife import RIFE
    p = RIFE(os.environ["RECLANK_RIFE"], gpu=False, cpu_threads=4).prober()
    p.push(real["probe"][0])
    p.push(real["probe"][1])
    r1 = p.residual()
    p.push(real["probe"][2])
    r2 = p.residual()
    assert np.allclose([r1, r2], real["probe_rd"], rtol=1e-5)


def _box_pass(img, bounds, k, axis):
    """one of PIL's 8-bit resample passes (ENC_RESAMPLE_U8's integers) in NumPy"""
    a = np.moveaxis(img, axis, 0).astype(np.int64)
    out = np.empty((len(bounds),) + a.shape[1:], np.uint8)
    for o, (x0, n) in enumerate(bounds):
        acc = (1 << 21) + np.tensordot(k[o, :n].astype(np.int64), a[x0:x0 + n], axes=(0, 0))
        out[o] = np.clip(acc >> 22, 0, 255)
    return np.moveaxis(out, 0, axis)


def _sizes(real):
    """our frames as is, three times larger (each pixel repeated), cropped to odd sizes, transposed"""
    big = np.repeat(np.repeat(real["probe"][2], 3, 0), 3, 1)
    out = [real["frames"][0], real["probe"][0], big, big[:-4, :-3], real["frames"][1][:-1, :-5]]
    return out + [np.ascontiguousarray(f.transpose(1, 0, 2)) for f in out]


def test_probe_plan_is_pil_box(real):
    """video.probe_plan's passes (the GPU's ENC_RESAMPLE_U8, here in NumPy integers) give PIL's BOX
    resize of probe_frame bit for bit: down by 3 and by odd ratios, up, the same size, portrait"""
    for f in _sizes(real):
        p = V.probe_plan(*f.shape[:2])
        x = f
        if p["horiz"] is not None:
            bh, kh, y0, y1 = p["horiz"]
            x = _box_pass(f[y0:y1], bh, kh, 1)
        if p["vert"] is not None:
            x = _box_pass(x, *p["vert"], 0)
        assert np.array_equal(x, V.probe_frame(f)), f.shape


def test_thumbs_of_parts(real):
    """thumbs() from its parts (the taps at thumb_plan's rows and columns, the grey blocks' channel
    sums: the GPU's ENC_THUMB_U8) the same bits"""
    for f in _sizes(real):
        p = V.thumb_plan(*f.shape[:2])
        sums = np.array([[int(f[a:b, c:d].astype(np.int64).sum()) for c, d in zip(p["bx0"], p["bx1"])]
                         for a, b in zip(p["by0"], p["by1"])])
        t32, grey = V.thumbs_of(p, f[np.ix_(p["rows"], p["cols"])], sums)
        w32, wgrey = V.thumbs(f)
        assert np.array_equal(bits(t32), bits(w32)) and np.array_equal(bits(grey), bits(wgrey)), f.shape


def test_mean_abs_bits(real):
    """rife.mean_abs: the bits of np.abs(a - b).mean(-1).mean() (RIFE.residual's means)"""
    from reclank.models.vfi.rife import mean_abs
    a, b = real["out_05"], real["out_025"]
    x0, x1 = (f.astype(np.float32) / f32(255) for f in real["frames"])
    for u, v in ((a, b), (x0, x1), (a[:-3, 5:], x1[:-3, 5:])):
        assert mean_abs(u, v) == float(np.abs(u - v).mean(-1).mean())


def test_deconv_scatter_inverts_gather():
    """deconv_scatter (ENC_CONV_MM's scatter of a deconv's phases) is the inverse of deconv_index"""
    from reclank.models.vfi.rife import deconv_index, deconv_scatter
    for h, w in ((3, 5), (17, 30)):
        for shuffle in (False, True):
            q = deconv_index(h, w, shuffle).reshape(-1)
            s = deconv_scatter(h, w, shuffle)
            assert s.size == q.size and np.array_equal(s[q], np.arange(q.size))


@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get("RECLANK_RIFE"), reason="RECLANK_RIFE: the folder of flownet.safetensors")
def test_gpu_byte_path(real):
    """the GPU's byte path (frames as their bytes, the padding, crop, thumbnails' inputs and probe frame
    on the GPU; deconvs by ENC_CONV_MM's scatter) against the GPU's previous programs and the host: the
    same output bytes, thumbnails and probe frames"""
    from reclank.models.vfi.rife import RIFE
    new, old = RIFE(os.environ["RECLANK_RIFE"], gpu=True), RIFE(os.environ["RECLANK_RIFE"], gpu=True)
    old.conv_mm = False
    big = [np.repeat(np.repeat(f, 3, 0), 3, 1)[:-2, :-1] for f in real["frames"]]
    for fr in (list(real["frames"]), big):
        h, w = fr[0].shape[:2]
        new.reset()
        old.reset()
        new.aux_plan(h, w, V.thumb_plan(h, w), V.probe_plan(h, w))
        for f in fr:
            new.push(f)
            old.push(f)
            a = new.aux()
            t32, grey = V.thumbs_of(V.thumb_plan(h, w), a["taps"], a["sums"])
            w32, wgrey = V.thumbs(f)
            assert np.array_equal(bits(t32), bits(w32)) and np.array_equal(bits(grey), bits(wgrey))
            assert np.array_equal(a["probe"], V.probe_frame(f))
        for t in (0.5, 0.25):
            assert np.array_equal(new.frame(t), old.frame(t)), (h, w, t)
    new.runner.release()
    old.runner.release()


class _Blend:
    """a stand-in model for the pipeline's plumbing: the mean of the pair"""

    def __init__(self):
        self.frames = []

    def runners(self):
        return {}

    def reset(self):
        self.frames = []

    def push(self, f):
        self.frames = (self.frames + [f.astype(np.float32)])[-2:]

    def frame(self, t):
        return np.rint((1 - t) * self.frames[0] + t * self.frames[1]).astype(np.uint8)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_video_plumbing(real, tmp_path):
    """an MP4 of our two frames (each 4 times, 24 fps, an audio track) doubled and to 60 fps: the
    frame counts, the rates, the audio kept; slowmo without audio"""
    src = str(tmp_path / "in.mp4")
    fr = np.concatenate([np.repeat(real["frames"][:1], 4, 0), np.repeat(real["frames"][1:], 4, 0)])
    h, w = fr.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
                    "-r", "24", "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", src], input=fr.tobytes(), check=True)

    def count(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                              "stream=codec_type,nb_read_frames,r_frame_rate", "-of", "csv=p=0", path],
                             capture_output=True, text=True, check=True).stdout.split()
        return out

    dst = str(tmp_path / "x2.mp4")
    r = V.interpolate_video(_Blend(), src, dst, factor=2)
    assert r["frames_in"] == 8 and r["frames"] == 16 and r["audio"]
    assert r["cuts"] == []
    info = V.probe(dst)
    assert info["r_rate"] == 48 and info["audio"] == "aac"
    dst = str(tmp_path / "f60.mp4")
    r = V.interpolate_video(_Blend(), src, dst, fps="60")
    assert r["frames"] == 20 and V.probe(dst)["r_rate"] == 60
    dst = str(tmp_path / "slow.mp4")
    r = V.interpolate_video(_Blend(), src, dst, factor=4, slowmo=True)
    assert r["frames"] == 32 and not r["audio"] and V.probe(dst)["audio"] is None
    assert V.probe(dst)["r_rate"] == 24


@pytest.mark.skipif(not os.environ.get("RECLANK_RIFE"), reason="RECLANK_RIFE: the folder of flownet.safetensors")
def test_rife_cpu_against_reference(real):
    """the model on the CPU against rife_reference.py: at most a level of 255"""
    from reclank.models.vfi.rife import RIFE
    R = RIFE(os.environ["RECLANK_RIFE"], gpu=False, cpu_threads=4)
    a, b = real["frames"]
    for t, key in ((0.5, "out_05"), (0.25, "out_025")):
        R.reset()
        R.push(a)
        R.push(b)
        got = R.frame(t)
        want = np.clip(np.rint(real[key] * f32(255)), 0, 255).astype(np.uint8)
        assert np.abs(got.astype(int) - want).max() <= 1, t
        h, w, Hp, Wp, _ = R.size
        out = R.outputs(["out"])["out"].reshape(Hp, Wp, 3)[:h, :w]
        assert np.abs(out - real[key]).max() < 1e-4, t


def test_encode_colors():
    """the encode's yuv420p by the source's tags (media.yuv_colors: tests/unit/test_media.py)"""
    src = {"color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709", "color_range": "pc"}
    cmd = V.encode_cmd("in.mp4", "out.mp4", 64, 48, Fraction(24), "h264", 16, None, src, True)
    assert cmd[cmd.index("-vf") + 1] == "scale=out_color_matrix=bt709:out_range=pc"
    assert cmd[cmd.index("-colorspace") + 1] == "bt709" and cmd[cmd.index("-color_range") + 1] == "pc"


def test_server_options():
    """the route's fields: factor or fps (the NTSC rates exact), the checks of a bad request"""
    from server.interpolate import options

    def form(**kw):
        return {k: [(str(v).encode(), None)] for k, v in kw.items()}

    o = options(form())
    assert o["factor"] == 2 and o["fps"] is None and o["scene_cuts"] and o["codec"] == "h264" and o["crf"] == 16
    assert options(form(fps="59.94"))["fps"] == "60000/1001"
    assert options(form(fps="23.976"))["fps"] == "24000/1001"
    assert options(form(fps="60000/1001"))["fps"] == "60000/1001"
    assert options(form(fps="50"))["fps"] == "50"
    assert options(form(factor=4, scene_cuts="off", scale="0.5", slowmo="true"))["scale"] == 0.5
    for bad in (dict(factor=2, fps=60), dict(factor=1), dict(factor=17), dict(fps="x"), dict(scene_cuts="maybe"),
                dict(scale=2), dict(slowmo="true", fps=60), dict(codec="vp9"), dict(crf=60), dict(max_frames=1)):
        with pytest.raises(ValueError):
            options(form(**bad))
