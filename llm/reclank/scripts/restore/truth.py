"""Each restoration of one piece against a clean recording of it (the ground truth).
1. Alignment: the reference's speed (tape speed error) and offset against the original, by cross-correlating
   their log-energy envelopes in the band both carry (150-2500 Hz), speeds 0.90-1.10, then a finer search.
2. The reference resampled onto the original's timeline -> DEST/PIECE/5 reference (aligned).flac
3. Per version: the log-spectral distance (dB, RMS over frames and bins of the 20*log10 STFT magnitudes, each
   version loudness matched to the reference in 150-2500 Hz) in the band the tape kept (150-2500 Hz), above it
   (3-8 kHz, 8-16 kHz), and the long-term spectrum's error per band.
    truth.py PIECE REFERENCE   (PIECE: a folder of compare.py's; REFERENCE: any file ffmpeg reads)
"""
import os, subprocess, sys
import numpy as np, soundfile as sf
from scipy import signal

RATE = 48000
DST = os.environ["RESTORE_DEST"]        # compare.py's folder: PIECE/0 original.flac and the versions
VERS = ["0 original", "1 restore", "1b restore less bright", "2 apollo", "3 audiosr", "4 a2sb"]


def load(path, rate=RATE):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).astype(np.float64)


def env(x, rate, hop=0.01):
    """log energy in 150-2500 Hz, one value per hop s"""
    sos = signal.butter(4, [150, 2500], "bandpass", fs=rate, output="sos")
    y = signal.sosfilt(sos, x)
    n = int(rate * hop)
    e = np.log10((y[:len(y) // n * n].reshape(-1, n) ** 2).mean(1) + 1e-10)
    e = e - np.convolve(e, np.ones(101) / 101, "same")     # the level's slow drift out
    return (e - e.mean()) / (e.std() + 1e-9)


def best_lag(a, b):
    """lag of b in a and the normalized peak"""
    n = len(a) + len(b)
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    k = int(np.argmax(c))
    lag = k if k < len(a) else k - n
    return lag, c[k] / np.sqrt((a ** 2).sum() * (b ** 2).sum())


def align(orig, ref):
    eo = env(orig, RATE)
    best = None
    for grid in (np.arange(0.90, 1.1001, 0.005), None):
        if grid is None:
            s0 = best[0]
            grid = np.arange(s0 - 0.006, s0 + 0.0061, 0.0005)
        for s in grid:
            # ref played at speed s: its time t lands at t/s
            er = env(ref, RATE * s)                         # the same samples read as if at rate RATE*s
            lag, pk = best_lag(eo, er)
            if best is None or pk > best[2]:
                best = (float(s), lag, float(pk))
    return best


def stft_db(x):
    f, t, Z = signal.stft(x, RATE, nperseg=2048, noverlap=1536)
    return f, 20 * np.log10(np.abs(Z) + 1e-7)


def main():
    piece, refp = sys.argv[1], sys.argv[2]
    d = os.path.join(DST, piece)
    orig = load(os.path.join(d, "0 original.flac"))
    ref = load(refp)
    s, lag, pk = align(orig, ref)
    # ref sample i is at time i/(RATE*s) on the original's timeline, shifted by lag hops of 0.01 s
    t_ref = np.arange(len(ref)) / (RATE * s) + lag * 0.01
    t = np.arange(len(orig)) / RATE

    def place(t_ref):
        ok = (t >= t_ref[0]) & (t <= t_ref[-1])
        al = np.zeros(len(orig))
        al[ok] = np.interp(t[ok], t_ref, ref)
        return al, ok

    # refine: the local offset in 10 s windows (5 s apart), a straight line through it (weighted by the match)
    for it in range(2):
        al, ok = place(t_ref)
        eo, er = env(orig, RATE), env(al, RATE)
        pts = []
        for a0 in range(0, len(eo) - 1000, 500):
            w = eo[a0:a0 + 1000]
            c = [(np.dot(w, er[a0 + l:a0 + l + 1000]) / np.sqrt((w ** 2).sum() * (er[a0 + l:a0 + l + 1000] ** 2).sum() + 1e-9), l)
                 for l in range(-50, 51) if a0 + l >= 0 and a0 + l + 1000 <= len(er)]
            m, l = max(c)
            if m > 0.5:
                pts.append(((a0 + 500) * 0.01, l * 0.01, m))
        P = np.array(pts)
        c1, c0 = np.polyfit(P[:, 0], P[:, 1], 1, w=P[:, 2])
        t_ref = t_ref - (c0 + c1 * t_ref)
        resid = P[:, 1] - (c0 + c1 * P[:, 0])
        print("  refine %d: %d windows, drift %.2f ms/s, offset %.1f ms, residual %.1f ms rms" % (
            it + 1, len(P), c1 * 1000, c0 * 1000, np.sqrt((resid ** 2).mean()) * 1000))
    al, ok = place(t_ref)
    s_eff = (len(ref) - 1) / RATE / (t_ref[-1] - t_ref[0])
    print("%s: reference speed %.4f (the tape runs %.2f %% %s), offset %.2f s, envelope match %.2f, overlap %.0f s" % (
        piece, s_eff, abs(s_eff - 1) * 100, "fast" if s_eff > 1 else "slow", t_ref[0], pk, ok.sum() / RATE))
    s, lag = s_eff, t_ref[0] * 100
    sf.write(os.path.join(d, "5 reference (aligned).flac"), al.astype(np.float32), RATE, subtype="PCM_24")
    a, b = np.where(ok)[0][[0, -1]]
    a, b = a + RATE, b - RATE
    f, R = stft_db(al[a:b])
    bands = dict(kept=(150, 2500), mid=(3000, 8000), high=(8000, 16000))
    rows = []
    for v in VERS:
        p = os.path.join(d, v + ".flac")
        if not os.path.exists(p):
            continue
        x = load(p)[a:b]
        _, X = stft_db(x)
        n = min(X.shape[1], R.shape[1])
        Xv, Rv = X[:, :n], R[:, :n]
        kb = (f >= 150) & (f <= 2500)
        Xv = Xv + (Rv[kb].mean() - Xv[kb].mean())          # loudness matched in the kept band
        out = {}
        for k, (lo, hi) in bands.items():
            m = (f >= lo) & (f < hi)
            out["lsd_" + k] = float(np.sqrt(((Xv[m] - Rv[m]) ** 2).mean()))
            out["lt_" + k] = float(Xv[m].mean() - Rv[m].mean())   # long-term level vs the reference
        rows.append((v, out))
    L = ["| Version | LSD 150-2500 Hz | LSD 3-8 kHz | LSD 8-16 kHz | level 3-8 kHz vs ref | level 8-16 kHz vs ref |", "|---|---|---|---|---|---|"]
    for v, o in rows:
        L.append("| %s | %.1f dB | %.1f dB | %.1f dB | %+.1f dB | %+.1f dB |" % (v, o["lsd_kept"], o["lsd_mid"], o["lsd_high"], o["lt_mid"], o["lt_high"]))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(d, "vs reference.md"), "w").write(
        "# %s against the reference\n\nReference: `%s`, speed %.4f, offset %.2f s, envelope match %.2f. Log-spectral distance "
        "(lower is closer; versions loudness matched in 150-2500 Hz).\n\n%s\n" % (piece, os.path.basename(refp), s, lag * 0.01, pk, txt))


if __name__ == "__main__":
    main()
