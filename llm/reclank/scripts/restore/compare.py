"""Restorations side by side: DEST/PIECE/{0 original,1 restore,2 apollo,3 audiosr,4 a2sb}.flac, an A/B
montage per piece (the same 12 s of each version in that order, loudness matched, 0.7 s apart) and summary.md (the
band each version carries: where its average spectrum is within 30/50 dB of its peak; the level above 4 kHz).
RESTORE_IN, RESTORE_OUT, RESTORE_DEST: the inputs, the outputs, the listening folder."""
import glob, os, subprocess
import numpy as np, soundfile as sf

SR_IN = os.environ["RESTORE_IN"]        # the pieces as 44.1 kHz WAVs (the sandbox's /in)
OUT = os.environ["RESTORE_OUT"]         # the models' outputs, OUT/MODEL/NAME.wav (the sandbox's /out)
DST = os.environ["RESTORE_DEST"]        # the listening folder
VERS = [("0 original", None), ("1 restore", "restore"), ("1b restore less bright", "restore_dim"), ("2 apollo", "apollo"), ("3 audiosr", "audiosr"), ("4 a2sb", "a2sb")]
RATE = 48000


def load(path):
    """stereo float at 48 kHz"""
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "2", "-ar", str(RATE), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2)


def band(x):
    y = x.mean(1)
    n = 8192
    fr = y[:len(y) // n * n].reshape(-1, n) * np.hanning(n)
    P = 10 * np.log10((np.abs(np.fft.rfft(fr, axis=1)) ** 2).mean(0) + 1e-20)
    hz = np.fft.rfftfreq(n, 1 / RATE)
    sm = np.convolve(P, np.ones(9) / 9, "same")
    pk = sm[(hz > 100) & (hz < 4000)].max()
    e = lambda db: int(hz[np.where(sm > pk - db)[0][-1]])  # noqa: E731
    return e(30), e(50), float(sm[(hz > 4000) & (hz < 16000)].mean() - pk)


def main():
    os.makedirs(DST, exist_ok=True)
    rows = []
    for src in sorted(glob.glob(SR_IN + "/*.wav")):
        name = os.path.splitext(os.path.basename(src))[0]
        d = os.path.join(DST, name)
        os.makedirs(d, exist_ok=True)
        have = []
        for label, m in VERS:
            p = src if m is None else os.path.join(OUT, m, name + ".wav")
            if not os.path.exists(p):
                rows.append((name, label, None))
                continue
            x = load(p)
            sf.write(os.path.join(d, label + ".flac"), x, RATE, subtype="PCM_24")
            have.append((label, x))
            rows.append((name, label, band(x)))
        refp = os.path.join(d, "5 reference (aligned).flac")            # truth.py's, when there is a clean recording
        if os.path.exists(refp):
            have.append(("5 reference", load(refp)))
        for old in glob.glob(os.path.join(d, "AB *.flac")):
            os.remove(old)
        # the A/B montage: 12 s from 40 % into the piece, each version loudness matched to the original
        if len(have) > 1:
            ref = have[0][1]
            a = int(len(ref) * 0.4)
            b = a + 12 * RATE
            r0 = np.sqrt(np.mean(ref[a:b] ** 2)) + 1e-9
            gap = np.zeros((int(0.7 * RATE), 2), np.float32)
            parts = []
            for label, x in have:
                seg = x[a:b]
                seg = seg * (r0 / (np.sqrt(np.mean(seg ** 2)) + 1e-9))
                parts += [seg, gap]
            mont = np.concatenate(parts)
            mont = mont / max(1.0, np.abs(mont).max() / 0.98)
            sf.write(os.path.join(d, "AB " + ", ".join(l.split(" ", 1)[1] for l, _ in have) + ".flac"), mont, RATE, subtype="PCM_24")
    L = ["# m1/m2 restorations", "", "The band each version carries (its average spectrum within 30 and 50 dB of its peak) "
         "and its average level at 4-16 kHz relative to that peak. Files are in each piece's folder; `AB *.flac` plays the "
         "same 12 s of every version in order.", "", "| Piece | Version | -30 dB edge | -50 dB edge | 4-16 kHz level |",
         "|---|---|---|---|---|"]
    for name, label, b in rows:
        L.append("| %s | %s | %s |" % (name, label, "missing | | " if b is None else "%d Hz | %d Hz | %.0f dB" % b))
    open(os.path.join(DST, "summary.md"), "w").write("\n".join(L) + "\n")
    print("compare: %s" % DST)


if __name__ == "__main__":
    main()
