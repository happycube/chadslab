"""Average spectrum of each file: the band that carries music (where the level is within 30/50 dB of its peak),
the hiss floor above it, the encoding."""
import json, subprocess, sys
import numpy as np
SR = 44100
for f in sys.argv[1:]:
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", f], capture_output=True, text=True).stdout)["streams"][0]
    y = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout, np.float32)
    n = 8192
    fr = y[:len(y) // n * n].reshape(-1, n) * np.hanning(n)
    P = 10 * np.log10((np.abs(np.fft.rfft(fr, axis=1)) ** 2).mean(0) + 1e-20)
    hz = np.fft.rfftfreq(n, 1 / SR)
    sm = np.convolve(P, np.ones(9) / 9, "same")
    pk = sm[(hz > 100) & (hz < 4000)].max()
    def edge(db):
        on = np.where(sm > pk - db)[0]
        return int(hz[on[0]]), int(hz[on[-1]])
    bands = {b: round(float(sm[(hz >= a) & (hz < z)].mean() - pk), 1) for b, (a, z) in
             dict(lo50=(0, 50), b100=(80, 150), b3k=(2800, 3400), b5k=(4500, 5500), b8k=(7500, 8500), b12k=(11000, 13000), b16k=(15000, 17000)).items()}
    print("%s  %s %s Hz %s ch %s kb/s  -30dB band %s  -50dB band %s  rel dB %s" % (f.split('/')[-1], info["codec_name"], info["sample_rate"], info["channels"],
          int(info.get("bit_rate", 0)) // 1000, edge(30), edge(50), bands))
