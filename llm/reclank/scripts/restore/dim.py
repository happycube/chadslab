"""RESTORE's output with less brightness: a high shelf (RBJ biquad, applied forwards and backwards so no phase
shift; each pass half the gain) on every OUT/restore/NAME.wav -> OUT/restore_dim/NAME.wav (OUT: RESTORE_OUT).
    dim.py [GAIN_DB (-4)] [SHELF_HZ (7000)]"""
import glob, os, sys
import numpy as np, soundfile as sf
from scipy import signal

SRC = os.path.join(os.environ["RESTORE_OUT"], "restore")
DST = os.path.join(os.environ["RESTORE_OUT"], "restore_dim")


def shelf(gain_db, f0, fs, S=0.7):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    c = np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * alpha)]
    a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def main():
    g = float(sys.argv[1]) if len(sys.argv) > 1 else -4.0
    f0 = float(sys.argv[2]) if len(sys.argv) > 2 else 7000.0
    os.makedirs(DST, exist_ok=True)
    for p in sorted(glob.glob(SRC + "/*.wav")):
        x, fs = sf.read(p, dtype="float64", always_2d=True)
        b, a = shelf(g / 2, f0, fs)                    # filtfilt applies it twice
        y = signal.filtfilt(b, a, x, axis=0)
        sf.write(os.path.join(DST, os.path.basename(p)), y.astype(np.float32), fs, subtype="FLOAT")
    print("restore_dim: %d files, shelf %+.1f dB above %.0f Hz" % (len(glob.glob(SRC + "/*.wav")), g, f0))


if __name__ == "__main__":
    main()
