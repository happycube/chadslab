"""The engine's image spans reach the encoder (models/dgemma/engine.py)."""
import numpy as np

from reclank.encoders import media as MD
from reclank.models.dgemma import sampler as S
from reclank.models.dgemma.engine import DGemmaEngine, DGPrompt


class _Tok:
    def encode(self, text):
        return [ord(c) % 50 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "x"


class _Model:
    C = 4
    max_len = 64

    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append("reset")

    def reuse(self, target):
        self.calls.append(("reuse", list(target)))
        return 0

    def encode(self, ids, spans=None):
        self.calls.append(("encode", list(ids), spans))

    def steps(self):
        def step(cur, temp, ban, u):
            return (np.zeros(self.C, np.float32), np.zeros(self.C, np.int64),
                    np.zeros(self.C, np.int64), np.ones(self.C, np.float32),
                    np.ones(self.C, np.float32))
        return step


class _Sampler:
    rng = np.random.default_rng(0)


def test_generate_sends_the_spans_to_the_encoder():
    m = _Model()
    s = S.Settings(canvas=4, vocab=64, max_steps=2, entropy_bound=0.1, confidence=0.005,
                   stability=0, eos=(1,))
    eng = DGemmaEngine(m, _Tok(), s, 64)
    spans = [MD.Span(1, np.zeros((2, 8), np.float32), True, "image", "k")]
    p = DGPrompt([1, 2, 3])
    p.spans = spans
    p.template = None
    p.options = {}
    out = list(eng.generate(p, 4, _Sampler(), []))
    assert out == [0, 0, 0, 0]
    assert "reset" in m.calls
    first = next(c for c in m.calls if isinstance(c, tuple) and c[0] == "encode")
    assert [int(v) for v in first[1]] == [1, 2, 3]
    assert first[2] is not None and first[2][0] is spans[0]
    assert not any(isinstance(c, tuple) and c[0] == "reuse" for c in m.calls)
