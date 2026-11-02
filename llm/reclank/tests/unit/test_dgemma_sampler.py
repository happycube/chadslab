"""DiffusionGemma's sampler (reclank/models/dgemma/sampler.py) with a stand-in decoder."""
import numpy as np

from reclank.models.dgemma import sampler as S


def _settings():
    return S.Settings(max_steps=12, canvas=16, vocab=64, eos=(1,))


def _decoder(s, want):
    """Logits that put most weight on want[i] at position i (a peaked, fixed answer)."""
    def decode(canvas, sc):
        z = np.full((s.canvas, s.vocab), -4.0, np.float32)
        z[np.arange(s.canvas), want] = 12.0
        return z
    return decode


def test_denoise_converges_to_the_answer():
    s = _settings()
    want = (np.arange(s.canvas) * 5 + 2) % s.vocab
    am, steps = S.denoise(_decoder(s, want), s, np.random.default_rng(0))
    assert np.array_equal(am, want)
    assert 1 <= steps <= s.max_steps


def test_pinned_tokens_stay_and_end_waits():
    s = _settings()
    want = np.full(s.canvas, 1)                 # the decoder wants to end everywhere
    fixed = np.full(s.canvas, -1)
    fixed[10:12] = [7, 8]
    am, _ = S.denoise(_decoder(s, want), s, np.random.default_rng(1), fixed=fixed, no_end=11)
    assert am[10] == 7 and am[11] == 8
    assert not np.isin(am[:11], s.eos).any()


def test_template_and_generate():
    s = _settings()
    words = {"ab": [20, 21], "cd": [22]}
    t = S.parse_template("ab[[3]]cd[[2]]", lambda x: words[x])
    assert t.tolist() == [20, 21, -1, -1, -1, 22, -1, -1]
    want = (np.arange(s.canvas) + 30) % s.vocab
    seen = []
    out, st = S.generate(lambda ids: seen.append(np.asarray(ids).tolist()), _decoder(s, want),
                         [2, 3], 16, s, np.random.default_rng(2), fixed=t)
    # the pinned start is a prefix: encoded with the prompt and first in the result
    assert seen[0] == [2, 3, 20, 21]
    assert out[:2] == [20, 21]
    # the pinned token after it keeps its place (canvas offset 3)
    assert out[2 + 3] == 22
    assert len(st.steps) >= 1
