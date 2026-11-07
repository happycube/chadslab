"""The image prompt of the dgemma engine (models/dgemma/engine.py) without a model."""
import numpy as np

from reclank.encoders import media as MD
from reclank.models.dgemma import sampler as S
from reclank.models.dgemma.engine import DGemmaEngine


class _Tok:
    """A tokenizer that makes one id for each character and the image placeholder."""

    def encode(self, text):
        out = []
        for i, part in enumerate(text.split("<|image|>")):
            if i:
                out.append(MD.IMAGE_TOKEN)
            out.extend(ord(c) for c in part)
        return out


class _Emb:
    hidden = 8
    bidir = True

    def image(self, data, budget):
        rows = np.arange(budget * self.hidden, dtype=np.float32).reshape(budget, self.hidden)
        return rows, (1, budget)


class _Src:
    def image(self, part, budget):
        return b"image-bytes", budget


def _engine(embedder=True):
    eng = DGemmaEngine(None, _Tok(), S.Settings(canvas=8, vocab=1024), 1024)
    if embedder:
        eng.embedder = _Emb()
        eng.image_budget = 4
    return eng


def _messages():
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
        {"type": "text", "text": "hi"}]}]


def test_image_prompt_places_one_span():
    p = _engine().prompt_ids(_messages(), False, source=_Src())
    assert len(p.spans) == 1
    sp = p.spans[0]
    assert sp.rows.shape == (4, 8) and sp.kind == "image"
    assert p[sp.start - 1] == MD.BOI
    assert p[sp.end] == MD.EOI
    assert all(v == MD.IMAGE_TOKEN for v in p[sp.start:sp.end])
    assert p.template is None


def test_image_prompt_with_a_response_template():
    msgs = _messages() + [{"role": "assistant", "content": "Answer: [[2]]"}]
    p = _engine().prompt_ids(msgs, False, source=_Src())
    assert p.template == "Answer: [[2]]"
    assert len(p.spans) == 1


def test_image_needs_an_embedder():
    try:
        _engine(embedder=False).prompt_ids(_messages(), False, source=_Src())
    except ValueError as e:
        assert "mmproj" in str(e)
    else:
        raise AssertionError("no error without an embedder")


def test_image_needs_a_source():
    try:
        _engine().prompt_ids(_messages(), False, source=None)
    except ValueError as e:
        assert "media source" in str(e)
    else:
        raise AssertionError("no error without a source")


def test_audio_is_refused():
    msgs = [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "eA=="}},
        {"type": "text", "text": "hi"}]}]
    try:
        _engine().prompt_ids(msgs, False, source=_Src())
    except ValueError as e:
        assert "images only" in str(e)
    else:
        raise AssertionError("no error for audio")
