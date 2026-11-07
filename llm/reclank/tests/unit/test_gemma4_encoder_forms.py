"""The forms of the Gemma 4 encoders' weights (models/gemma4/encoders.py weight_form): the names, and on
a real mmproj (Nemotron 3.5 Bolt E4B's, RECLANK_E4B_MMPROJ; skipped without it) the BF12 encoder
against the bfloat16 one on a real image (a frame of tests/unit/data/vfi_real.npz; the tower's rows
within 1e-4)."""
import os

import numpy as np
import pytest

from reclank.models.gemma4.encoders import weight_form

MM = os.environ.get("RECLANK_E4B_MMPROJ") or os.path.join(
    os.path.expanduser(os.environ.get("RECLANK_MODELS", "~/models")),
    "NVIDIA-Nemotron-3.5-Bolt-E4B-BF16", "nemotron-3.5-bolt-e4b-mmproj.gguf")
FRAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vfi_real.npz")


def test_weight_form_names():
    assert weight_form(True) == "q8" and weight_form("on") == "q8"
    assert weight_form(False) == weight_form(None) == weight_form("off") == "bf16"
    assert weight_form("bf12") == "bf12"
    with pytest.raises(ValueError):
        weight_form("q4")


@pytest.mark.skipif(not os.path.exists(MM), reason="no gemma4v mmproj of an E4B")
def test_bf12_encoder_is_the_bf16_one():
    from PIL import Image
    from reclank.models.gemma4.encoders import image_patches16, load_embedder
    from reclank.models.gemma4.unified import image_pixels
    frame = Image.fromarray(np.load(FRAMES)["probe"][0])
    patches, pos = image_patches16(image_pixels(frame, 70))
    a = load_embedder(MM, q8=False).vision.encode(patches, pos)
    e = load_embedder(MM, q8="bf12")
    assert e.form == "bf12" and e.bf12 > 200
    b = e.vision.encode(patches, pos)
    assert np.linalg.norm(b - a) / np.linalg.norm(a) < 1e-4
