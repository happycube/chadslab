"""An NVFP4 checkpoint of Qwen3.6 MoE (reclank/formats/q35nv_st.py) on its real tensors: the detection
of the checkpoint and of its weightkeeper plugin, and the KQ_NVX stacks of the experts (and the shared
expert) against the float32 values of the same NVFP4 matrices. Skipped without the checkpoint
(RECLANK_Q36_NVFP4, else models/Qwen3.6-35B-A3B-NVFP4 under RECLANK_MODELS or ~/models)."""
import os

import numpy as np
import pytest

from reclank.backends.cpu import cops

PATH = os.environ.get("RECLANK_Q36_NVFP4") or os.path.join(
    os.path.expanduser(os.environ.get("RECLANK_MODELS", "~/models")), "Qwen3.6-35B-A3B-NVFP4")

pytestmark = pytest.mark.skipif(not os.path.isdir(PATH), reason="no NVFP4 checkpoint of Qwen3.6")


def test_detect():
    from reclank.formats.q35nv_st import is_q35nv
    from reclank.models.qwen import keep_q35nv
    from reclank.runtime import weightkeeper as W
    assert is_q35nv(PATH)
    assert W.plugin(PATH) is keep_q35nv.PLUGIN


def test_nvx_stacks_are_the_nvfp4_values():
    from reclank.formats.q35nv_st import Q35NVSource, nvfp4_dequant
    src = Q35NVSource(PATH)
    try:
        g = "blk.5.ffn_down_exps.weight"
        cols, rows, _e = src.tensors[g][0]
        stack = src.raw(g)[0]
        for e in (0, 37, 255):
            p = "model.language_model.layers.5.mlp.experts.%d.down_proj." % e
            want = nvfp4_dequant(src._get(p + "weight", None), src._get(p + "weight_scale", None),
                                 src._scalar(p + "weight_scale_2"))
            got = cops.kq_rows(stack, 53, cols, np.arange(e * rows, (e + 1) * rows))
            assert np.array_equal(got, want)
        s = "blk.5.ffn_gate_shexp.weight"
        p = "model.language_model.layers.5.mlp.shared_expert.gate_proj."
        want = nvfp4_dequant(src._get(p + "weight", None), src._get(p + "weight_scale", None),
                             src._scalar(p + "weight_scale_2"))
        assert np.array_equal(src.dequant(s), want)
    finally:
        src.close()
