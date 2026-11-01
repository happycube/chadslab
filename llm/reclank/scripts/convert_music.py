#!/usr/bin/env python3
"""The weights of the music embeddings (docs/MUSIC_EMBED_PLAN.md) as
safetensors for reclank, float32 as trained:

    --clap ORIG.pt TOKDIR OUT   LAION-CLAP's original checkpoint
        (lukewys/laion_clap music_audioset_epoch_15_esc_90.14.pt) under the
        names of transformers' ClapModel. Not laion/larger_clap_music: that
        upload is an untrained model (zero biases; every text has the same
        vector). TOKDIR gives the tokenizer of RoBERTa (tokenizer.json) and
        the config files.
    --muq DIR XLMR OUT          MuQ-MuLan's pytorch_model.bin with its names;
        XLMR is a directory of xlm-roberta-base (its tokenizer.json).

Run it in a venv with torch (the venv of scripts/music_reference.py).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

import numpy as np

RENAME = [("text_branch", "text_model"), ("audio_branch", "audio_model.audio_encoder"),
          ("attn", "attention.self"), ("self.proj", "output.dense"), ("mlp.fc1", "intermediate.dense"),
          ("mlp.fc2", "output.dense"), ("norm1", "layernorm_before"), ("norm2", "layernorm_after"),
          ("bn0", "batch_norm")]
# the loss's MLPs, the front end (reclank's own), the tscam head, the shift masks (made again)
UNUSED = ("text_transform", "audio_transform", "spectrogram_extractor", "logmel_extractor", "tscam_conv",
          "head.", "attn_mask")


def clap_names(sd):
    out = {}
    for k, v in sd.items():
        k = k.removeprefix("module.")
        if any(u in k for u in UNUSED):
            continue
        for a, b in RENAME:
            k = k.replace(a, b)
        m = re.match(r"(.*_projection)\.(\d+)\.(.*)", k)
        if m:
            k = "%s.linear%d.%s" % (m.group(1), 1 if m.group(2) == "0" else 2, m.group(3))
        if "audio" in k and "qkv" in k:
            for name, part in zip(("query", "key", "value"), v.chunk(3, 0)):
                out[k.replace("qkv", name)] = part.clone()
            continue
        out[k] = v
    return out


def save(sd, out):
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    sd = {k: v.contiguous() for k, v in sd.items() if v.dtype.is_floating_point or k.endswith("position_ids")}
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    print("wrote %d tensors to %s" % (len(sd), out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clap", nargs=3, metavar=("ORIG", "TOKDIR", "OUT"))
    ap.add_argument("--muq", nargs=3, metavar=("DIR", "XLMR", "OUT"))
    a = ap.parse_args()
    import torch
    if a.clap:
        orig, tokdir, out = a.clap
        from numpy._core.multiarray import scalar
        from transformers import ClapConfig, ClapModel
        # the checkpoint's epoch and metrics are numpy scalars: allow those alone
        safe = [(scalar, "numpy.core.multiarray.scalar"), np.dtype] + \
            [type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32")]
        with torch.serialization.safe_globals(safe):
            sd = clap_names(torch.load(orig, map_location="cpu", weights_only=True)["state_dict"])
        model = ClapModel(ClapConfig.from_pretrained(tokdir)).eval()
        r = model.load_state_dict(sd, strict=False)
        # the buffer of the token types (zeros) is made by the model
        assert not r.unexpected_keys and set(r.missing_keys) <= {"text_model.embeddings.token_type_ids"}, r
        save(model.state_dict(), out)
        for f in ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt"):
            if os.path.exists(os.path.join(tokdir, f)):
                shutil.copy(os.path.join(tokdir, f), out)
    if a.muq:
        d, xlmr, out = a.muq
        sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        save(sd, out)
        shutil.copy(os.path.join(d, "config.json"), out)
        shutil.copy(os.path.join(xlmr, "tokenizer.json"), out)


if __name__ == "__main__":
    main()
