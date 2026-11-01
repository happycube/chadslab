#!/usr/bin/env python3
"""PyTorch checkpoints (.pth / .pt) as safetensors for reclank, which reads only safetensors and
needs no torch. Each file is loaded with torch.load(weights_only=True) (a pickle that names
anything but tensors and plain containers is refused) and its tensors written as they are
(dtype, shape, names) to NAME.safetensors beside it (or in --out); the checkpoint's other entries
that JSON can hold (MoGe-2's model_config) go to NAME.json. Then the pages of both files are
dropped from the page cache.

    pt_to_safetensors.py FILE.pth [FILE.pt ...] [--key KEY[,KEY...]] [--out DIR]

The tensors: the checkpoint itself when it is a dict of tensors, a bare tensor (saved under the
file's stem: SeedVR2's pos_emb.pt), else its entry KEY (the first of the keys given that it has,
such as "params_ema,params" for Real-ESRGAN; default: the one entry that is a dict of tensors,
such as "model"). The depth models of models/depth:
Video-Depth-Anything-Small's video_depth_anything_vits.pth and MoGe-2's model.pt. Run it in a
venv with torch and safetensors (the generation reference venv)."""
from __future__ import annotations

import argparse
import json
import os


def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fdatasync(fd)
    except OSError:
        pass
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def convert(src, out_dir=None, key=None):
    import torch
    from safetensors.torch import save_file
    obj = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    is_sd = lambda d: isinstance(d, dict) and d and all(isinstance(v, torch.Tensor) for v in d.values())  # noqa: E731
    rest = {}
    stem = os.path.splitext(os.path.basename(src))[0]
    if isinstance(obj, torch.Tensor):
        sd = {stem: obj}
    elif is_sd(obj):
        sd = obj
    else:
        if key is not None:
            key = next((k for k in key.split(",") if k in obj), None)
            if key is None:
                raise ValueError("%s: none of the keys given (it has: %s)" % (src, ", ".join(map(str, obj))))
        if key is None:
            keys = [k for k, v in obj.items() if is_sd(v)]
            if len(keys) != 1:
                raise ValueError("%s: give --key (the dicts of tensors: %s)" % (src, keys))
            key = keys[0]
        sd = obj[key]
        rest = {k: v for k, v in obj.items() if k != key and not isinstance(v, torch.Tensor) and not is_sd(v)}
    out_dir = out_dir or os.path.dirname(os.path.abspath(src))
    dst = os.path.join(out_dir, stem + ".safetensors")
    save_file({k: v.contiguous() for k, v in sd.items()}, dst,
              metadata={"format": "pt", "source": os.path.basename(src)})
    print("%s -> %s: %d tensors" % (src, dst, len(sd)), flush=True)
    if rest:
        side = os.path.join(out_dir, stem + ".json")
        with open(side, "w") as f:
            json.dump(rest, f, indent=1)
        print("%s -> %s: %s" % (src, side, ", ".join(rest)), flush=True)
    drop(src)
    drop(dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--key")
    ap.add_argument("--out")
    a = ap.parse_args()
    for f in a.files:
        convert(f, a.out, a.key)


if __name__ == "__main__":
    main()
