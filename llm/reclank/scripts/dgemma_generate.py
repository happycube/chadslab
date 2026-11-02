"""Generate with DiffusionGemma 26B-A4B (docs/DIFFUSIONGEMMA_PLAN.md): a chat turn, the canvases
denoised by the checkpoint's sampler, optionally around pinned text.

    python scripts/dgemma_generate.py MODEL_DIR --prompt "Why is the sky blue?"
    python scripts/dgemma_generate.py MODEL_DIR --prompt "..." \\
        --response "[[60]]'What lessons do we learn from this fact, comrades? [[200]]"

--response is a template of the answer: its text is pinned where it stands and each [[n]] is n
free tokens the model writes; after the template the answer goes on freely to an end token or
--max-tokens. Text at the start of the template is a plain prefix. --engine gpu (the default) runs
the BF12 GGUF on the GPU (MODEL_DIR: convert_dgemma_bf12.py's directory; the experts of the layers
that do not fit on the CPU); bf16 the CPU runtime on the checkpoint's bf16 weights (about 55 GB),
q4 on our Q4_0 (KL 0.2-0.7 against float32 on the canvas checks: not for use), ref the float32
NumPy reference (about 103 GB).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.dgemma import sampler as S  # noqa: E402
from reclank.models.gemma4.tokenizer import Tokenizer  # noqa: E402


def chat_ids(tok, prompt, system=None, think=False):
    """The checkpoint's chat template for one user turn (the same ids as transformers'
    apply_chat_template with add_generation_prompt)."""
    text = "<bos>"
    if system or think:
        text += "<|turn>system\n" + ("<|think|>\n" if think else "") + (system or "").strip() + "<turn|>\n"
    text += "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n"
    return tok.encode(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system")
    ap.add_argument("--think", action="store_true", help="thinking on (<|think|> in the system turn)")
    ap.add_argument("--response", help="a template of the answer: pinned text and [[n]] free tokens")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=("gpu", "q4", "bf16", "ref"), default="gpu")
    ap.add_argument("--steps", action="store_true", help="print each step's argmax canvas")
    ap.add_argument("--bench", action="store_true",
                    help="generate twice and time the second (the first builds the programs, reads the weights)")
    a = ap.parse_args()

    tok = Tokenizer(os.path.join(a.model, "tokenizer.json"))
    import json
    gen = json.load(open(os.path.join(a.model, "generation_config.json")))
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    s = S.Settings.from_config(gen, cfg)
    t0 = time.time()
    steps = None
    if a.engine == "gpu":
        from reclank.models.dgemma.gpu import DGemmaGPU
        m = DGemmaGPU(a.model)
        m.reset()
        encode, decode, steps = m.encode, None, m.steps
    elif a.engine == "ref":
        from reclank.models.dgemma.model import DGemmaRef
        m = DGemmaRef(a.model)
        cache = m.cache()
        encode = lambda ids: m.encode(ids, cache)            # noqa: E731
        decode = lambda c, sc: m.decode(c, cache, sc)        # noqa: E731
    else:
        from reclank.models.dgemma.runtime import DGemma
        m = DGemma(a.model, dtype="int4" if a.engine == "q4" else "bf16")
        cache = m.cache()
        pos = [0]

        def encode(ids):
            m.encode(ids, cache, pos[0])
            pos[0] += len(ids)

        decode = lambda c, sc: m.decode(c, cache, pos[0], sc)  # noqa: E731
    print("load %.0f s" % (time.time() - t0), file=sys.stderr, flush=True)

    prompt = chat_ids(tok, a.prompt, a.system, a.think)
    fixed = S.parse_template(a.response, tok.encode) if a.response else None
    on_step = None
    if a.steps:
        def on_step(step, am, keep, ent):
            print("[step %2d, %3d kept, entropy %.3f] %s" % (step, keep.sum(), ent.mean(),
                  tok.decode(am.tolist())[:160].replace("\n", " ")), file=sys.stderr, flush=True)
    for run in range(2 if a.bench else 1):
        if a.engine == "gpu":
            m.reset()
        else:
            cache = m.cache()
            if a.engine != "ref":
                pos[0] = 0
        t0 = time.time()
        out, st = S.generate(encode, decode, prompt, a.max_tokens, s, np.random.default_rng(a.seed),
                             on_step=on_step, fixed=fixed, steps=steps)
        dt = time.time() - t0
    print(tok.decode(out))
    print("\n%d tokens in %.1f s; %d decoder passes (%s), %.1f tokens a pass" % (
        st.tokens, dt, sum(st.steps), " ".join(map(str, st.steps)), st.per_pass), file=sys.stderr)


if __name__ == "__main__":
    main()
