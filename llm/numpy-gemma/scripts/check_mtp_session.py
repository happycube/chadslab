#!/usr/bin/env python3
"""Check the MTP decode of Session with a sampler that has a seed.

Session.generate_stream uses the drafter when the session has one. The
sampler picks each token at each verify row, so the tokens must be the same
as the tokens of the plain session with the same seed. The script runs two
turns of a chat, so the second turn also checks the reuse of the cache after
an MTP turn.

    OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=ACTIVE PYTHONPATH=. \\
        python scripts/check_mtp_session.py
    NP_GEMMA_ATTN=0 ...    (the float attention of scripts/serve.py)
"""
from __future__ import annotations

import argparse
import glob
import os
import time

from np_gemma import Model
from np_gemma.assistant import Assistant
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.model import Session
from np_gemma.sampling import Sampler
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
HUB = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub"
REPO = "models--google--gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
TURNS = ["Write a haiku about the sea, then explain the image in one sentence.",
         "Now write one more haiku about the same image."]


def run(model, tok, drafter, max_new, seed):
    session = Session(model, max_len=2048, drafter=drafter, n_draft=2)
    msgs = []
    outs = []
    t = 0.0
    for text in TURNS:
        msgs.append({"role": "user", "content": text})
        ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                 thinking=False))
        sampler = Sampler(temperature=1.0, top_k=64, top_p=0.95, seed=seed)
        t0 = time.perf_counter()
        out = list(session.generate_stream(ids, max_new_tokens=max_new,
                                           eos_ids=set(tok.stop_ids), sampler=sampler))
        t += time.perf_counter() - t0
        outs.append(out)
        reply = out[:-1] if out and out[-1] in tok.stop_ids else out
        msgs.append({"role": "assistant", "content": tok.decode(reply)})
    return outs, t, session.mtp_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    path = sorted(glob.glob(os.path.join(HUB, REPO, "snapshots", "*")))[-1]
    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    drafter = Assistant(path, dtype="int4")
    print("NP_GEMMA_ATTN=%s" % os.environ.get("NP_GEMMA_ATTN", "1"))
    ok = True
    for seed in args.seeds:
        ref, t_plain, _ = run(model, tok, None, args.max_new_tokens, seed)
        out, t_mtp, st = run(model, tok, drafter, args.max_new_tokens, seed)
        same = out == ref
        ok = ok and same
        n = sum(len(o) for o in ref)
        print("seed %d  tokens %d  plain %.2f tok/s  mtp %.2f tok/s  same %s  "
              "(last turn: drafts %d accepted %d)" % (
                  seed, n, n / t_plain, sum(len(o) for o in out) / t_mtp,
                  "yes" if same else "NO", st.get("drafts", 0), st.get("accepted", 0)))
    print("PASS" if ok else "FAIL")
    g.close()


if __name__ == "__main__":
    main()
