#!/usr/bin/env python3
"""Teacher-forced accuracy of the MTP layer: on the chat transcripts (the
greedy answers of the RQ8_0 model), the main model reads each text once
(the streams), then the MTP layer runs on those streams in rows of 16 (as
the prompt of the server): row j gets the stream at j - 1 and token j and
drafts token j + 1. The share of answer positions whose draft is the next
token of the text. Same inputs for every expert file, so it compares MTP
weights with none of the noise of a free decode (whose text, and so its
acceptance, changes from run to run).

On the 2-socket Xeon (NVFP4 experts): MTP experts Q8_0 of the FP8 of NVIDIA
(-31.3 dB) 83.86%, BF12 of the bfloat16 of the original 83.80%; the MIX
file (RQ8_0 MTP experts, its own routed experts) 84.37%.

    GF=model.gguf [EXPERTS=a.gguf,b.gguf] python scripts/check_mtp_accuracy.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import np_gemma  # noqa
import numpy as np
from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
from np_gemma.qwen4_gpu import Qwen4GPU
GF = os.environ["GF"]
m = Qwen4CPU(GF, experts=os.environ.get("EXPERTS") or None)
g = Qwen4GPU(m, ctx=16384)
convs = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests/texts/qwen38_chat.json")))
convs = convs["convs"] if isinstance(convs, dict) else convs
hit = tot = 0
per = []
for c in convs:
    ids = list(c["ids"])
    n = len(ids)
    g.attach(Qwen4Cache(m.cfg, -(-n // 4096) * 4096 + 64))
    H = np.concatenate([g.prefill(ids[c0:c0 + 4096], pos=c0, streams=True) for c0 in range(0, n, 4096)])
    Hp = np.concatenate([np.zeros((1, H.shape[1]), np.float32), H[:-1]])
    draft = np.full(n, -1, np.int64)
    for c0 in range(0, n, 16):
        r = min(16, n - c0)
        g.mtp(Hp[c0:c0 + r], ids[c0:c0 + r], c0)
        draft[c0:c0 + r] = g.argmax(r)
    want = np.zeros(n, bool)
    for a, b in c["spans"]:
        want[a:b - 1] = True              # draft at j predicts j + 1, inside the answer
    ok = (draft[:-1] == np.asarray(ids[1:])) & want[:-1]
    h, t = int(ok.sum()), int(want[:-1].sum())
    hit += h; tot += t
    per.append("%s %.1f%%" % (c.get("topic", "?")[:6], 100.0 * h / max(t, 1)))
print("RESULT MTP depth-1 accuracy on the answers: %.2f%% of %d positions; %s" % (100.0 * hit / tot, tot, ", ".join(per)))
