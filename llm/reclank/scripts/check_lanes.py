#!/usr/bin/env python3
"""Two lanes (reclank/models/qwen/lanes.py, docs/LANES_PLAN.md) of Qwen3.8-Flash-Next on two chat
prompts, greedy, N tokens each (env N, 128): each prompt alone (the model's steps), then both as a pair;
the tokens of each lane against its run alone, and the tokens a second of one lane and of the pair.
With RECLANK_GPU_HOT_DYN=0 (a fixed hot set) each lane must give the tokens of its run alone; with the
hot cache the tokens can part at a near tie (an expert on the GPU or the CPU). MTP=1: MTP lanes (3
drafts; each prompt alone with generate_mtp_gpu, the pair with greedy_mtp_pair).

    RECLANK_GPU_HOT_DYN=0 OMP_NUM_THREADS=1 python scripts/check_lanes.py MODEL_DIR
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reclank.models.qwen.qwen38 import Qwen4Cache, Qwen4CPU
from reclank.models.qwen.gpu38 import Qwen4GPU, generate_mtp_gpu
from reclank.models.qwen.lanes import Lane, greedy_mtp_pair, greedy_pair
from reclank.models.qwen.tokenizer import QwenTokenizer
P = sys.argv[1]
N = int(os.environ.get("N", "128"))
MTP = os.environ.get("MTP", "0") == "1"
PAD = 8 + (4 if MTP else 0)     # (a round writes up to 4 rows)
tok = QwenTokenizer(P + "/tokenizer.json")
def chat(q):
    return tok.encode("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
A = chat("Write a detailed explanation of how a CPU cache hierarchy works, with L1, L2 and L3, and why it matters for performance.")
B = chat("Write a Python function that parses an ISO 8601 date string without using datetime, with tests.")
m = Qwen4CPU(P); dev = Qwen4GPU(m, ctx=4096)
def alone(ids):
    dev.attach(Qwen4Cache(m.cfg, len(ids) + N + PAD))
    if MTP:
        st = {}
        out = generate_mtp_gpu(dev, ids, N, draft=3, stats=st)
        return out, N / st["decode_s"]
    dev.prefill(ids); y = int(dev.argmax(1)[0]); out = [y]; pos = len(ids); t0 = time.time()
    while len(out) < N:
        dev.step(y, pos); y = int(dev.argmax(1)[0]); out.append(y); pos += 1
    return out, (N - 1) / (time.time() - t0)
for rep in range(2):
    oa, ra = alone(A)
    ob, rb = alone(B)
print("alone: A %.1f tok/s, B %.1f tok/s" % (ra, rb), flush=True)
lane = Lane(dev)
for rep in range(2):
    dev.attach(Qwen4Cache(m.cfg, len(A) + N + PAD))
    lane.attach(Qwen4Cache(m.cfg, len(B) + N + PAD))
    t0 = time.time()
    st = {}
    pa, pb = greedy_mtp_pair(dev, lane, A, B, N, stats=st) if MTP else greedy_pair(dev, lane, A, B, N)
    dt = time.time() - t0
    print("pair run %d: %.1f s with the prompts; A same tokens as alone: %d of %d, B: %d of %d" % (
        rep, dt, next((i for i, (x, y) in enumerate(zip(pa, oa)) if x != y), N), N,
        next((i for i, (x, y) in enumerate(zip(pb, ob)) if x != y), N), N), flush=True)
    if MTP:
        print("  %d pair rounds (%.2f + %.2f tokens a round): drafts %.2f s, verify groups %.2f s: "
              "%.1f tok/s for both (alone %.1f): %.2fx" % (
                  st["rounds"], 1 + st["kept_a"] / st["rounds"], 1 + st["kept_b"] / st["rounds"],
                  st["draft_s"], st["verify_s"],
                  (2 * st["rounds"] + st["kept_a"] + st["kept_b"]) / (st["draft_s"] + st["verify_s"]),
                  (ra + rb) / 2, (2 * st["rounds"] + st["kept_a"] + st["kept_b"])
                  / (st["draft_s"] + st["verify_s"]) / ((ra + rb) / 2)), flush=True)
if MTP:
    lane.close()
    dev.close()
    sys.exit(0)
# the rate of the pair's steps alone (no prompts)
import reclank.models.qwen.lanes as LN
dev.attach(Qwen4Cache(m.cfg, len(A) + N + PAD)); lane.attach(Qwen4Cache(m.cfg, len(B) + N + PAD))
dev.prefill(A); ya = int(dev.argmax(1)[0]); yb = lane.prefill(B); pa_, pb_ = len(A), len(B)
t0 = time.time()
for s in range(N - 1):
    LN.step_pair(dev, lane, ya, pa_, yb, pb_)
    ya = int(dev.argmax(1)[0]); yb = int(lane.argmax()[0]); pa_ += 1; pb_ += 1
dt = time.time() - t0
print("pair: %.1f steps/s, %.1f tok/s for both (alone %.1f): %.2fx" % ((N - 1) / dt, 2 * (N - 1) / dt, (ra + rb) / 2, 2 * (N - 1) / dt / ((ra + rb) / 2)), flush=True)
lane.close()
dev.close()
