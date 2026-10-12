#!/usr/bin/env python3
"""The decode step of DeepSeek V4.1 Flash on the GPU (reclank/models/deepseek41/gpu.py)
against the CPU model (model.py, near) from the same state (DEEPSEEK_V41_PLAN.md
Phase 3).

A prompt of a reference file runs on the CPU; its state goes to the GPU; then
each step runs on both with the same token (the CPU's best): each layer's
output (dB of the difference), the experts of each layer (the share that
agree), the logits (dB, KL, the top token), and the caches the step wrote.
With --resync (the default) the GPU takes the CPU's state again before each
step (each step from the same state); --free lets it keep its own.

    RECLANK_GPU_PART_THREADS=18 taskset -c 0-19,24-43 python scripts/check_dsv41_gpu_step.py \\
        /home/cpage/models/DeepSeek-V4.1-Flash /home/cpage/npg-run/dsv41-ref/v1/prompt0.npz --steps 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# DeepSeek V4.1 on the GPU runs with OMP_NUM_THREADS=1 only (gpu.check_omp_threads), set before reclank loads
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.deepseek41.gpu import HOT_GB, DSV41GPU  # noqa: E402
from reclank.models.deepseek41.model import DSV41CPU  # noqa: E402


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(ref, mine):
    p = logsm(ref)
    return float((np.exp(p) * (p - logsm(mine))).sum(-1).mean())


def db(a, ref):
    a, ref = a.astype(np.float64), ref.astype(np.float64)
    return 10 * np.log10(max(((a - ref) ** 2).sum(), 1e-300) / max((ref ** 2).sum(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("ref")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096,
                    help="the most positions (load_state copies the CPU model's float32 caches of all of them)")
    ap.add_argument("--free", action="store_true", help="the GPU keeps its own state after the first step")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--gpu-prefill", action="store_true",
                    help="the prompt on the GPU too (groups): its logits and caches against the CPU's; the steps "
                         "then from the GPU's own state")
    ap.add_argument("--group", type=int, default=None, help="the tokens of a group of a prompt (default the model's, RECLANK_DS_GROUP)")
    ap.add_argument("--cache", choices=("min", "bf16"), default="min", help="the form of the caches")
    ap.add_argument("--hot-gb", type=float, default=HOT_GB, help="the GPU memory of the hot experts (default 12)")
    ap.add_argument("--hot-check", action="store_true",
                    help="each step twice from the same state: all experts cold, then its experts hot (the same bits)")
    ap.add_argument("--profile", action="store_true", help="then the time of each kind of record, and 8 steps")
    a = ap.parse_args()
    t0 = time.time()
    m = DSV41CPU(a.ckpt, exact=False, max_len=a.max_len)
    assert m.w.z is not None, "a zygote must serve the experts"
    ids = np.load(a.ref)["ids"]
    T = len(ids)
    logits, players = m.forward(ids, 0, keep=a.gpu_prefill)
    print("the prompt (%d tokens) on the CPU in %.1f s" % (T, time.time() - t0))
    t1 = time.time()
    hot_gb = max(a.hot_gb, 6 * 40 * 18.81e6 / 1e9 + 0.01) if a.hot_check else a.hot_gb
    # the first hot experts: the most chosen of each layer in the prompt
    counts = np.stack([np.bincount(m.routes[:, i].reshape(-1), minlength=m.w.src.E) for i in range(m.L)])
    g = DSV41GPU(m, graph=not a.no_graph, hot_gb=hot_gb, counts=counts, cache=a.cache)
    bad = 0
    print("the GPU program (%d records, %d hot slots a layer) in %.1f s" % (len(g.run(1).prog.recs), g.n_slots,
                                                                           time.time() - t1))
    if a.gpu_prefill:
        t2 = time.time()
        gl = g.prefill(ids, 0, group=a.group)
        tp = time.time() - t2
        g.save_state()
        lg, k = db(gl, logits[-1]), kl(logits[-1][None], gl[None])
        print("the prompt on the GPU (groups of %d) in %.2f s: logits of the last token %.1f dB, KL %.2e, "
              "top GPU %d CPU %d" % (a.group or g.group, tp, lg, k, int(np.argmax(gl)), int(np.argmax(logits[-1]))))
        if g.last.layers is not None:
            gla = g.download(g.last.layers)            # [L, t, hc, dim]: the last group
            tl = gla.shape[1]
            per = [db(gla[i], players[T - tl:, i]) for i in range(m.L)]
            print("  the layers of the last group (%d tokens, dB): %s" % (tl, " ".join("%.0f" % v for v in per)))
            first = [db(gla[i, 0], players[T - tl, i]) for i in range(m.L)]
            print("  its first token (dB): %s" % " ".join("%.0f" % v for v in first))
            gids = g.download(g.last.ids).reshape(m.L, tl, m.topk)
            agree = np.mean([len(set(gids[i, j]) & set(m.routes[T - tl + j, i])) / m.topk
                             for i in range(m.L) for j in range(tl)])
            print("  experts agree %.3f" % agree)
        win = [db(g.cache_f32(g.window[i], "w"), m.window[i]) for i in range(m.L)]
        print("  window rings (dB): %s" % " ".join("%.0f" % v for v in win))
        for i in sorted(g.ckv):
            n = T // m.ratios[i]
            extra = ""
            if i in g.kstate and T % m.ratios[i]:
                extra = "; open group %.1f dB" % db(g.kstate[i][:T % m.ratios[i]], m.kstate[i][:T % m.ratios[i]])
            print("  layer %d: %d entries: ckv %.1f dB, index keys %.1f dB%s" % (
                i, n, db(g.cache_f32(g.ckv[i][:n], "c"), m.ckv[i][:n]), db(g.cache_f32(g.kcache[i][:n], "k"), m.kcache[i][:n]), extra))
        bad += int(np.argmax(gl) != np.argmax(logits[-1]))
        a.free = True
    else:
        g.load_state(m)
    tok = int(np.argmax(logits[-1]))
    for s in range(a.steps):
        pos = T + s
        if s > 0 and not a.free:
            g.load_state(m)
        t2 = time.time()
        gl = g.step(tok, pos)
        tg = time.time() - t2
        hits = g.hits()
        if a.hot_check:
            gids0 = g.download(g.ids).copy()
            for i in range(m.L):
                g.maps[i][:] = -1
                g.slot_expert[i][:] = -1
                g.mirror.buffer_of(g.maps[i]).upload(g.maps[i])
            g.load_state(m)
            cold_l = g.step(tok, pos)
            cold_layers = g.download(g.layers_out).copy()
            for i in range(m.L):
                for slot, e in enumerate(gids0[i]):
                    g.load_expert(i, slot, int(e))
            g.load_state(m)
            hot_l = g.step(tok, pos)
            hot_layers = g.download(g.layers_out).copy()
            nd = int((cold_l.view(np.uint32) != hot_l.view(np.uint32)).sum())
            ndl = int((cold_layers.view(np.uint32) != hot_layers.view(np.uint32)).sum())
            print("  hot check: all cold (hits %.2f) vs all hot (hits %.2f): logits %s, layers %s" % (
                0.0, g.hits(), "the same bits" if nd == 0 else "%d DIFFER" % nd,
                "the same bits" if ndl == 0 else "%d DIFFER" % ndl))
            bad += (nd != 0) + (ndl != 0)
            gl = hot_l
        t2 = time.time()
        cl, layers = m.forward([tok], pos, keep=True)
        tc = time.time() - t2
        cl, layers = cl[0], layers[0]
        gl_layers = g.download(g.layers_out)
        gids = g.download(g.ids)
        per = [db(gl_layers[i], layers[i]) for i in range(m.L)]
        agree = np.mean([len(set(gids[i]) & set(m.routes[0, i])) / m.topk for i in range(m.L)])
        same = [i for i in range(m.L) if set(gids[i]) != set(m.routes[0, i])]
        g.save_state()
        caches = []
        for i in sorted(g.ckv):
            r = m.ratios[i]
            if (pos + 1) % r == 0:
                e = pos // r
                caches.append("ckv%d %.1f kc%d %.1f" % (i, db(g.cache_f32(g.ckv[i][e:e + 1], "c"), m.ckv[i][e:e + 1]), i,
                                                       db(g.cache_f32(g.kcache[i][e:e + 1], "k"), m.kcache[i][e:e + 1])))
        win = np.mean([db(g.cache_f32(g.window[i][pos % m.win:pos % m.win + 1], "w"), m.window[i][pos % m.win:pos % m.win + 1]) for i in range(m.L)])
        lg = db(gl, cl)
        k = kl(cl[None], gl[None])
        print("step %d (pos %d, token %d): GPU %.1f ms (hot %.2f), CPU %.1f s" % (s, pos, tok, tg * 1e3, hits, tc))
        print("  layers (dB): %s" % " ".join("%.0f" % v for v in per))
        print("  experts agree %.3f (layers differing: %s); window rows %.1f dB; %s" % (
            agree, same[:10], win, "; ".join(caches) or "no compressed entry this step"))
        print("  logits %.1f dB, KL %.2e, top GPU %d CPU %d" % (lg, k, int(np.argmax(gl)), int(np.argmax(cl))))
        bad += int(np.argmax(gl) != np.argmax(cl)) + (lg > -30)
        tok = int(np.argmax(cl))
    if a.profile:
        pos = T + a.steps
        g.load_state(m)
        rows = g.profile(tok, pos)
        print("profile of a step (pos %d), ms: %.1f in all" % (pos, sum(s for _n, (_c, s) in rows)))
        for name, (n, s) in rows:
            print("  %-14s %4d records %8.2f ms" % (name, n, s))
        t2 = time.time()
        for j in range(8):
            g.step(tok, pos + 1 + j)
        print("8 steps (the graph): %.1f ms each" % ((time.time() - t2) / 8 * 1e3))
    g.close()
    print("OK" if not bad else "%d checks off" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
