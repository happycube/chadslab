"""Nemotron 3.5 Bolt (P6 of docs/NEMOTRON_BOLT_PLAN.md) and Super: the routing profile of real text,
the count of each expert of each layer over prompts of three kinds (code, prose, an agent's turn with
tools and tool results), with the CPU engine (--gpu: NemotronGPU's prompt groups, Super: the
choices of its routers on the GPU). Writes MODEL/expert_profile.npz (NemotronGPU takes its first hot
experts from it) and prints the share of the expert reads that a hot set of each size covers.

    python scripts/nemotron_profile.py MODEL_DIR [tokens of each kind, 8000] [--gpu]
"""
import sys, os, glob, json, time, numpy as np
GPU = "--gpu" in sys.argv
if GPU:
    sys.argv.remove("--gpu")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reclank.models.nemotron.kq import NemotronKQ
from reclank.models.nemotron.engine import NemotronTok
D = sys.argv[1]
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
tok = NemotronTok(D + "/tokenizer.json", D + "/chat_template.jinja")
m = NemotronKQ(D); m.count_experts = True
cfg = m.cfg
counts = {i: np.zeros(E, np.int64) for i, (E, _, _) in cfg.moe.items()}
by_kind = {}
per = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
g = None
if GPU:
    from reclank.models.nemotron.gpu import GROUPS, NemotronGPU, task_threads
    os.environ.setdefault("RECLANK_GPU_CPU_THREADS", str(task_threads()))
    g = NemotronGPU(m, ctx=per + 1024)


def run(ids, kind):
    if g is not None:
        g.reset()
        c0 = 0
        while c0 < len(ids):
            size = next((s for s in GROUPS[::-1] if s <= len(ids) - c0), 1)
            chunk = ids[c0:c0 + size]
            g._run(chunk, c0)
            prog, gp = g._prog(size)
            for i in counts:
                gp.download("idx.%d" % i)
                a = prog.names["idx.%d" % i][:len(chunk)].reshape(-1)
                counts[i] += np.bincount(a[a >= 0], minlength=counts[i].size)
            c0 += size
        by_kind[kind] = by_kind.get(kind, 0) + len(ids)
        return
    c = m.cache(len(ids) + 8)
    for c0 in range(0, len(ids), m.CHUNK):
        chunk = ids[c0:c0 + m.CHUNK]
        prog = m.program(len(chunk))
        m._run(prog, chunk, c, c0); c.n += len(chunk)
        for i in counts:
            a = prog.names["idx.%d" % i].reshape(-1)
            counts[i] += np.bincount(a[a >= 0], minlength=counts[i].size)
    by_kind[kind] = by_kind.get(kind, 0) + len(ids)

t0 = time.time()
# code: reclank's own sources
code = "".join(open(f).read() for f in sorted(glob.glob(R + "reclank/models/qwen/*.py"))[:6])
run(tok.encode(code)[:per], "code")
# prose: the plans
docs = "".join(open(f).read() for f in sorted(glob.glob(R + "docs/*.md"))[:8])
run(tok.encode(docs)[:per], "docs")
# an agent's turn: a system prompt with tools, a request, tool results (files), in the template
tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object",
          "properties": {"path": {"type": "string"}, "command": {"type": "string"}}}}}
         for n, d in (("read_file", "Read a file"), ("bash", "Run a shell command"), ("edit", "Edit a file"))]
msgs = [{"role": "system", "content": "You are a coding agent in the reclank repository."},
        {"role": "user", "content": "Find why the lanes scheduler starves a decode while the other lane reads a prompt, and fix it."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "server/worker.py"}}}]},
        {"role": "tool", "content": open(R + "server/worker.py").read()[:30000]},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "server/backend.py"}}}]},
        {"role": "tool", "content": open(R + "server/backend.py").read()[:20000]}]
run(tok.encode(tok.apply_chat_template(msgs, thinking=True, tools=tools))[:per], "agent")
print("tokens", by_kind, "in %.0f s" % (time.time() - t0), flush=True)
np.savez(os.path.join(D, "expert_profile.npz"), **{"l%d" % i: c for i, c in counts.items()}, kinds=json.dumps(by_kind))
# coverage of a hot set of a share of each layer's experts, by its counts
for share in (0.05, 0.10, 0.15, 0.25, 0.33, 0.50, 0.75):
    cov = []
    for i, c in counts.items():
        n = int(share * c.size)
        s = np.sort(c)[::-1]
        cov.append(s[:n].sum() / max(s.sum(), 1))
    print("hot share %3.0f%%: covers %.1f%% of the expert reads (min layer %.1f%%)" % (100 * share, 100 * np.mean(cov), 100 * min(cov)))
