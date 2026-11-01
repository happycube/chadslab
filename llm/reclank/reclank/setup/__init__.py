"""First use: which models this machine runs, and how to start one.

    python -m reclank.setup                       the report, then a guide (asks which model)
    python -m reclank.setup --all                 the report with the models not yet supported
    python -m reclank.setup --model gemma4-26b    the guide of one model: its download, its
                                                  server, a first request
    python -m reclank.setup --model ID --download download it (asks first; --yes does not ask)
    python -m reclank.setup --json                the report as JSON (no questions)
    python -m reclank.setup --assume-ram 16 --assume-vram 0     a smaller machine
    python -m reclank.setup --check-hub           the table's sizes and licenses against the Hub

hardware.py finds the machine, catalog.py lists the models and their needs;
fit() decides each model's mode. Nothing downloads without a yes, and the
setup never starts a server: it prints the command.
"""
from __future__ import annotations

import os
import sys

from reclank import settings
from reclank.setup.catalog import KINDS, MODELS

# the share of the RAM and of a card a model may take (the system, the desktop)
RAM_SHARE = 0.92
VRAM_SHARE = 0.97
PORTS = {"llm": 8080, "embed": 8082, "generate": 8084}
MODULES = {"llm": "server", "embed": "server.embed", "generate": "server.generate"}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def models_dir():
    """The folder of the downloads: RECLANK_MODELS, else ~/models."""
    return os.path.expanduser(settings.get("MODELS", "~/models"))


def _cc_name(cc):
    return "sm_%d" % round(cc * 10) if cc else "unknown"


def _toolkit_for(m, cc):
    """The toolkit that builds cc (backends/cuda/build.py MIN_ARCH), or None."""
    try:
        from reclank.backends.cuda.build import MIN_ARCH
    except Exception:
        MIN_ARCH = {11: 35, 12: 50, 13: 75}
    for path, v in m.toolkits:
        major = int(v.split(".")[0])
        if cc is None or round(cc * 10) >= MIN_ARCH.get(major, 0):
            return path, v
    return None


def fit(model, m):
    """How model runs on machine m: a dict of modes ("gpu", "cpu": their
    notes), why (the reasons of the modes it cannot take) and warnings."""
    out = {"modes": {}, "why": [], "warn": []}
    if model.status == "not-yet":
        out["why"].append("not yet supported: " + model.why)
        return out
    if model.status == "manual":
        out["warn"].append(model.why)
    if m.os != "Linux":
        out["warn"].append("reclank runs on Linux")
    if not m.cc_compiler:
        out["warn"].append("needs a C compiler (gcc) for the kernels")
    ram = m.ram_gb * RAM_SHARE
    g = m.gpu()
    need = model.gpu
    if need:
        if g is None:
            out["why"].append("GPU: no NVIDIA GPU")
        elif g.cc is not None and g.cc < model.min_cc:
            out["why"].append("GPU: %s; needs %s or newer" % (_cc_name(g.cc), _cc_name(model.min_cc)))
        elif g.vram_gb * VRAM_SHARE < need.vram:
            out["why"].append("GPU: needs %s GB of GPU memory, the card has %.0f" % (_gb(need.vram, need.est),
                                                                                        g.vram_gb))
        elif ram < need.ram:
            out["why"].append("GPU: needs %s GB of RAM beside the card, the machine has %.0f (%.0f after "
                              "the system)" % (_gb(need.ram, need.est), m.ram_gb, ram))
        else:
            notes = []
            if g.free_gb < need.vram and "vram" not in m.assumed:
                notes.append("%.1f GB of the card is free now" % g.free_gb)
            tk = _toolkit_for(m, g.cc)
            if tk is None:
                notes.append("install the CUDA toolkit (nvcc; CUDA 12 for Pascal, 12 or 13 for sm_75 and up)")
            if need.note:
                notes.append(need.note)
            out["modes"]["gpu"] = notes
    need = model.cpu
    if need:
        if ram < need.ram:
            out["why"].append("CPU: needs %s GB of RAM, the machine has %.0f (%.0f after the system)" % (
                _gb(need.ram, need.est), m.ram_gb, ram))
        else:
            notes = []
            if m.tier == "scalar":
                notes.append("slow: no AVX2 (the scalar kernels)")
            if m.ram_free_gb < need.ram and "ram" not in m.assumed:
                notes.append("%.0f GB of RAM is free now" % m.ram_free_gb)
            out["modes"]["cpu"] = notes
    elif not model.gpu:
        out["why"].append("no mode recorded")
    elif "gpu" not in out["modes"]:
        out["why"].append("CPU: no CPU mode")
    if m.disk_free_gb is not None and m.disk_free_gb < model.gb:
        out["warn"].append("the download is %.1f GB; %s has %.1f GB free" % (model.gb, m.models_dir,
                                                                             m.disk_free_gb))
    return out


def _gb(x, est):
    return ("~%g" if est else "%g") % x


def needs_text(model):
    """The needs of a model in a few words: "GPU 8 + RAM 16 | CPU RAM 17"
    (~ an estimate)."""
    parts = []
    if model.gpu:
        parts.append("GPU %s + RAM %s" % (_gb(model.gpu.vram, model.gpu.est), _gb(model.gpu.ram, model.gpu.est)))
    if model.cpu:
        parts.append("CPU RAM %s" % _gb(model.cpu.ram, model.cpu.est))
    return " | ".join(parts) or "-"


def rows(m):
    """[(model, fit)] in the order of the report: by kind, then the ones that
    run (GPU first), then the ones that do not, then the ones not yet."""
    def key(item):
        model, f = item
        state = 0 if "gpu" in f["modes"] else 1 if "cpu" in f["modes"] else 3 if model.status == "not-yet" else 2
        return (KINDS.index(model.kind), state, model.gb)
    return sorted(((model, fit(model, m)) for model in MODELS), key=key)


def suggest(m):
    """The first model to try: the first of the first-model order that runs."""
    ok = [(model, f) for model, f in rows(m) if model.first and f["modes"]]
    ok.sort(key=lambda mf: mf[0].first)
    return ok[0][0] if ok else None


def machine_text(m):
    """The machine in a few lines."""
    L = []
    a = " (assumed: %s)" % ", ".join("%s %s" % kv for kv in m.assumed.items()) if m.assumed else ""
    L.append("Machine%s" % a)
    cores = "%d cores, " % m.cores if m.cores else ""
    L.append("  CPU     %s; %s%d threads (%d usable); C tier %s" % (m.cpu, cores, m.threads, m.usable, m.tier))
    if len(m.nodes) > 1:
        L.append("  NUMA    %d nodes: %s" % (len(m.nodes), ", ".join("%d: CPUs %s, %.0f GB" % n for n in m.nodes)))
    L.append("  RAM     %.0f GB (%.0f GB free)" % (m.ram_gb, m.ram_free_gb))
    if m.gpus:
        for g in m.gpus:
            L.append("  GPU %d   %s, %.1f GB (%.1f free), %s" % (g.index, g.name, g.vram_gb, g.free_gb, _cc_name(g.cc)))
        tks = ", ".join(v + (" (%s)" % p if i == 0 else "") for i, (p, v) in enumerate(m.toolkits))
        L.append("  CUDA    driver %s (CUDA %s); toolkits %s" % (
            m.driver or "?", m.cuda_driver or "?", tks or "none (no nvcc: the GPU kernels cannot build)"))
    else:
        L.append("  GPU     none (the models run on the CPU)")
    L.append("  Tools   Python %s, NumPy %s, C compiler %s, huggingface_hub %s" % (
        m.python, m.numpy or "MISSING (pip install numpy)", m.cc_compiler or "MISSING",
        "yes" if m.hub else "no (pip install huggingface_hub, for the downloads)"))
    disk = "%.0f GB free" % m.disk_free_gb if m.disk_free_gb is not None else "unknown"
    L.append("  Models  %s (%s; RECLANK_MODELS sets it)" % (m.models_dir, disk))
    for n in m.notes:
        L.append("  note    " + n)
    return "\n".join(L)


def report_text(m, all_rows=False):
    """The report: the machine and a line a model. The models not yet
    supported and the ones with a manual step show with all_rows only."""
    L = [machine_text(m), ""]
    shown = [(model, f) for model, f in rows(m) if all_rows or model.status == "ready"]
    hidden = [model for model, _ in rows(m) if not (all_rows or model.status == "ready")]
    L.append("%-6s %-19s %-41s %8s  %-9s  %-30s %s" % ("kind", "id", "model", "download", "runs here",
                                                      "needs GB (~ estimate)", "license"))
    last = None
    for model, f in shown:
        if last is not None and model.kind != last:
            L.append("")
        last = model.kind
        runs = ", ".join(k.upper() for k in ("gpu", "cpu") if k in f["modes"]) or (
            "not yet" if model.status == "not-yet" else "no")
        if model.status == "manual" and f["modes"]:
            runs += " *"
        lic = model.license + ("" if model.commercial else " (non-commercial)")
        short = m.disk_free_gb is not None and m.disk_free_gb < model.gb
        L.append("%-6s %-19s %-41s %5.1f GB%s %-9s  %-30s %s" % (model.kind, model.id, model.name[:41], model.gb,
                                                              "!" if short else " ", runs, needs_text(model), lic))
        if not f["modes"]:
            L.append("%-27s -> %s" % ("", "; ".join(f["why"])))
    L.append("")
    if any(model.status == "manual" for model, _ in shown):
        L.append("* a step besides the download (python -m reclank.setup --model ID tells it)")
    if hidden:
        n_yet = sum(x.status == "not-yet" for x in hidden)
        n_man = len(hidden) - n_yet
        L.append("%d more not yet supported%s: --all" % (
            n_yet, ", %d with a manual step" % n_man if n_man else ""))
    if m.disk_free_gb is not None and any(m.disk_free_gb < x.gb for x, _ in shown):
        L.append("! more than the %.0f GB free in %s (--models-dir or RECLANK_MODELS: another disk)" % (
            m.disk_free_gb, m.models_dir))
    s = suggest(m)
    if s:
        L.append("A first model: %s (%s): python -m reclank.setup --model %s" % (s.name, s.id, s.id))
    return "\n".join(L)


def report_json(m):
    """The report as a dict (for --json)."""
    import dataclasses
    mach = dataclasses.asdict(m)
    out = []
    for model, f in rows(m):
        d = dataclasses.asdict(model)
        d["fit"] = f
        out.append(d)
    s = suggest(m)
    return {"machine": mach, "models": out, "suggest": s.id if s else None}


def _q(s):
    """s quoted for a shell, when it needs it."""
    import shlex
    return shlex.quote(s)


def download_cmd(model, root, hf="hf"):
    """The command line of the download: hf download, else (no hf command)
    huggingface_hub's snapshot_download."""
    d = os.path.join(root, model.dir)
    if hf is None:
        py = ("from huggingface_hub import snapshot_download; snapshot_download(%r, revision=%r, "
              "local_dir=%r, allow_patterns=%r)" % (model.repo, model.revision, d, model.files))
        return "%s -c %s" % (_q(sys.executable or "python3"), _q(py))
    files = list(model.files or [])
    if any(set(f) & set("*?[") for f in files):
        # (the names after the repo are names; patterns go in --include)
        cmd = [hf, "download", model.repo] + [x for f in files for x in ("--include", f)]
    else:
        cmd = [hf, "download", model.repo] + files
    if model.revision:
        cmd += ["--revision", model.revision]
    cmd += ["--local-dir", d]
    return " ".join(_q(c) for c in cmd)


def serve_cmd(model, root, mode):
    """The command line of the server of the model in mode (gpu or cpu)."""
    d = os.path.join(root, model.dir)
    args = model.args + " " + (model.gpu_args if mode == "gpu" else model.cpu_args)
    args = " ".join(_q(a.replace("{d}", d)) for a in args.split())
    env = "OMP_NUM_THREADS=1 " if mode == "gpu" else ""
    py = sys.executable or "python3"
    return "cd %s && %sPYTHONPATH=. %s -m %s %s --port %d" % (_q(ROOT), env, _q(py), MODULES[model.server],
                                                             args, PORTS[model.server])


def request_cmd(model):
    """A first request with curl."""
    port = PORTS[model.server]
    head = ("curl -s http://127.0.0.1:%d/v1/%s -H 'Authorization: Bearer change-me' "
            "-H 'Content-Type: application/json' -d '%s'")
    k = model.kind
    if k == "chat":
        return head % (port, "chat/completions", '{"messages": [{"role": "user", "content": "Hello!"}]}')
    if k == "embed":
        body = '{"input": "A first sentence to embed."}' if not model.model_id else \
            '{"model": "%s", "input": "calm piano with soft rain"}' % model.model_id
        return head % (port, "embeddings", body)
    if k == "image":
        size = {"sd-1.5": "512x512", "sd-2.0-base": "512x512", "sd-2.0-768": "768x768"}.get(model.id, "1024x1024")
        body = ('{"model": "%s", "prompt": "a red fox in fresh snow, morning light", "size": "%s", '
                '"response_format": "b64_json"}' % (model.model_id, size))
        return (head % (port, "images/generations", body) + " | python3 -c \"import sys, json, base64; "
                "open('first.png', 'wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))\"")
    if k == "speech":
        body = '{"model": "%s", "input": "Hello from reclank.", "voice": "default"}' % model.model_id
        return head % (port, "audio/speech", body) + " -o first.mp3"
    if k == "music":
        body = '{"prompt": "calm piano with soft rain", "duration": 30}'
        return head % (port, "music/generations", body) + " -o first.wav"
    if k == "video":
        body = '{"model": "%s", "prompt": "waves on a rocky shore at sunset", "size": "832x480", "seconds": 4}' % (
            model.model_id)
        return (head % (port, "videos", body) + "\n    (then GET /v1/videos/ID until it is completed, and "
                "GET /v1/videos/ID/content -o first.mp4)")
    if k == "upscale":
        return ("curl -s http://127.0.0.1:%d/v1/images/upscale -H 'Authorization: Bearer change-me' "
                "-F image=@photo.jpg -F model=%s | python3 -c \"import sys, json, base64; "
                "open('upscaled.png', 'wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))\""
                % (port, model.model_id))
    if k == "stereo" and model.model_id == "vda-small":
        return ("curl -s http://127.0.0.1:%d/v1/videos/stereo -H 'Authorization: Bearer change-me' "
                "-F video=@clip.mp4 -F layout=sbs\n    (then GET /v1/videos/ID until it is completed, and "
                "GET /v1/videos/ID/content -o clip_LR.mp4)" % port)
    if k == "stereo":
        return ("curl -s http://127.0.0.1:%d/v1/images/stereo -H 'Authorization: Bearer change-me' "
                "-F image=@photo.jpg -F model=%s -F layout=sbs | python3 -c \"import sys, json, base64; "
                "open('photo_LR.jpg', 'wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))\""
                % (port, model.model_id))
    if k == "interpolate":
        return ("curl -s http://127.0.0.1:%d/v1/videos/interpolate -H 'Authorization: Bearer change-me' "
                "-F video=@clip.mp4 -F factor=2\n    (then GET /v1/videos/ID until it is completed, and "
                "GET /v1/videos/ID/content -o clip_x2.mp4)" % port)
    if k == "asr":
        return ("curl -s http://127.0.0.1:%d/v1/audio/transcriptions -H 'Authorization: Bearer change-me' "
                "-F file=@speech.mp3 -F model=%s -F response_format=srt" % (port, model.model_id))
    return None


def guide_text(model, m):
    """The steps of one model on machine m: the download, the server, a request."""
    root = models_dir()
    f = fit(model, m)
    L = ["== %s (%s)" % (model.name, model.id),
         "   %s; %.1f GB; license %s%s" % (model.repo + ("@" + model.revision if model.revision else ""),
                                          model.gb, model.license,
                                          "" if model.commercial else " (research / non-commercial use)")]
    if model.status == "not-yet":
        L.append("   Not yet supported: %s" % model.why)
        L += ["   note: " + n for n in model.notes]
        return "\n".join(L)
    modes = list(f["modes"])
    if modes:
        L.append("   Runs here on: %s" % " or ".join(x.upper() for x in modes))
    else:
        L.append("   Does not run here: %s" % "; ".join(f["why"]))
    for k in modes:
        for n in f["modes"][k]:
            L.append("   %s: %s" % (k.upper(), n))
    for w in f["warn"]:
        L.append("   warning: %s" % w)
    if model.gpu:
        n = model.gpu
        L.append("   GPU needs GPU %s GB + RAM %s GB (%s)" % (_gb(n.vram, n.est), _gb(n.ram, n.est), n.src))
    if model.cpu:
        n = model.cpu
        L.append("   CPU needs RAM %s GB (%s)" % (_gb(n.ram, n.est), n.src))
    L.append("")
    if model.status == "manual":
        L.append("1. Prepare the weights (not a plain download) into %s:" % os.path.join(root, model.dir))
        L.append("   " + model.why + "; the source of %s: %s" % (model.name, model.repo))
    else:
        L.append("1. Download (%.1f GB into %s):" % (model.gb, os.path.join(root, model.dir)))
        L.append("   " + download_cmd(model, root, m.hf_cli))
    show = modes or [k for k in ("gpu", "cpu") if getattr(model, k)]
    L.append("")
    L.append("2. Start the server%s:" % ("" if modes else " (on a machine that runs it)"))
    for k in show:
        L.append("   %s:  %s" % (k.upper(), serve_cmd(model, root, k)))
    L.append("   The first start builds the kernels (the C build about a minute, CUDA some minutes).")
    r = request_cmd(model)
    if r:
        L.append("")
        L.append("3. A first request (in another terminal, once the server is ready):")
        L.append("   " + r)
    for n in model.notes:
        L.append("   note: " + n.replace("{d}", os.path.join(root, model.dir)))
    return "\n".join(L)


def download(model, root):
    """Download the model's files into its folder (huggingface_hub)."""
    from huggingface_hub import snapshot_download
    d = os.path.join(root, model.dir)
    os.makedirs(d, exist_ok=True)
    path = snapshot_download(model.repo, revision=model.revision, local_dir=d, allow_patterns=model.files)
    # drop the files' pages: a download that fills a NUMA node's page cache
    # stalls the programs of that node
    for top, _, names in os.walk(d):
        for n in names:
            try:
                fd = os.open(os.path.join(top, n), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass
            finally:
                os.close(fd)
    return path
