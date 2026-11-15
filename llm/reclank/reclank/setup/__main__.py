"""python -m reclank.setup: the first-use report and guide (see __init__.py)."""
from __future__ import annotations

import argparse
import json
import os
import sys

from reclank import setup as S
from reclank.setup import hardware
from reclank.setup.catalog import MODELS, by_id


def parser():
    ap = argparse.ArgumentParser(prog="python -m reclank.setup",
                                 description="Which models this machine runs, and how to start one.")
    ap.add_argument("--model", "-m", default=None, metavar="ID", help="the guide of one model (an id of the report)")
    ap.add_argument("--download", action="store_true", help="download --model (asks first, unless --yes)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="no questions: the report and the guide only, and --download without asking")
    ap.add_argument("--all", action="store_true",
                    help="also the models not yet supported and the ones with a manual step")
    ap.add_argument("--json", action="store_true", help="the report (or --model's fit) as JSON; no questions")
    ap.add_argument("--models-dir", default=None, help="the folder of the downloads (default RECLANK_MODELS, "
                                                       "else ~/models)")
    ap.add_argument("--check-hub", action="store_true",
                    help="compare the table's sizes and licenses with the Hub (reads the Hub; downloads nothing)")
    g = ap.add_argument_group("a machine other than this one")
    g.add_argument("--assume-ram", type=float, default=None, metavar="GB")
    g.add_argument("--assume-vram", type=float, default=None, metavar="GB", help="0: no GPU")
    g.add_argument("--assume-cc", type=float, default=None, metavar="X.Y", help="the GPU's compute capability (6.1)")
    g.add_argument("--assume-cpu", default=None, choices=("scalar", "avx2", "avx2_vnni", "avx512", "avx512_vnni"),
                   help="the C tier")
    g.add_argument("--assume-cores", type=int, default=None)
    g.add_argument("--assume-disk", type=float, default=None, metavar="GB", help="the free disk of the models folder")
    return ap


def _ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def check_hub():
    """Each model's download size and license on the Hub, beside the table's."""
    from fnmatch import fnmatch

    from huggingface_hub import HfApi
    api = HfApi()
    bad = 0
    for model in MODELS:
        try:
            info = api.model_info(model.repo, revision=model.revision, files_metadata=True)
        except Exception as e:
            print("%-19s %s: %s" % (model.id, model.repo, str(e).splitlines()[0][:90]))
            bad += 1
            continue
        files = [s for s in info.siblings
                 if model.files is None or any(fnmatch(s.rfilename, p) for p in model.files)]
        size = sum(s.size or 0 for s in files) / 1e9
        missing = [p for p in model.files or [] if not any(fnmatch(s.rfilename, p) for s in files)]
        card = info.card_data or {}
        lic = card.get("license_name") or card.get("license") or "?"
        off = abs(size - model.gb) > max(0.05, 0.02 * model.gb)
        bad += off or bool(missing)
        print("%-19s %8.2f GB (table %8.2f)%s  hub license %-28s table %s%s" % (
            model.id, size, model.gb, " SIZE" if off else "     ", lic, model.license,
            "  MISSING " + ", ".join(missing) if missing else ""))
    return bad


def main(argv=None):
    a = parser().parse_args(argv)
    if a.models_dir:
        os.environ["RECLANK_MODELS"] = a.models_dir
    m = hardware.detect(S.models_dir())
    hardware.assume(m, ram=a.assume_ram, vram=a.assume_vram, cc=a.assume_cc, tier=a.assume_cpu,
                    cores=a.assume_cores, disk=a.assume_disk)
    if a.check_hub:
        return 1 if check_hub() else 0
    interactive = sys.stdin.isatty() and not (a.yes or a.json)
    if a.model is not None and by_id(a.model) is None:
        raise SystemExit("no model %r; the ids: %s" % (a.model, ", ".join(x.id for x in MODELS)))
    if a.json:
        if a.model:
            model = by_id(a.model)
            out = {"model": a.model, "fit": S.fit(model, m),
                   "download": S.download_cmd(model, S.models_dir(), m.hf_cli),
                   "serve": {k: S.serve_cmd(model, S.models_dir(), k) for k in ("gpu", "cpu") if getattr(model, k)},
                   "request": S.request_cmd(model)}
        else:
            out = S.report_json(m)
        print(json.dumps(out, indent=1, default=str))
        return 0
    if a.model is None:
        print(S.report_text(m, all_rows=a.all))
        if not interactive:
            return 0
        s = S.suggest(m)
        while True:
            pick = _ask("\nThe guide of which model? (an id%s; q: quit) " % (", Enter: " + s.id if s else ""))
            if pick.lower() in ("q", "quit", "exit") or (not pick and not s):
                return 0
            if by_id(pick or s.id):
                break
            print("no model %r" % pick)
        a.model = pick or s.id
    model = by_id(a.model)
    print()
    print(S.guide_text(model, m))
    if model.status != "ready":
        if a.download:
            raise SystemExit("%s is not a plain download (see above)" % model.id)
        return 0
    want = a.download
    if not want and interactive:
        want = _ask("\nDownload it now (%.1f GB)? [y/N] " % model.gb).lower() in ("y", "yes")
    elif want and not a.yes:
        if not interactive:
            raise SystemExit("--download asks first: run it in a terminal, or add --yes")
        want = _ask("\nDownload %s (%.1f GB) into %s? [y/N] " % (model.repo, model.gb, S.models_dir())).lower() \
            in ("y", "yes")
    if want:
        if not m.hub:
            raise SystemExit("the download needs huggingface_hub: pip install huggingface_hub")
        path = S.download(model, S.models_dir())
        print("downloaded into %s" % path)
        print("Start the server with the command of step 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
