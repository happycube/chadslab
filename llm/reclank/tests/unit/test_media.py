"""models/media.py: ffmpeg and the pipelines' threads on the start CPUs (not the one CPU OMP_PROC_BIND gives
the main thread), and the colour of an encode."""
from __future__ import annotations

import os
import threading

import pytest

from reclank.models import media


def _cpus(text):
    line = next(l for l in text.splitlines() if l.startswith("Cpus_allowed_list"))
    return line.split(":")[1].strip()


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity") or len(media.start_cpus() or ()) < 2,
                    reason="needs two CPUs to start on")
def test_tool_and_thread_on_start_cpus():
    """a tool started from a thread bound to one CPU runs on the start CPUs without OMP_PROC_BIND, and the
    thread keeps its CPU; widen() gives a thread the start CPUs"""
    start = media.start_cpus()
    one = min(start)
    out = {}

    def body():
        os.sched_setaffinity(0, {one})
        p = media.run(["sh", "-c", "grep Cpus_allowed_list /proc/self/status; echo bind=$OMP_PROC_BIND"],
                      capture_output=True, text=True)
        out["child"], out["after"] = p.stdout, os.sched_getaffinity(0)
        media.widen()
        out["widened"] = os.sched_getaffinity(0)
    t = threading.Thread(target=body)
    t.start()
    t.join()
    assert out["after"] == {one} and out["widened"] == start
    assert _cpus(out["child"]) != str(one) and "bind=\n" in out["child"] + "\n"


def test_yuv_colors():
    """the yuv420p conversion by the source's tags: its matrix and range for the scale filter, the tags
    kept; a source without a matrix tag BT.601 limited and tagged so"""
    vf, tags = media.yuv_colors({})
    assert vf == "scale=out_color_matrix=bt601:out_range=tv"
    assert tags == {"color_space": "smpte170m", "color_range": "tv"}
    src = {"color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709", "color_range": "pc"}
    vf, tags = media.yuv_colors(src)
    assert vf == "scale=out_color_matrix=bt709:out_range=pc" and tags == src
    vf, tags = media.yuv_colors({"color_space": "bt470bg"})
    assert vf == "scale=out_color_matrix=bt601:out_range=tv" and tags["color_space"] == "bt470bg"
    args = media.yuv_args(src)
    assert args[:2] == ["-vf", "scale=out_color_matrix=bt709:out_range=pc"]
    assert args[args.index("-color_range") + 1] == "pc"
