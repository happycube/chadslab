#!/bin/bash
# Qwen3.8-27B (byteshape's ShapeLearn GGUF, the IQ4_XS-3.84bpw file) through the OpenAI-compatible server
# on the GPU, with the DFlash 2 drafter (models/qwen/dflash2.py, 7 slots a round) when its GGUF is there. A
# reclank server already on the port is stopped first (so this is also the restart). No weightkeeper: the
# file's pages stay in the page cache; the start (about 1.5 minutes on a P6000) makes the GPU rows of the
# i-quants (docs/QWEN27_P6000.md) and the drafter's and verify groups' programs.
#
#     scripts/serve_qwen27.sh                               # in the foreground on 127.0.0.1:8080
#     HOST=0.0.0.0 KEY=secret scripts/serve_qwen27.sh -d    # detached: waits until it serves
#     scripts/serve_qwen27.sh -d --thinking off             # other arguments go to the server
#
# MODEL         the GGUF (default ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf)
# DRAFTER       the DFlash2 GGUF (default ~/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf;
#               off: none, the MTP layer of the file drafts instead)
# HOST PORT CTX 127.0.0.1, 8080, 131072
# KEY           the API key; else RECLANK_API_KEY; else the file ~/.config/reclank/api_key; else the
#               server's default (server/http.py)
# PY            the Python of reclank's venv (default: the active venv's, else .venv's, else
#               ~/.venvs/reclank's, else python3)
# LOG           the server's log with -d (default ~/.cache/reclank/serve_qwen27.log)
# The settings of a machine (the CUDA compiler of the builds, the GPUs to use) go in
# ~/.config/reclank/env (or the file RECLANK_ENV names), sourced first when it exists.
set -u
cd "$(dirname "$0")/.." || exit 1
ENVF=${RECLANK_ENV:-$HOME/.config/reclank/env}
# shellcheck disable=SC1090
[ -f "$ENVF" ] && . "$ENVF"

DETACH=0
if [ "${1:-}" = "-d" ] || [ "${1:-}" = "--detach" ]; then DETACH=1; shift; fi
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf}
DRAFTER=${DRAFTER:-$HOME/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CTX=${CTX:-131072}
LOG=${LOG:-$HOME/.cache/reclank/serve_qwen27.log}
if [ -z "${PY:-}" ]; then
    for p in "${VIRTUAL_ENV:-/nonexistent}/bin/python" .venv/bin/python "$HOME/.venvs/reclank/bin/python"; do
        [ -x "$p" ] && { PY=$p; break; }
    done
    PY=${PY:-python3}
fi
if [ -z "${KEY:-}" ]; then
    KEY=${RECLANK_API_KEY:-}
    [ -z "$KEY" ] && [ -f "$HOME/.config/reclank/api_key" ] && KEY=$(head -n 1 "$HOME/.config/reclank/api_key")
fi
[ -f "$MODEL" ] || { echo "serve_qwen27: no GGUF at $MODEL (MODEL=...)" >&2; exit 1; }
MODEL=$(realpath "$MODEL")
mkdir -p "$(dirname "$LOG")"

# ---- the server on the port (a reclank server only: anything else, stop here)
pid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -n 1 | cut -d= -f2)
if [ -n "$pid" ]; then
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -- "-m server"; then
        echo "serve_qwen27: port $PORT is taken by pid $pid: $(tr '\0' ' ' < "/proc/$pid/cmdline" | cut -c1-120)" >&2
        exit 1
    fi
    echo "serve_qwen27: stopping the server on port $PORT (pid $pid)"
    kill "$pid"
    for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    while kill -0 "$pid" 2>/dev/null; do sleep 1; done
fi

# ---- the server
args=(-m server -m "$MODEL" --host "$HOST" --port "$PORT" --backend cuda --ctx "$CTX")
[ "$DRAFTER" != "off" ] && [ -f "$DRAFTER" ] && args+=(--drafter "$(realpath "$DRAFTER")")
args+=("$@")
# (no key given: the server's default, not an empty one, which would turn the check off)
if [ -n "$KEY" ]; then export RECLANK_API_KEY=$KEY; else unset RECLANK_API_KEY; fi
if [ "$DETACH" = "0" ]; then
    exec "$PY" "${args[@]}"
fi
setsid nohup "$PY" "${args[@]}" > "$LOG" 2>&1 < /dev/null &
sp=$!
echo "serve_qwen27: the server starting (pid $sp, its log: $LOG)"
until grep -q -E "reclank server http" "$LOG" 2>/dev/null || ! kill -0 $sp 2>/dev/null; do sleep 2; done
if ! grep -q "reclank server http" "$LOG"; then
    echo "serve_qwen27: the server stopped:" >&2
    tail -n 20 "$LOG" >&2
    exit 1
fi
grep -E "^ready in|reclank server http" "$LOG"
