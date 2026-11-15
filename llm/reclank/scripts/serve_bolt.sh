#!/bin/bash
# Nemotron 3.5 Bolt through the OpenAI-compatible server, its weights from a weightkeeper: the
# LiLiCorr drafter when its directory is there, the image encoder on the CPU (--mmproj-gpu off: the
# GPU's memory stays with the hot experts), a 262144-position cache. A reclank server already on the
# port is stopped first (so this is also the restart), and so are the keepers of other models, whose
# memory Bolt's keeper needs.
#
#     scripts/serve_bolt.sh                                  # in the foreground on 127.0.0.1:8080
#     HOST=0.0.0.0 KEY=secret scripts/serve_bolt.sh -d       # detached: waits until it serves
#     scripts/serve_bolt.sh -d --image-budget 512            # other arguments go to the server
#
# BOLT          the checkpoint (default ~/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
# DRAFTER       the drafter (default $BOLT-LiLiCorr; off: none)
# HOST PORT CTX 127.0.0.1, 8080, 262144
# KEY           the API key; else RECLANK_API_KEY; else the file ~/.config/reclank/api_key; else the
#               server's default (server/http.py)
# KEEPER=0      the weights from the files, no keeper; KEEP_OTHERS=1: the other keepers stay
# PY            the Python of reclank's venv (default: the active venv's, else .venv's, else
#               ~/.venvs/reclank's, else python3)
# LOG           the server's log with -d (default ~/.cache/reclank/serve_bolt.log; the keeper's
#               beside it)
# The settings of a machine (the CUDA compiler of the builds, the GPUs to use) go in
# ~/.config/reclank/env (or the file RECLANK_ENV names), sourced first when it exists.
set -u
cd "$(dirname "$0")/.." || exit 1
ENVF=${RECLANK_ENV:-$HOME/.config/reclank/env}
# shellcheck disable=SC1090
[ -f "$ENVF" ] && . "$ENVF"

DETACH=0
if [ "${1:-}" = "-d" ] || [ "${1:-}" = "--detach" ]; then DETACH=1; shift; fi
BOLT=${BOLT:-$HOME/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CTX=${CTX:-262144}
LOG=${LOG:-$HOME/.cache/reclank/serve_bolt.log}
KLOG=$(dirname "$LOG")/weightkeeper_bolt.log
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
[ -f "$BOLT/config.json" ] || { echo "serve_bolt: no checkpoint at $BOLT (BOLT=...)" >&2; exit 1; }
BOLT=$(realpath "$BOLT")
DRAFTER=${DRAFTER:-$BOLT-LiLiCorr}
mkdir -p "$(dirname "$LOG")"

# ---- the server on the port (a reclank server only: anything else, stop here)
pid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -n 1 | cut -d= -f2)
if [ -n "$pid" ]; then
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -- "-m server"; then
        echo "serve_bolt: port $PORT is taken by pid $pid: $(tr '\0' ' ' < "/proc/$pid/cmdline" | cut -c1-120)" >&2
        exit 1
    fi
    echo "serve_bolt: stopping the server on port $PORT (pid $pid)"
    kill "$pid"
    for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    while kill -0 "$pid" 2>/dev/null; do sleep 1; done
fi

# ---- the keeper: Bolt's kept or started; the others stopped (their memory)
if [ "${KEEPER:-1}" != "0" ]; then
    have=0
    # (--list: NAME pid PID MODEL GB "GB" [locked] CHECKPOINT (SOCKET), or "no weightkeeper running")
    while read -r -a f; do
        [ ${#f[@]} -ge 7 ] && [[ ${f[2]} =~ ^[0-9]+$ ]] || continue
        name=${f[0]}; kpid=${f[2]}; path=${f[${#f[@]}-2]}
        if [ "$(realpath -m "$path")" = "$BOLT" ]; then have=1; continue; fi
        [ "${KEEP_OTHERS:-0}" = "1" ] && continue
        echo "serve_bolt: stopping the keeper $name (pid $kpid, ${f[4]} GB: $path)"
        if [ "$name" = "(default)" ]; then "$PY" scripts/weightkeeper.py --stop > /dev/null
        else "$PY" scripts/weightkeeper.py --stop --name "$name" > /dev/null; fi
        while kill -0 "$kpid" 2>/dev/null; do sleep 1; done
        sleep 5                          # (its memory back before the next keeper counts it)
    done < <("$PY" scripts/weightkeeper.py --list 2>/dev/null)
    if [ "$have" = "0" ]; then
        name=""
        "$PY" scripts/weightkeeper.py --status > /dev/null 2>&1 && name="--name bolt"   # (the default socket is taken)
        for try in 1 2 3; do
            echo "serve_bolt: starting Bolt's keeper (its log: $KLOG)"
            # shellcheck disable=SC2086
            setsid nohup "$PY" scripts/weightkeeper.py "$BOLT" $name > "$KLOG" 2>&1 < /dev/null &
            kp=$!
            until grep -q -E "ready in|not enough memory|Traceback" "$KLOG" 2>/dev/null || ! kill -0 $kp 2>/dev/null; do sleep 2; done
            grep -q "ready in" "$KLOG" && break
            grep -E "not enough memory|Error" "$KLOG" | tail -n 1 >&2
            [ $try -lt 3 ] && sleep 10
        done
        grep -q "ready in" "$KLOG" || { echo "serve_bolt: the keeper did not start (KEEPER=0: from the files)" >&2; exit 1; }
        grep "ready in" "$KLOG"
    fi
fi

# ---- the server
args=(-m server -m "$BOLT" --host "$HOST" --port "$PORT" --backend cuda --ctx "$CTX" --mmproj-gpu off)
[ "$DRAFTER" != "off" ] && [ -f "$DRAFTER/config.json" ] && args+=(--drafter "$DRAFTER")
args+=("$@")
# (no key given: the server's default, not an empty one, which would turn the check off)
if [ -n "$KEY" ]; then export RECLANK_API_KEY=$KEY; else unset RECLANK_API_KEY; fi
if [ "$DETACH" = "0" ]; then
    exec "$PY" "${args[@]}"
fi
setsid nohup "$PY" "${args[@]}" > "$LOG" 2>&1 < /dev/null &
sp=$!
echo "serve_bolt: the server starting (pid $sp, its log: $LOG)"
until grep -q -E "reclank server http" "$LOG" 2>/dev/null || ! kill -0 $sp 2>/dev/null; do sleep 2; done
if ! grep -q "reclank server http" "$LOG"; then
    echo "serve_bolt: the server stopped:" >&2
    tail -n 20 "$LOG" >&2
    exit 1
fi
grep -E "^ready in|reclank server http" "$LOG"
