#!/bin/bash
# start.sh: start a reclank server for a model, the one command to remember.
#
#     ./start.sh                         the models downloaded here and the mode each runs in
#     ./start.sh gemma4-26b              start a model by its id (python -m reclank.setup lists them)
#     ./start.sh ~/models/x.gguf         or by its file or folder
#     ./start.sh gemma4-26b -d           in the background: waits until it serves, prints pid and log
#     ./start.sh gemma4-26b --cpu        choose the mode (default: the GPU when the model fits it)
#     ./start.sh gemma4-26b --ctx 8192   arguments it does not know go to the server
#     ./start.sh gemma4-26b --dry-run    print the command; start nothing
#     ./start.sh --stop                  stop the reclank server on the port (8080; --port N)
#     ./start.sh --help                  every option and setting
#
# A chat model serves an OpenAI-compatible API on http://127.0.0.1:8080/v1 (embeddings 8082,
# generation 8084). The key clients send: KEY, else RECLANK_API_KEY, else the first line of
# ~/.config/reclank/api_key, else change-me.
#
# First use: python -m reclank.setup shows what this machine runs, and
# python -m reclank.setup --model ID --download fetches a model into ~/models (RECLANK_MODELS).
#
# PY     the Python with reclank's requirements (default: the active venv's, else .venv's here,
#        else ~/.venvs/reclank's, else python3)
# The settings of a machine (CUDA_HOME or NVCC, CUDA_VISIBLE_DEVICES, RECLANK_MODELS, ...) go in
# ~/.config/reclank/env (or the file RECLANK_ENV names), sourced first when it exists.
#
# The work is done by reclank/setup/start.py; the models come from reclank/setup/catalog.py.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
ENVF=${RECLANK_ENV:-$HOME/.config/reclank/env}
# shellcheck disable=SC1090
[ -f "$ENVF" ] && . "$ENVF"
if [ -z "${PY:-}" ]; then
    for p in "${VIRTUAL_ENV:-/nonexistent}/bin/python" .venv/bin/python "$HOME/.venvs/reclank/bin/python"; do
        [ -x "$p" ] && { PY=$p; break; }
    done
    PY=${PY:-python3}
fi
command -v "$PY" > /dev/null || { echo "start.sh: no Python at $PY (PY=...)" >&2; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m reclank.setup.start "$@"
