#!/usr/bin/env bash
#
# Create a self-contained PyTorch venv for the Gemma 4 12B QAT checkpoint
# (google/gemma-4-12B-it-qat-q4_0-unquantized), quantized to 4-bit at load time.
#
# Usage:
#   ./setup.sh                # auto-detect: CUDA if nvidia-smi works, else CPU
#   TORCH_VARIANT=cpu ./setup.sh
#   TORCH_VARIANT=cu130 ./setup.sh
#   PYTHON_VERSION=3.13 ./setup.sh
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VARIANT="${TORCH_VARIANT:-auto}"
VENV="$DIR/.venv"

# Keep every cache inside the project. Under the DSH file sandbox $HOME is
# read-only, so uv/pip/HF caches must live under the writable workspace.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DIR/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$DIR/.cache/uv-python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DIR/.cache/pip}"
export TMPDIR="${TMPDIR:-$DIR/.cache/tmp}"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

if [[ "$TORCH_VARIANT" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    TORCH_VARIANT=cu130
  else
    TORCH_VARIANT=cpu
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/)." >&2
  echo "       install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo ">> project : $DIR"
echo ">> python  : $PYTHON_VERSION"
echo ">> torch   : 2.14.0+$TORCH_VARIANT"
echo ">> venv    : $VENV"

uv venv --relocatable --python "$PYTHON_VERSION" "$VENV"

if [[ "$TORCH_VARIANT" == "cpu" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX="https://download.pytorch.org/whl/$TORCH_VARIANT"
fi

# torch + torchvision + torchaudio must all come from the same build channel.
echo ">> installing torch/torchvision/torchaudio (2.14.0+$TORCH_VARIANT) from $TORCH_INDEX"
uv pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX" \
  "torch==2.14.0" torchvision torchaudio

if [[ "$TORCH_VARIANT" != "cpu" ]]; then
  # torchao int4 weight-only kernels need mslk, which ships per-CUDA wheels only.
  echo ">> installing mslk (torchao int4 kernels) from $TORCH_INDEX"
  uv pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX" mslk || \
    echo "   warning: mslk install failed; use --quant int8 or --quant nf4 instead of int4"
fi

echo ">> installing the rest from PyPI"
uv pip install --python "$VENV/bin/python" -r "$DIR/requirements.txt"

# ---------------------------------------------------------------------------
# Make the venv portable across hosts.
#
# This project is shared over sshfs and may be seen at a different absolute
# path on each machine, so no host-specific path may be baked into the venv:
#   1. point bin/python at the project-local interpreter with a RELATIVE symlink
#      (the interpreter under .cache/uv-python is inside the project)
#   2. uv already made bin/activate* relocatable (see --relocatable above)
#   3. give console scripts an env-based shebang
# ---------------------------------------------------------------------------
REAL_PY="$(readlink -f "$VENV/bin/python" 2>/dev/null || true)"
if [ -n "$REAL_PY" ] && [ -f "$REAL_PY" ]; then
  REL_PY="$(cd "$DIR" && "$VENV/bin/python" -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$REAL_PY" "$VENV/bin")"
  ln -sf "$REL_PY" "$VENV/bin/python"
  echo ">> bin/python -> $REL_PY (relative)"
  # pyvenv.cfg "home" is informational once bin/python is a relative symlink;
  # keep it relative so the file carries no host-specific path either.
  PY_BIN_DIR="$(dirname "$REAL_PY")"
  REL_HOME="$(cd "$DIR" && "$VENV/bin/python" -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$PY_BIN_DIR" "$VENV")"
  sed -i "s|^home = .*|home = $REL_HOME|" "$VENV/pyvenv.cfg"
  echo ">> pyvenv.cfg home -> $REL_HOME (relative)"
fi

for f in "$VENV"/bin/*; do
  [ -f "$f" ] || continue
  case "$(basename "$f")" in
    activate*|python*|deactivate*) continue ;;
  esac
  if head -c 2 "$f" | grep -q '^#!'; then
    sed -i '1s|^#!.*|#!/usr/bin/env python3|' "$f"
  fi
done
echo ">> console scripts shebang -> /usr/bin/env python3"

echo
echo ">> done."
echo "   source $VENV/bin/activate"
echo "   python scripts/check_env.py"
echo "   python scripts/download_model.py          # ~24 GB"
echo "   python scripts/chat.py --prompt 'Hello in one sentence.'"
