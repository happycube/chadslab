#!/bin/bash
# Run a command in a bubblewrap sandbox: no network, no home directory, CPU only.
#   sbx.sh VENV OUTDIR [-r HOSTPATH:SANDBOXPATH]... -- CMD...
# VENV: name under $SBX_ROOT/venvs (bound read-only at /venv); SBX_ROOT: the sandbox's root (venvs, sources).
# OUTDIR: the one writable dir (bound at /out).
# -r: extra read-only binds (repo copy, model dir, input files, driver scripts).
set -euo pipefail
SR=${SBX_ROOT:?set SBX_ROOT: the directory of the venvs and the sources}
venv=$1; out=$2; shift 2
binds=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case $1 in
    -r) host=${2%%:*}; dst=${2#*:}; binds+=(--ro-bind "$host" "$dst"); shift 2 ;;
    *) echo "bad arg $1" >&2; exit 2 ;;
  esac
done
shift
mkdir -p "$out"
T=${SBX_THREADS:-12}
exec /usr/bin/bwrap \
  --unshare-all --die-with-parent --new-session --clearenv \
  --setenv PATH /venv/bin:/usr/bin:/bin \
  --setenv HOME /home/sandbox \
  --setenv OMP_NUM_THREADS "$T" --setenv MKL_NUM_THREADS "$T" \
  --setenv CUDA_VISIBLE_DEVICES "" \
  --setenv HF_HUB_OFFLINE 1 --setenv TRANSFORMERS_OFFLINE 1 \
  --setenv PYTHONDONTWRITEBYTECODE 1 \
  --setenv MPLCONFIGDIR /tmp/mpl \
  --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --ro-bind /bin /bin --ro-bind /etc /etc \
  --tmpfs /home --tmpfs /tmp --proc /proc --dev /dev \
  --dir /home/sandbox \
  --ro-bind "$SR/venvs/$venv" /venv \
  "${binds[@]}" \
  --bind "$out" /out \
  --remount-ro / \
  --chdir /out \
  "$@"
