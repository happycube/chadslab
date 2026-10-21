#!/bin/bash
# Build bench/ds_prefill.cu (the kernel module kernels/dsv41/prefill.cu alone) when its sources
# change, then run it: ds_prefill.sh attn|hot DIR [reps]. NVCC, ARCH (86), OUT (the binary's folder).
set -e
B=$(cd "$(dirname "$0")" && pwd)
C=$(dirname "$B")
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
OUT=${OUT:-$HOME/.cache/reclank/bench}
mkdir -p "$OUT"
bin=$OUT/ds_prefill_sm${ARCH:-86}
src="$B/ds_prefill.cu $C/kernels/dsv41/prefill.cu $C/kernels/dsv41/hot.cu $C/kernels/dsv41/common.cuh $C/kernels/dsv41/types.cu $C/kernels/frag.cuh $C/kernels/operands.cu"
if [ ! -x "$bin" ] || [ -n "$(find $src -newer "$bin")" ]; then
    "$NVCC" -O3 -std=c++17 -arch=sm_${ARCH:-86} -lineinfo -I"$C" -I"$C/../../include" -o "$bin" "$B/ds_prefill.cu"
fi
exec "$bin" "$@"
