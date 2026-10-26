#!/bin/bash
# Build bench/nm_f32.cu (the kernel module kernels/nemotron/iq2_f32.cu alone) when its sources change,
# then run it: nm_f32.sh lin|hot|group|all [reps]. NVCC, ARCH (52), OUT (the binary's folder), NVCC_CCBIN.
set -e
B=$(cd "$(dirname "$0")" && pwd)
C=$(dirname "$B")
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
A=${ARCH:-52}
OUT=${OUT:-$HOME/.cache/reclank/bench}
mkdir -p "$OUT"
bin=$OUT/nm_f32_sm$A
src="$B/nm_f32.cu $C/kernels/nemotron/iq2_f32.cu $C/kernels/nemotron/iq2.cuh $C/kernels/operands.cu $C/port.cuh"
if [ ! -x "$bin" ] || [ -n "$(find $src -newer "$bin")" ]; then
    # (as the library: -dlcm=ca below sm_70, build.py)
    extra=""
    [ "$A" -lt 70 ] && extra="-Xptxas -dlcm=ca"
    "$NVCC" -O3 -std=c++17 -arch=sm_$A -lineinfo -Wno-deprecated-gpu-targets ${NVCC_CCBIN:+-ccbin $NVCC_CCBIN} \
        $extra -I"$C" -I"$C/../../include" -o "$bin" "$B/nm_f32.cu"
fi
exec "$bin" "$@"
