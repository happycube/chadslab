#!/bin/bash
# Build the CUDA library for each arch with each toolkit (compile only; no
# GPU needed), and report which builds pass (REFACTOR_PLAN.md, section 4.2):
#
#   scripts/check_cuda_builds.sh [ARCH ...]       default: 52 61 70 75 80 86 89 90 120 121
#
# The toolkits are /usr/local/cuda-12* and /usr/local/cuda-13* (CUDA_DIRS
# sets others). A toolkit that does not know an arch is skipped (CUDA 13
# starts at sm_75; CUDA 12 cannot build sm_121 before 12.9).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CSRC=$HERE/../reclank/backends/cuda/csrc
INC=$HERE/../reclank/backends/include
ARCHS=${*:-52 61 70 75 80 86 89 90 120 121}
DIRS=${CUDA_DIRS:-$(ls -d /usr/local/cuda-12* /usr/local/cuda-13* 2>/dev/null | sort -V)}
OUT=${TMPDIR:-/tmp}/reclank-cuda-builds
mkdir -p "$OUT"
fail=0
for d in $DIRS; do
    nvcc=$d/bin/nvcc
    [ -x "$nvcc" ] || continue
    ver=$("$nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
    known=$("$nvcc" --list-gpu-code 2>/dev/null | tr '\n' ' ')
    for a in $ARCHS; do
        if ! echo " $known " | grep -q " sm_${a%a} "; then
            printf "CUDA %-5s sm_%-5s skipped (not in this toolkit)\n" "$ver" "$a"
            continue
        fi
        log=$OUT/cuda${ver}_sm${a}.log
        t0=$(date +%s)
        if "$nvcc" -O3 -arch=sm_$a -Wno-deprecated-gpu-targets -I "$INC" -I "$CSRC" -c \
                -o "$OUT/cuda${ver}_sm${a}.o" "$CSRC/reclank_cuda.cu" > "$log" 2>&1; then
            printf "CUDA %-5s sm_%-5s ok   (%ds)\n" "$ver" "$a" $(( $(date +%s) - t0 ))
        else
            printf "CUDA %-5s sm_%-5s FAIL (%d errors; %s)\n" "$ver" "$a" "$(grep -c 'error' "$log")" "$log"
            fail=1
        fi
        rm -f "$OUT/cuda${ver}_sm${a}.o"
    done
done
exit $fail
