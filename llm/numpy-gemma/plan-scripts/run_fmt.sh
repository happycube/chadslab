#!/bin/sh
# The BF12 / RQ10 / RQ12 test kernels of BF12_PLAN.md: the data, the CPU
# kernels, the GPU kernels. From numpy-gemma:  sh plan-scripts/run_fmt.sh [ORIG]
# NPG_FMT_DATA (default /tmp/np_gemma_fmt) holds the 350 MB of test data.
# PY: the python of the runtime (default: the venv of gemma4-12b-qat-pytorch).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
DATA=${NPG_FMT_DATA:-/tmp/np_gemma_fmt}
PY=${PY:-$REPO/../gemma4-12b-qat-pytorch/.venv/bin/python}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
export NPG_FMT_DATA=$DATA
mkdir -p "$DATA"
cd "$REPO"
[ -f "$DATA/meta.json" ] || PYTHONPATH=. "$PY" "$HERE/fmt_make_data.py" "$@"
gcc -O3 -march=native -mf16c -fopenmp -shared -fPIC -o "$DATA/fmt_cpu.so" "$HERE/fmt_cpu.c"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-18} OMP_PROC_BIND=spread OMP_WAIT_POLICY=ACTIVE \
    PYTHONPATH=. "$PY" "$HERE/fmt_cpu_bench.py"
if [ -x "$NVCC" ] && nvidia-smi >/dev/null 2>&1; then
    "$NVCC" -O3 -arch=native -w -I "$REPO/np_gemma/csrc" -o "$DATA/fmt_gpu" "$HERE/fmt_gpu.cu"
    "$DATA/fmt_gpu" "$DATA"
fi
# the three zero rules of BF12 (none, gap15, neg0)
PYTHONPATH=. "$PY" "$HERE/bf12_zero_make.py"
gcc -O3 -march=native -fopenmp -shared -fPIC -o "$DATA/bf12_zero_cpu.so" "$HERE/bf12_zero_cpu.c"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-18} OMP_PROC_BIND=spread OMP_WAIT_POLICY=ACTIVE \
    "$PY" "$HERE/bf12_zero_cpu_bench.py"
if [ -x "$NVCC" ] && nvidia-smi >/dev/null 2>&1; then
    "$NVCC" -O3 -arch=native -w -o "$DATA/bf12_zero_gpu" "$HERE/bf12_zero_gpu.cu"
    "$DATA/bf12_zero_gpu" "$DATA"
fi
