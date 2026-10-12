#!/bin/bash
# Run the golden dumps of every family and backend, each in its own process:
#   tests/golden/run_golden.sh OUT [CONFIG ...]
# A CONFIG is FAMILY:BACKEND with BACKEND gpu, cpu-native (the best tier of
# the CPU), or cpu-TIER (a tier of backends/cpu/build.py: cpu-avx2, cpu-avx512, ...)
# (default: all of DEFAULT). gpu-resident: the Gemma GPU path with every
# expert on the GPU (g26 on a 24 GB card). OUT/models.json gives the model paths
# (models.example.json). GOLDEN_CPUS and GOLDEN_THREADS fix the CPU set and
# the OpenMP team: keep them the same for runs that are compared.
set -u
OUT=${1:?OUT}; shift
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PYTHON:-python3}
CPUS=${GOLDEN_CPUS:-0-$(( $(nproc) - 1 ))}
THREADS=${GOLDEN_THREADS:-$(nproc)}
DEFAULT="g26:cpu-native g12:cpu-native e4b:cpu-native q36:cpu-native q38:cpu-native
g26:cpu-avx2 g12:cpu-avx2 e4b:cpu-avx2 q36:cpu-avx2
g26:gpu g12:gpu e4b:gpu q36:gpu q38:gpu"
CONFIGS=${*:-$DEFAULT}
mkdir -p "$OUT"
fail=0
for c in $CONFIGS; do
    fam=${c%%:*}; be=${c#*:}
    tier=; backend=$be
    case $be in
        cpu-native) backend=cpu ;;
        cpu-*) tier=${be#cpu-}; backend=cpu ;;
    esac
    mkdir -p "$OUT/$be"
    echo "== $fam $be ($(date +%T))"
    env ${tier:+RECLANK_CPU=$tier} OMP_NUM_THREADS=$THREADS PYTHONPATH="$ROOT" \
        taskset -c "$CPUS" "$PY" "$HERE/golden.py" run --family "$fam" --backend "$backend" \
        --out "$OUT/$be" --models "$OUT/models.json" > "$OUT/$be/$fam.log" 2>&1
    rc=$?
    tail -1 "$OUT/$be/$fam.log"
    [ $rc -ne 0 ] && { echo "   exit $rc (see $OUT/$be/$fam.log)"; fail=1; }
done
exit $fail
