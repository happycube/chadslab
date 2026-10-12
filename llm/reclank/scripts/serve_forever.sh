#!/bin/bash
# Run a server command again when it ends with an error (a native crash, or
# serve_qwen4.py exit 70 after a fatal CUDA error). A clean exit (0) or
# Ctrl-C (130) stops it. The pause doubles up to 60 s when it ends often.
#
#     scripts/serve_forever.sh python scripts/serve_qwen4.py --port 8082 ...
wait=5
while true; do
    start=$(date +%s)
    "$@"
    code=$?
    if [ $code -eq 0 ] || [ $code -eq 130 ]; then
        exit $code
    fi
    echo "[serve_forever] $(date '+%F %T') exit $code; again in ${wait} s" >&2
    sleep $wait
    if [ $(( $(date +%s) - start )) -gt 600 ]; then wait=5; else wait=$(( wait * 2 > 60 ? 60 : wait * 2 )); fi
done
