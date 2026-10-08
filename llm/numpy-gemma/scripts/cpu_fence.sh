#!/bin/bash
# Keep other work off the CPUs of the model (the 2-socket Xeon: the model on
# taskset -c 0-19,24-43, the side processes on 20-23,44-47).
#
#   scripts/cpu_fence.sh status            the top CPU users, their allowed CPUs, their nodes
#   scripts/cpu_fence.sh fence [SIDE]      move them to SIDE (default 20-23,44-47)
#   scripts/cpu_fence.sh restore           put back what fence changed
#
# fence moves, without root:
#   - every process of this user (all its threads, taskset -a), but those
#     whose command line matches EXCLUDE (default: the runs of np_gemma) and
#     this script; the children they start later inherit SIDE;
#   - the running Docker containers (docker update --cpuset-cpus; the user is
#     in the docker group), e.g. clickhouse-server.
# The old CPU sets go to $STATE (restore reads them). Start the model after
# fence with its own set, as always:
#   taskset -c 0-19,24-43 python scripts/serve_qwen4.py ...
# fence prints the commands for root for the rest (the system services and
# the other users), with systemd's cpusets (cgroup v2, --runtime: until the
# next boot).
set -u
SIDE_DEFAULT="20-23,44-47"
STATE="${STATE:-$HOME/.cache/npg_cpu_fence.state}"
EXCLUDE="${EXCLUDE:-np_gemma|serve_qwen4|scripts/(bench|check|profile)_|npg-run/|cpu_fence}"

status() {
    printf "%-8s %-10s %6s  %-16s %s\n" PID USER %CPU ALLOWED "NODE0/NODE1 THREADS (>5% CPU)  COMMAND"
    ps -eo pid=,user=,pcpu=,comm= --sort=-pcpu | head -15 | while read -r pid user pcpu comm; do
        allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/$pid/status 2>/dev/null)
        split=$(ps -L -o psr=,pcpu= -p "$pid" 2>/dev/null |
                awk '$2>5 {if ($1<24) a++; else b++} END{printf "%d/%d", a, b}')
        printf "%-8s %-10s %6s  %-16s %-12s %s\n" "$pid" "$user" "$pcpu" "$allowed" "$split" "$comm"
    done
    echo
    docker ps --format '{{.Names}}' 2>/dev/null | while read -r c; do
        echo "docker $c: cpuset=$(docker inspect --format '{{.HostConfig.CpusetCpus}}' "$c")"
    done
}

fence() {
    side="${1:-$SIDE_DEFAULT}"
    mkdir -p "$(dirname "$STATE")"
    if [ -s "$STATE" ]; then
        echo "a fence is on ($STATE): restore it first" >&2
        exit 1
    fi
    : > "$STATE"
    me=$$
    n=0
    for pid in $(ps -u "$(id -u)" -o pid=); do
        [ "$pid" = "$me" ] && continue
        [ -r /proc/$pid/cmdline ] || continue
        cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
        [ -z "$cmd" ] && continue
        echo "$cmd" | grep -Eq "$EXCLUDE" && continue
        old=$(awk '/Cpus_allowed_list/ {print $2}' /proc/$pid/status 2>/dev/null) || continue
        if taskset -a -p -c "$side" "$pid" > /dev/null 2>&1; then
            echo "pid $pid $old" >> "$STATE"
            n=$((n + 1))
        fi
    done
    echo "moved $n processes of $(id -un) to $side"
    for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
        old=$(docker inspect --format '{{.HostConfig.CpusetCpus}}' "$c")
        if docker update --cpuset-cpus "$side" "$c" > /dev/null; then
            echo "docker $c ${old:-all}" >> "$STATE"
            echo "moved docker container $c to $side"
        fi
    done
    cat <<EOF

For the system services and the other users (needs root; until the next boot):
  sudo systemctl set-property --runtime system.slice AllowedCPUs=$side
  sudo systemctl set-property --runtime user-<UID>.slice AllowedCPUs=$side    # each other user
and to undo:
  sudo systemctl set-property --runtime system.slice AllowedCPUs=
EOF
}

restore() {
    if [ ! -s "$STATE" ]; then
        echo "no fence ($STATE)" >&2
        exit 1
    fi
    all="0-$(($(nproc --all) - 1))"
    while read -r kind id old; do
        case "$kind" in
        pid) taskset -a -p -c "$old" "$id" > /dev/null 2>&1 ;;
        docker) docker update --cpuset-cpus "$([ "$old" = all ] && echo "$all" || echo "$old")" "$id" \
                    > /dev/null 2>&1 && echo "docker $id back to $old" ;;
        esac
    done < "$STATE"
    echo "restored $(grep -c '^pid' "$STATE") processes"
    rm -f "$STATE"
}

case "${1:-status}" in
status) status ;;
fence) fence "${2:-}" ;;
restore) restore ;;
*) echo "usage: $0 status | fence [SIDE] | restore" >&2; exit 2 ;;
esac
