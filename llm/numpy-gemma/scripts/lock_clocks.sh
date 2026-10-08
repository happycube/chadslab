#!/usr/bin/env bash
# Lock the CPU clocks for repeatable measurements, and put them back.
#
#     sudo scripts/lock_clocks.sh lock [CORE_MHZ] [UNCORE_MHZ] [--fast MHZ [--fast-cpus LIST]]
#                                                                default 2400 2400
#     sudo scripts/lock_clocks.sh restore
#     scripts/lock_clocks.sh status
#
# --fast MHZ: the CPUs of the threads that drive the GPU at another clock
# (min = max = MHZ), the rest at CORE_MHZ. --fast-cpus LIST (default
# 0,1,19): the CPUs of np_gemma.gpu._place_threads for the server on
# 0-19,24-43 with the GPU on node 0: 0 the model thread (it launches the
# programs and runs the host side of each step), 1 the master of the teams
# (the GPU runner: the busiest thread of a decode, waits on the GPU and the
# CPU experts in turn), 19 the copy workers (the stager of the copies of
# HotCache). A clock above the base clock (2900 MHz here) is a turbo clock:
# turbo goes back on for it, and the others stay at CORE_MHZ (their max),
# but a turbo clock holds only while the power and the AVX license allow.
#
# lock: the performance governor, no turbo, and min = max = CORE_MHZ on every
# CPU (intel_pstate through sysfs); with the intel_uncore_frequency driver,
# min = max = UNCORE_MHZ for the uncore of each package (the memory
# controllers, the mesh, and UPI: the latency of memory and of the barriers
# of the parts). A clock below the all-core AVX-512 turbo holds under load.
# The first lock saves the settings in $SAVE; restore writes them back.
#
# A turbo clock moves with the count of active cores, the AVX-512 license,
# and the power limit, and the parts wait at each barrier for the slower
# socket; see SPLIT_PLAN.md.
set -euo pipefail

SAVE=/var/tmp/np-gemma-clocks.saved
CPU=/sys/devices/system/cpu
UNC=$CPU/intel_uncore_frequency

need_root() {
    if [ "$(id -u)" != 0 ]; then
        echo "run as root: sudo $0 $*" >&2
        exit 1
    fi
}

expand_list() {   # "0-11,24-35" -> 0 1 ... 11 24 ... 35
    local part
    for part in ${1//,/ }; do
        seq "${part%-*}" "${part#*-}"
    done
}

uncore_dirs() {
    [ -d "$UNC" ] && ls -d "$UNC"/package_* 2>/dev/null || true
}

status() {
    printf "governor %s, driver %s, no_turbo %s, epp %s\n" \
        "$(cat $CPU/cpu0/cpufreq/scaling_governor)" "$(cat $CPU/cpu0/cpufreq/scaling_driver)" \
        "$(cat $CPU/intel_pstate/no_turbo 2>/dev/null || echo -)" \
        "$(cat $CPU/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo -)"
    printf "cpu0 min/max %s/%s kHz (hardware %s-%s)\n" \
        "$(cat $CPU/cpu0/cpufreq/scaling_min_freq)" "$(cat $CPU/cpu0/cpufreq/scaling_max_freq)" \
        "$(cat $CPU/cpu0/cpufreq/cpuinfo_min_freq)" "$(cat $CPU/cpu0/cpufreq/cpuinfo_max_freq)"
    for node in $CPU/../node/node[0-9]*; do
        list=$(cat "$node/cpulist")
        mhz=$(for c in $(expand_list "$list"); do
                  cat $CPU/cpu$c/cpufreq/scaling_cur_freq; done |
              awk '{s += $1} END {if (NR) printf "%.0f", s / NR / 1000}')
        echo "$(basename "$node") (cpus $list): now ${mhz:-?} MHz"
    done
    # the CPUs whose clock range is not the range of most of them (lock --fast)
    local ranges
    ranges=$(for p in $CPU/cpu[0-9]*/cpufreq; do
                 c=${p%/cpufreq}; c=${c##*cpu}
                 echo "$(cat "$p/scaling_min_freq")/$(cat "$p/scaling_max_freq") $c $(( $(cat "$p/scaling_cur_freq") / 1000 ))"
             done | sort -n -k2)
    echo "$ranges" | awk '{n[$1]++; l[$1] = l[$1] (l[$1] ? ", " : "") $2 " (" $3 " MHz)"} END {
        for (r in n) if (n[r] > top) {top = n[r]; most = r}
        print "most cpus (" top "): min/max " most " kHz"
        for (r in n) if (r != most) print "cpus at min/max " r " kHz: " l[r]}'

    for d in $(uncore_dirs); do
        printf "uncore %s: min/max %s/%s kHz (hardware %s-%s)\n" "$(basename "$d")" \
            "$(cat "$d/min_freq_khz")" "$(cat "$d/max_freq_khz")" \
            "$(cat "$d/initial_min_freq_khz")" "$(cat "$d/initial_max_freq_khz")"
    done
    for h in /sys/class/hwmon/hwmon*; do
        if [ "$(cat "$h/name")" = coretemp ]; then
            echo "$(cat "$h/temp1_label"): $(( $(cat "$h/temp1_input") / 1000 )) C"
        fi
    done
    [ -f "$SAVE" ] && echo "saved settings: $SAVE (restore puts them back)" || true
}

save() {
    if [ -f "$SAVE" ]; then
        # a save from before the last boot holds the settings of that boot
        # (a lock, or the clocks of another memory layout): start again
        local boot now saved
        now=$(date +%s)
        boot=$((now - $(cut -d. -f1 /proc/uptime)))
        saved=$(stat -c %Y "$SAVE")
        if [ "$saved" -lt "$boot" ]; then
            echo "the saved settings are from before the last boot: saving them again"
            rm -f "$SAVE"
        else
            echo "keeping the settings saved before: $SAVE"
            return
        fi
    fi
    {
        echo "no_turbo $(cat $CPU/intel_pstate/no_turbo 2>/dev/null || echo -)"
        for p in $CPU/cpu[0-9]*/cpufreq; do
            echo "cpu $p $(cat "$p/scaling_governor") $(cat "$p/scaling_min_freq") $(cat "$p/scaling_max_freq") $(cat "$p/energy_performance_preference" 2>/dev/null || echo -)"
        done
        for d in $(uncore_dirs); do
            echo "uncore $d $(cat "$d/min_freq_khz") $(cat "$d/max_freq_khz")"
        done
    } > "$SAVE"
    echo "saved the settings in $SAVE"
}

# Write min and max in the order that keeps min <= max at each step.
set_range() {   # dir min_file max_file khz
    local d=$1 fmin=$2 fmax=$3 khz=$4
    if [ "$khz" -ge "$(cat "$d/$fmax")" ]; then
        echo "$khz" > "$d/$fmax"; echo "$khz" > "$d/$fmin"
    else
        echo "$khz" > "$d/$fmin"; echo "$khz" > "$d/$fmax"
    fi
}

lock() {
    local pos=() fast="" fast_cpus="0,1,19"
    while [ $# -gt 0 ]; do
        case $1 in
        --fast)      fast=$2; shift 2 ;;
        --fast-cpus) fast_cpus=$2; shift 2 ;;
        --*)         echo "unknown option $1" >&2; exit 2 ;;
        *)           pos+=("$1"); shift ;;
        esac
    done
    local core=${pos[0]:-2400} uncore=${pos[1]:-2400}
    local khz=$((core * 1000)) ukhz=$((uncore * 1000))
    local lo hi base fkhz=""
    lo=$(cat $CPU/cpu0/cpufreq/cpuinfo_min_freq); hi=$(cat $CPU/cpu0/cpufreq/cpuinfo_max_freq)
    base=$(cat $CPU/cpu0/cpufreq/base_frequency 2>/dev/null || echo "$hi")
    if [ "$khz" -lt "$lo" ] || [ "$khz" -gt "$base" ]; then
        echo "core clock $core MHz is outside $((lo / 1000))-$((base / 1000)) MHz (no turbo)" >&2; exit 1
    fi
    if [ -n "$fast" ]; then
        fkhz=$((fast * 1000))
        if [ "$fkhz" -lt "$lo" ] || [ "$fkhz" -gt "$hi" ]; then
            echo "fast clock $fast MHz is outside $((lo / 1000))-$((hi / 1000)) MHz" >&2; exit 1
        fi
        for c in $(expand_list "$fast_cpus"); do
            [ -d "$CPU/cpu$c/cpufreq" ] || { echo "no cpu $c" >&2; exit 1; }
        done
    fi
    save
    # turbo stays off unless a fast clock is a turbo clock; the max of the
    # other CPUs keeps them at CORE_MHZ either way
    if [ -f $CPU/intel_pstate/no_turbo ]; then
        if [ -n "$fkhz" ] && [ "$fkhz" -gt "$base" ]; then
            echo 0 > $CPU/intel_pstate/no_turbo
        else
            echo 1 > $CPU/intel_pstate/no_turbo
        fi
    fi
    for p in $CPU/cpu[0-9]*/cpufreq; do
        echo performance > "$p/scaling_governor"
        [ -f "$p/energy_performance_preference" ] && \
            { echo performance > "$p/energy_performance_preference" 2>/dev/null || true; }
        set_range "$p" scaling_min_freq scaling_max_freq "$khz"
    done
    if [ -n "$fkhz" ]; then
        for c in $(expand_list "$fast_cpus"); do
            set_range "$CPU/cpu$c/cpufreq" scaling_min_freq scaling_max_freq "$fkhz"
        done
    fi
    for d in $(uncore_dirs); do
        local ulo uhi
        ulo=$(cat "$d/initial_min_freq_khz"); uhi=$(cat "$d/initial_max_freq_khz")
        if [ "$ukhz" -lt "$ulo" ] || [ "$ukhz" -gt "$uhi" ]; then
            echo "uncore clock $uncore MHz is outside $((ulo / 1000))-$((uhi / 1000)) MHz" >&2; exit 1
        fi
        set_range "$d" min_freq_khz max_freq_khz "$ukhz"
    done
    [ -n "$(uncore_dirs)" ] || echo "no intel_uncore_frequency driver: the uncore clock stays free (modprobe intel_uncore_frequency)"
    if [ -n "$fkhz" ]; then
        echo "locked: core $core MHz, cpus $fast_cpus at $fast MHz$([ "$fkhz" -gt "$base" ] && echo " (turbo on)"), uncore $uncore MHz"
    else
        echo "locked: core $core MHz (no turbo), uncore $uncore MHz"
    fi
    status
}

restore() {
    if [ ! -f "$SAVE" ]; then
        echo "nothing saved in $SAVE" >&2; exit 1
    fi
    while read -r kind a b c d e; do
        case $kind in
        no_turbo)
            [ "$a" != - ] && echo "$a" > $CPU/intel_pstate/no_turbo ;;
        cpu)   # path governor min max epp
            echo "$b" > "$a/scaling_governor"
            echo "$d" > "$a/scaling_max_freq"; echo "$c" > "$a/scaling_min_freq"
            [ "$e" != - ] && { echo "$e" > "$a/energy_performance_preference" 2>/dev/null || true; } ;;
        uncore)   # path min max
            echo "$c" > "$a/max_freq_khz"; echo "$b" > "$a/min_freq_khz" ;;
        esac
    done < "$SAVE"
    rm -f "$SAVE"
    echo "restored"
    status
}

case "${1:-status}" in
lock)    need_root "$@"; shift; lock "$@" ;;
restore) need_root "$@"; restore ;;
status)  status ;;
*)       echo "usage: $0 lock [CORE_MHZ] [UNCORE_MHZ] [--fast MHZ [--fast-cpus LIST]] | restore | status" >&2
         exit 2 ;;
esac
