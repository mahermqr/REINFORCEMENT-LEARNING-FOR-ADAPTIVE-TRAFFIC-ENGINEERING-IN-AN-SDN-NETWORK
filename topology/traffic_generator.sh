#!/usr/bin/env bash
# ==============================================================================
# SDN Mininet Network Traffic Generator (EC499)
# Generates realistic synthetic traffic patterns:
#  1. Continuous background unicast flows (Mice & Elephant flows)
#  2. Bursty Poisson traffic load surges
#  3. Asymmetric cross-pod elephant flows to test lateral link offload
# ==============================================================================

set -e

echo "======================================================================"
echo " Starting SDN Traffic Generator for Adaptive Traffic Engineering"
echo " Modes: Unicast Background, Bursty Surges, Elephant Flows"
echo "======================================================================"

check_mininet() {
    if ! pgrep -f "mininet:" >/dev/null 2>&1; then
        echo "[!] Error: No active Mininet host processes detected." >&2
        echo "    Please start the Mininet network first:" >&2
        echo "      sudo python3 topology/mininet_topo.py" >&2
        echo "    Or if running without Mininet, use the autonomous simulation driver:" >&2
        echo "      python3 simulate_live_traffic.py --mode auto" >&2
        exit 1
    fi
}

check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        if ! sudo -n true 2>/dev/null; then
            if [ -t 0 ]; then
                echo "[*] Requesting sudo credentials for Mininet host namespace access..."
                sudo -v || { echo "[!] Sudo authorization failed."; exit 1; }
            else
                echo "[!] Error: Sudo privileges required to attach to Mininet host namespaces." >&2
                echo "    Please run with 'sudo' or authenticate sudo credentials first ('sudo -v')." >&2
                exit 1
            fi
        fi
    fi
}

if [ "$EUID" -ne 0 ]; then
    SUDO_CMD="sudo"
else
    SUDO_CMD=""
fi

mn_exec() {
    local host="$1"
    shift
    local pid
    pid=$(pgrep -f "mininet:${host}$" | head -n 1)
    if [ -z "$pid" ]; then
        pid=$(pgrep -f "mininet:${host}\b" | head -n 1)
    fi
    if [ -z "$pid" ]; then
        echo "[!] Error: Host '${host}' not found in active Mininet network." >&2
        return 1
    fi
    $SUDO_CMD mnexec -a "$pid" "$@"
}

clean_stale_iperf() {
    for host in h1 h2 h3 h4 h5 h6 h7 h8; do
        local pid
        pid=$(pgrep -f "mininet:${host}$" | head -n 1)
        if [ -n "$pid" ]; then
            $SUDO_CMD mnexec -a "$pid" pkill -f iperf 2>/dev/null || true
        fi
    done
}

cleanup() {
    echo -e "\n[*] Stopping traffic generator and cleaning up background iperf processes..."
    clean_stale_iperf
    kill $(jobs -p) 2>/dev/null || true
    echo "[*] Cleanup complete."
}

generate_background() {
    echo "[+] Launching continuous background traffic (Mice flows: web/RPC)..."
    mn_exec h1 iperf -s -u -p 5001 &
    mn_exec h5 iperf -s -u -p 5002 &
    mn_exec h7 iperf -s -u -p 5003 &
    sleep 1
    mn_exec h2 iperf -c 10.0.0.1 -u -p 5001 -b 5M -t 300 &
    mn_exec h3 iperf -c 10.0.0.5 -u -p 5002 -b 8M -t 300 &
    mn_exec h6 iperf -c 10.0.0.7 -u -p 5003 -b 5M -t 300 &
    echo "[+] Background mice flows initiated."
}

generate_elephant_burst() {
    echo "[+] Launching high-bandwidth elephant flow (h4 -> h8, 45 Mbps)..."
    mn_exec h8 iperf -s -u -p 5004 &
    sleep 1
    mn_exec h4 iperf -c 10.0.0.8 -u -p 5004 -b 45M -t 120 &
    echo "[+] Elephant flow active across core/lateral mesh."
}

generate_pod_surge() {
    echo "[+] Launching asymmetric pod surge (Pod 1 to Pod 2 cross-links)..."
    mn_exec h1 iperf -s -u -p 5005 &
    mn_exec h6 iperf -s -u -p 5006 &
    sleep 1
    mn_exec h2 iperf -c 10.0.0.6 -u -p 5006 -b 30M -t 60 &
    mn_exec h3 iperf -c 10.0.0.1 -u -p 5005 -b 25M -t 60 &
    echo "[+] Pod surge active."
}

check_mininet
check_sudo

trap cleanup EXIT INT TERM
clean_stale_iperf

case "$1" in
    background)
        generate_background
        ;;
    elephant)
        generate_elephant_burst
        ;;
    surge)
        generate_pod_surge
        ;;
    all|*)
        generate_background
        sleep 3
        generate_elephant_burst
        sleep 5
        generate_pod_surge
        ;;
esac

echo "[*] Traffic generation active. Press Ctrl+C to stop."
wait
