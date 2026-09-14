#!/usr/bin/env bash
# ==============================================================================
# SDN Mininet Network Traffic Generator & DDoS Attack Simulator
# Generates realistic background unicast, multicast, and attack traffic
# ==============================================================================

set -e

echo "======================================================================"
echo " Starting SDN Traffic Generator"
echo " Modes: Unicast (iperf), Multicast (iperf UDP), DDoS Flooding (iperf)"
echo "======================================================================"

# Verify Mininet is running and has active host namespaces
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

# Check for root or cached sudo credentials
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

# Helper to execute command inside a Mininet host namespace using mnexec
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

# Clean up stale iperf processes on mininet hosts
clean_stale_iperf() {
    for host in h1 h2 h3 h4 h5 h6 h7 h8; do
        local pid
        pid=$(pgrep -f "mininet:${host}$" | head -n 1)
        if [ -n "$pid" ]; then
            $SUDO_CMD mnexec -a "$pid" pkill -f iperf 2>/dev/null || true
        fi
    done
}

# Cleanup on script termination
cleanup() {
    echo -e "\n[*] Stopping traffic generator and cleaning up background iperf processes..."
    clean_stale_iperf
    kill $(jobs -p) 2>/dev/null || true
    echo "[*] Cleanup complete."
}

generate_unicast() {
    echo "[+] Launching continuous background unicast flows..."
    # h2 -> h1 (10 Mbps, port 5001)
    # h3 -> h5 (15 Mbps, port 5002)
    # h6 -> h1 (10 Mbps, port 5001)
    mn_exec h1 iperf -s -u -p 5001 &
    mn_exec h5 iperf -s -u -p 5002 &
    sleep 1
    mn_exec h2 iperf -c 10.0.0.1 -u -p 5001 -b 10M -t 300 &
    mn_exec h3 iperf -c 10.0.0.5 -u -p 5002 -b 15M -t 300 &
    mn_exec h6 iperf -c 10.0.0.1 -u -p 5001 -b 10M -t 300 &
    echo "[+] Unicast background traffic initiated."
}

generate_multicast() {
    echo "[+] Launching multicast UDP stream to group 224.1.1.1 (port 5005)..."
    mn_exec h5 iperf -s -u -B 224.1.1.1 -p 5005 &
    mn_exec h7 iperf -s -u -B 224.1.1.1 -p 5005 &
    mn_exec h8 iperf -s -u -B 224.1.1.1 -p 5005 &
    sleep 1
    mn_exec h2 iperf -c 224.1.1.1 -u -p 5005 -b 15M -t 300 &
    echo "[+] Multicast streams initiated."
}

generate_ddos() {
    echo "[!] Launching high-rate DDoS flood from attacker h4 -> target h1..."
    mn_exec h4 iperf -c 10.0.0.1 -u -p 5001 -b 50M -t 60 &
    echo "[!] DDoS flood active for 60 seconds."
}

# 1. Pre-flight checks
check_mininet
check_sudo

# 2. Setup trap and clean existing instances
trap cleanup EXIT INT TERM
clean_stale_iperf

# 3. Dispatch requested traffic mode
case "$1" in
    unicast)
        generate_unicast
        ;;
    multicast)
        generate_multicast
        ;;
    ddos)
        generate_ddos
        ;;
    all|*)
        generate_unicast
        generate_multicast
        sleep 5
        generate_ddos
        ;;
esac

echo "[*] Traffic generation running. Press Ctrl+C to terminate background jobs."
wait
