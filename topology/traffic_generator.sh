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

generate_unicast() {
    echo "[+] Launching continuous background unicast flows..."
    # h2 -> h1 (10 Mbps)
    # h3 -> h5 (15 Mbps)
    # h6 -> h1 (10 Mbps)
    sudo mn-exec h1 iperf -s -u &
    sudo mn-exec h5 iperf -s -u &
    sleep 1
    sudo mn-exec h2 iperf -c 10.0.0.1 -u -b 10M -t 300 &
    sudo mn-exec h3 iperf -c 10.0.0.5 -u -b 15M -t 300 &
    sudo mn-exec h6 iperf -c 10.0.0.1 -u -b 10M -t 300 &
    echo "[+] Unicast background traffic initiated."
}

generate_multicast() {
    echo "[+] Launching multicast UDP stream to group 224.1.1.1..."
    sudo mn-exec h5 iperf -s -u -B 224.1.1.1 &
    sudo mn-exec h7 iperf -s -u -B 224.1.1.1 &
    sudo mn-exec h8 iperf -s -u -B 224.1.1.1 &
    sleep 1
    sudo mn-exec h1 iperf -c 224.1.1.1 -u -b 20M -t 300 &
    echo "[+] Multicast streams initiated."
}

generate_ddos() {
    echo "[!] Launching high-rate DDoS flood from attacker h4 -> target h1..."
    sudo mn-exec h4 iperf -c 10.0.0.1 -u -b 50M -t 60 &
    echo "[!] DDoS flood active for 60 seconds."
}

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
