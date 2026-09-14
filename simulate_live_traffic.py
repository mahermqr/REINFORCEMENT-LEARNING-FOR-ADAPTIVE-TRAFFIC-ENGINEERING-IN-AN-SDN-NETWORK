#!/usr/bin/env python3
"""
Live Traffic Generator & SDN Simulation Driver for Ryu Controller (EC499)
Injects dynamic traffic patterns, multicast requests, and DDoS bursts
to drive the live Web Dashboard (http://localhost:8080) and test the Multi-Agent DRL fabric.
"""

import time
import os
import json
import argparse
import urllib.request

BASE_URL = os.environ.get("SDN_CONTROLLER_URL", "http://localhost:8080")

def http_get(endpoint):
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req, timeout=3) as resp:
        return json.loads(resp.read().decode('utf-8'))

def http_post(endpoint, data=None):
    payload = json.dumps(data or {}).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}{endpoint}", data=payload, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=3) as resp:
        return json.loads(resp.read().decode('utf-8'))

def run_simulation(mode="auto", interval=4.0, max_steps=0):
    print("=" * 80)
    print(" 🚀 STARTING SDN LIVE TRAFFIC GENERATOR & DRL TELEMETRY DRIVER")
    print(f" Target Controller Web API: {BASE_URL}")
    print(f" Simulation Mode:           {mode.upper()} (Interval: {interval}s)")
    print("=" * 80)

    step = 0
    scenarios = ["traffic_burst", "core_jamming", "multicast", "ddos_attack", "reset"]
    scenario_descriptions = {
        "traffic_burst": "⚡ Pod Ingress Surge (h2->h5, h3->h7) via Double DQN",
        "core_jamming":  "🔥 Core Switch Saturation (96% Load) -> Lateral Cross-Link Offload",
        "multicast":     "📡 Multicast Group 224.1.1.1 Stream via Dueling DQN Steiner Tree",
        "ddos_attack":   "🚨 High-Rate DDoS Flooding (h4: 5200 PPS) -> DDPG Drop Rule",
        "reset":         "🔄 Nominal Baseline Network Stabilization"
    }

    try:
        while True:
            step += 1
            if max_steps > 0 and step > max_steps:
                print("\n[Done] Reached maximum requested steps.")
                break

            # If specific mode requested or cycling in auto
            current_action = mode if mode != "auto" else scenarios[(step - 1) % len(scenarios)]
            action_desc = scenario_descriptions.get(current_action, current_action)

            # 1. Trigger simulation action on controller
            try:
                trigger_res = http_post(f"/api/simulate/{current_action}")
            except Exception as e:
                trigger_res = {"message": f"Connection pending ({e})"}

            time.sleep(1.0)

            # 2. Query telemetry
            try:
                topo = http_get("/api/topology")
                sec = http_get("/api/security")
                stats = http_get("/api/stats")
                rl = http_get("/api/rl_metrics")

                switches_count = len(topo.get('nodes', []))
                links_count = len(topo.get('links', []))
                throughput = stats.get('total_throughput_mbps', 0.0)
                pps = stats.get('total_packet_rate_pps', 0.0)
                entropy = sec.get('shannon_entropy', 1.0)
                blocked = sec.get('blocked_ips', [])
                flows = stats.get('active_flows_count', 0)

                # Link loads
                max_u = max([l.get('utilization', 0.0) for l in topo.get('links', [])] or [0.0])

                status_str = "🚨 ATTACK MITIGATED" if (blocked or sec.get('ddos_alert')) else "🟢 NORMAL"

                print(f"[Step {step:03d}] {action_desc}")
                print(f"         Switches: {switches_count} | Links: {links_count} | Throughput: {throughput:5.1f} Mbps | "
                      f"PPS: {pps:5.0f} | Entropy: {entropy:.3f} | Max Link: {max_u:4.1f}% | {status_str} | Blocked: {blocked}")

            except Exception as e:
                print(f"[Step {step:03d}] Waiting for Ryu controller at {BASE_URL}: {e}")

            time.sleep(max(0.5, interval - 1.0))

    except KeyboardInterrupt:
        print("\n\n[*] Stopping traffic generator driver. Cleaning up...")
        try:
            http_post("/api/simulate/reset")
        except Exception:
            pass
        print("[*] Exited cleanly.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SDN Multi-Agent Live Traffic Generator & Driver")
    parser.add_argument('--mode', choices=['auto', 'traffic_burst', 'core_jamming', 'multicast', 'ddos_attack', 'reset'],
                        default='auto', help="Traffic simulation mode")
    parser.add_argument('--interval', type=float, default=4.0, help="Delay between scenario cycles (seconds)")
    parser.add_argument('--steps', type=int, default=0, help="Number of steps to run (0 for continuous)")
    args = parser.parse_args()

    run_simulation(mode=args.mode, interval=args.interval, max_steps=args.steps)
