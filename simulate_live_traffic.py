#!/usr/bin/env python3
"""
Live Traffic Generator & SDN Simulation Driver for Ryu Controller (EC499).
Injects dynamic traffic patterns, pod surges, core jamming, and link failures
to drive the live Web Dashboard (http://localhost:8080) and test the DQN Adaptive Traffic Engineering agent.
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
    print("=" * 85)
    print(" 🚀 STARTING ADAPTIVE SDN TRAFFIC ENGINEERING SIMULATION DRIVER (EC499)")
    print(f" Target Controller Web API: {BASE_URL}")
    print(f" Simulation Mode:           {mode.upper()} (Interval: {interval}s)")
    print("=" * 85)

    step = 0
    scenarios = ["traffic_burst", "core_jamming", "wcmp", "reset"]
    scenario_descriptions = {
        "traffic_burst": "⚡ Pod Ingress Surge (h2->h5, h3->h7) -> DQN Path Diversity",
        "core_jamming":  "🔥 Core Switch Saturation (96% Load) -> Autonomous Lateral Rerouting",
        "wcmp":          "⚖️ Weighted Cost Multi-Path (WCMP) Load Distribution",
        "reset":         "🔄 Nominal Baseline Network Stabilization"
    }

    try:
        while True:
            step += 1
            if max_steps > 0 and step > max_steps:
                print("\n[Done] Reached maximum requested steps.")
                break

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
                stats = http_get("/api/stats")
                ctrl = http_get("/api/control_overhead")
                rl = http_get("/api/rl_metrics")

                switches_count = len(topo.get('nodes', []))
                links_count = len(topo.get('links', []))
                throughput = stats.get('total_throughput_mbps', 0.0)
                bottleneck_u = stats.get('bottleneck_utilization_pct', 0.0)
                latency = stats.get('mean_latency_ms', 0.0)
                jitter = stats.get('mean_jitter_ms', 0.0)
                loss = stats.get('mean_packet_loss_pct', 0.0)
                flows = stats.get('active_flows_count', 0)
                flow_mods = ctrl.get('flow_mod_count', 0)
                dec_time = ctrl.get('mean_decision_latency_ms', 0.0)

                print(f"[Step {step:03d}] {action_desc}")
                print(f"         Switches: {switches_count} | Links: {links_count} | Flows: {flows} | "
                      f"Throughput: {throughput:5.1f} Mbps | Peak Link: {bottleneck_u:4.1f}%")
                print(f"         Latency: {latency:4.1f} ms | Jitter: {jitter:4.2f} ms | Packet Loss: {loss:4.2f}% | "
                      f"FlowMods: {flow_mods} | Decision Latency: {dec_time:.2f} ms")

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
    parser = argparse.ArgumentParser(description="SDN Adaptive Traffic Engineering Live Driver")
    parser.add_argument('--mode', choices=['auto', 'traffic_burst', 'core_jamming', 'wcmp', 'reset'],
                        default='auto', help="Traffic simulation mode")
    parser.add_argument('--interval', type=float, default=3.0, help="Delay between scenario cycles (seconds)")
    parser.add_argument('--steps', type=int, default=0, help="Number of steps to run (0 for continuous)")
    args = parser.parse_args()

    run_simulation(mode=args.mode, interval=args.interval, max_steps=args.steps)
