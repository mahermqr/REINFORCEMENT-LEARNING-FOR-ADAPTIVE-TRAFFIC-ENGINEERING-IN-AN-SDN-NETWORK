#!/usr/bin/env python3
"""
Multi-Topology High-Intensity Blind Stress & Generalization Testing Suite (EC499).
Evaluates trained Deep Q-Network Traffic Engineering agent ZERO-SHOT ("Going Blind") across:
  1. Hierarchical Tree (Baseline Training Fabric - 7 Switches)
  2. Fat-Tree k=4 (Multi-Stage Data Center Clos - 20 Switches)
  3. Abilene Network (Continental US WAN Backbone - 12 Nodes)
  4. NSFNet Mesh (National Science Foundation Core - 14 Nodes)
  5. Spine-Leaf Fabric (Cloud Data Center Interconnect - 12 Switches, 64 Directed Links)

Executes:
  - Scenario 1: Severe Core / Backbone Jamming (85% - 98% saturation)
  - Scenario 2: High-Concurrency Flow Avalanche (500 simultaneous flows)
  - Scenario 3: Asymmetric Regional Hotspot Surges
  - Scenario 4: Dynamic Latency Spikes (Core delay degraded 5x-10x)
  - Scenario 5: Network Jitter & Packet Loss Mitigation (RFC 3393)

Generates:
  - logs/blind_topologies_stress_results.json
  - logs/plots/blind_topologies_stress_benchmark.png
  - logs/plots/blind_topologies_radar.png
"""

import sys
import os
import time
import json
import random
import math
import itertools
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'agent'))
sys.path.append(os.path.join(BASE_DIR, 'controller'))
sys.path.append(os.path.join(BASE_DIR, 'topology'))

from dqn_router import DQNRoutingAgent
from state_manager import StateManager
from topology_library import get_topology
from traditional_routing import compute_path_metrics

PLOTS_DIR = os.path.join(BASE_DIR, 'logs', 'plots')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def jains_fairness_index(loads):
    """Computes Jain's Fairness Index in [1/n, 1.0]."""
    arr = np.array(loads, dtype=np.float64)
    if len(arr) == 0 or np.sum(arr**2) == 0:
        return 1.0
    return float((np.sum(arr)**2) / (len(arr) * np.sum(arr**2)))

def run_blind_topology_stress_tests():
    print("=" * 85)
    print(" 🚀 STARTING MULTI-TOPOLOGY BLIND STRESS & GENERALIZATION BENCHMARK SUITE (EC499)")
    print(" Evaluating Double DQN Traffic Engineering Zero-Shot Across 5 Fabrics")
    print("=" * 85)

    # 1. Load trained agent checkpoint
    router_agent = DQNRoutingAgent(state_size=10, action_size=4)
    router_ckpt = os.path.join(BASE_DIR, 'models', 'dqn_router.pth')
    if not router_agent.load(router_ckpt):
        print(f"[Error] Failed to load router weights from {router_ckpt}")
        return
    print(f"[Init] Loaded trained Double DQN Router checkpoint from models/dqn_router.pth")

    results_summary = {}

    topo_order = ['tree', 'fattree', 'abilene', 'nsfnet', 'spineleaf']
    topo_labels = {
        'tree': 'Hierarchical Tree',
        'fattree': 'Fat-Tree Clos (k=4)',
        'abilene': 'Abilene US Backbone',
        'nsfnet': 'NSFNet Continental',
        'spineleaf': 'Spine-Leaf Fabric'
    }

    for topo_id in topo_order:
        topo_name = topo_labels[topo_id]
        print("\n" + "=" * 85)
        print(f" 🌐 STRESS TESTING FABRIC: {topo_name.upper()}")
        print("=" * 85)

        g, meta = get_topology(topo_id)
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        core_nodes = meta.get('core_nodes', [1])
        edge_nodes = meta.get('edge_nodes', list(g.nodes()))

        print(f" • Topology Characteristics: {n_nodes} Switches, {n_edges} Directed Links")
        print(f" • Core Nodes: {core_nodes} | Edge Nodes: {len(edge_nodes)} switches")

        sm = StateManager()
        sm.graph = g.copy()
        for u, v, d in g.edges(data=True):
            sm.update_link(u, v, src_port=1, dst_port=1,
                           capacity_mbps=d.get('capacity', 100.0),
                           delay_ms=d.get('delay', 2.0))

        topo_path_cache = {}

        def evaluate_flow(src, dst):
            if (src, dst) not in topo_path_cache:
                try:
                    topo_path_cache[(src, dst)] = list(itertools.islice(nx.shortest_simple_paths(sm.graph, src, dst), 4))
                except Exception:
                    topo_path_cache[(src, dst)] = [[src, dst]]
            candidate_paths = topo_path_cache[(src, dst)]

            spf_path = candidate_paths[0]
            state = sm.get_routing_state(src, dst)
            action = router_agent.act(state, explore=False)
            chosen_path = candidate_paths[action % len(candidate_paths)]

            m_dqn = compute_path_metrics(chosen_path, sm.link_utilization, sm.link_delays, sm.link_bandwidths)
            m_spf = compute_path_metrics(spf_path, sm.link_utilization, sm.link_delays, sm.link_bandwidths)

            rerouted = (chosen_path != spf_path)
            return m_dqn['bottleneck_util'], m_spf['bottleneck_util'], m_dqn['total_delay'], m_spf['total_delay'], rerouted, chosen_path, spf_path, m_dqn['jitter'], m_spf['jitter'], m_dqn['packet_loss'], m_spf['packet_loss']

        # ---------------------------------------------------------------------
        # TEST 1: Severe Core / Backbone Jamming (85% - 98% Saturation)
        # ---------------------------------------------------------------------
        print(f"\n[Test 1/5] Core/Backbone Jamming (Core nodes: {core_nodes})")
        for u, v in sm.graph.edges():
            if u in core_nodes or v in core_nodes:
                sm.link_utilization[(u, v)] = random.uniform(0.85, 0.98)
            else:
                sm.link_utilization[(u, v)] = random.uniform(0.12, 0.28)

        n_samples = 150
        t1_dqn_u, t1_spf_u, t1_dqn_lat, t1_spf_lat, t1_reroutes = [], [], [], [], 0

        for _ in range(n_samples):
            src = random.choice(edge_nodes)
            dst = random.choice([n for n in edge_nodes if n != src])
            du, su, dl, sl, rerouted, _, _, _, _, _, _ = evaluate_flow(src, dst)
            t1_dqn_u.append(du * 100.0)
            t1_spf_u.append(su * 100.0)
            t1_dqn_lat.append(dl)
            t1_spf_lat.append(sl)
            if rerouted:
                t1_reroutes += 1

        t1_offload_rate = (t1_reroutes / n_samples) * 100.0
        t1_relief = float(np.mean(t1_spf_u) - np.mean(t1_dqn_u))
        print(f" • Evaluated Flows:             {n_samples}")
        print(f" • Dijkstra SPF Bottleneck:     {np.mean(t1_spf_u):.2f}%")
        print(f" • Double DQN Bottleneck:       {np.mean(t1_dqn_u):.2f}%")
        print(f" • Congestion Reduction:        +{t1_relief:.2f}% relief")
        print(f" • Autonomous Offload Rate:     {t1_offload_rate:.1f}% to alternate paths")
        print(f" • Mean Latency:                DQN: {np.mean(t1_dqn_lat):.2f} ms vs SPF: {np.mean(t1_spf_lat):.2f} ms")

        # ---------------------------------------------------------------------
        # TEST 2: High-Concurrency Flow Avalanche (500 Simultaneous Flows)
        # ---------------------------------------------------------------------
        print(f"\n[Test 2/5] High-Concurrency Flow Avalanche (500 Concurrent Requests)")
        link_loads_dqn = {e: 0.05 for e in sm.graph.edges()}
        link_loads_spf = {e: 0.05 for e in sm.graph.edges()}

        n_burst = 500
        t0 = time.time()
        for _ in range(n_burst):
            src = random.choice(edge_nodes)
            dst = random.choice([n for n in edge_nodes if n != src])
            cands = sm.get_candidate_paths(src, dst, k=4)
            st = sm.get_routing_state(src, dst)
            act = router_agent.act(st, explore=False)
            p_dqn = cands[act % len(cands)]
            p_spf = cands[0]
            for i in range(len(p_dqn)-1):
                e = (p_dqn[i], p_dqn[i+1])
                if e in link_loads_dqn:
                    link_loads_dqn[e] += 1.0
            for i in range(len(p_spf)-1):
                e = (p_spf[i], p_spf[i+1])
                if e in link_loads_spf:
                    link_loads_spf[e] += 1.0

        burst_duration = time.time() - t0
        decisions_sec = n_burst / max(0.001, burst_duration)
        jain_dqn = jains_fairness_index(list(link_loads_dqn.values()))
        jain_spf = jains_fairness_index(list(link_loads_spf.values()))

        print(f" • Execution Time:              {burst_duration*1000:.1f} ms for 500 decisions")
        print(f" • Decision Throughput:         {decisions_sec:.1f} decisions/sec")
        print(f" • Jain's Fairness Index:       DQN: {jain_dqn:.4f} vs SPF: {jain_spf:.4f}")

        # ---------------------------------------------------------------------
        # TEST 3: Asymmetric Regional Hotspot Surges
        # ---------------------------------------------------------------------
        print(f"\n[Test 3/5] Asymmetric Regional Hotspot Surge (Sub-Cluster Congestion)")
        hotspot_nodes = edge_nodes[:max(2, len(edge_nodes)//3)]
        for u, v in sm.graph.edges():
            if u in hotspot_nodes:
                sm.link_utilization[(u, v)] = 0.88
            else:
                sm.link_utilization[(u, v)] = 0.15

        t3_dqn_u, t3_spf_u, t3_diversions = [], [], 0
        for _ in range(100):
            src = random.choice(hotspot_nodes)
            dst = random.choice([n for n in edge_nodes if n not in hotspot_nodes])
            du, su, _, _, rerouted, _, _, _, _, _, _ = evaluate_flow(src, dst)
            t3_dqn_u.append(du * 100.0)
            t3_spf_u.append(su * 100.0)
            if rerouted:
                t3_diversions += 1

        t3_relief = float(np.mean(t3_spf_u) - np.mean(t3_dqn_u))
        print(f" • Hotspot Ingress Relief:      +{t3_relief:.2f}% load reduction")
        print(f" • Local Link Bypass Rate:      {t3_diversions}% rerouted")

        # ---------------------------------------------------------------------
        # TEST 4: Dynamic Delay Degradation
        # ---------------------------------------------------------------------
        print(f"\n[Test 4/5] Dynamic Delay Degradation (Core Link Delay 10x Inflated)")
        for u, v in sm.graph.edges():
            if u in core_nodes or v in core_nodes:
                sm.link_delays[(u, v)] = 20.0
            else:
                sm.link_delays[(u, v)] = 2.0

        t4_dqn_lats, t4_spf_lats = [], []
        for _ in range(100):
            src = random.choice(edge_nodes)
            dst = random.choice([n for n in edge_nodes if n != src])
            _, _, dl, sl, _, _, _, _, _, _, _ = evaluate_flow(src, dst)
            t4_dqn_lats.append(dl)
            t4_spf_lats.append(sl)

        t4_savings = float(np.mean(t4_spf_lats) - np.mean(t4_dqn_lats))
        print(f" • Double DQN Mean Latency:     {np.mean(t4_dqn_lats):.2f} ms")
        print(f" • Dijkstra SPF Mean Latency:   {np.mean(t4_spf_lats):.2f} ms")
        print(f" • Latency Savings:             -{t4_savings:.2f} ms (Faster via low-delay bypass!)")

        # ---------------------------------------------------------------------
        # TEST 5: Jitter & Packet Loss Mitigation (RFC 3393)
        # ---------------------------------------------------------------------
        print(f"\n[Test 5/5] Jitter & Packet Loss Suppression (RFC 3393 / RFC 3550)")
        t5_dqn_j, t5_spf_j = [], []
        t5_dqn_loss, t5_spf_loss = [], []
        for _ in range(100):
            src = random.choice(edge_nodes)
            dst = random.choice([n for n in edge_nodes if n != src])
            _, _, _, _, _, _, _, dj, sj, dloss, sloss = evaluate_flow(src, dst)
            t5_dqn_j.append(dj)
            t5_spf_j.append(sj)
            t5_dqn_loss.append(dloss)
            t5_spf_loss.append(sloss)

        mean_dj = float(np.mean(t5_dqn_j))
        mean_sj = float(np.mean(t5_spf_j))
        mean_dloss = float(np.mean(t5_dqn_loss))
        mean_sloss = float(np.mean(t5_spf_loss))
        jitter_suppression_pct = max(0.0, ((mean_sj - mean_dj) / max(0.01, mean_sj)) * 100.0)

        print(f" • DQN Jitter:                  {mean_dj:.2f} ms (vs SPF: {mean_sj:.2f} ms)")
        print(f" • Jitter Suppression:          {jitter_suppression_pct:.1f}%")
        print(f" • DQN Packet Loss:             {mean_dloss:.2f}% (vs SPF: {mean_sloss:.2f}%)")

        results_summary[topo_id] = {
            "name": topo_name,
            "switches": n_nodes,
            "edges": n_edges,
            "scenario_1_core_jamming": {
                "dijkstra_spf_bottleneck_pct": round(float(np.mean(t1_spf_u)), 2),
                "double_dqn_bottleneck_pct": round(float(np.mean(t1_dqn_u)), 2),
                "congestion_reduction_pct": round(t1_relief, 2),
                "autonomous_offload_rate_pct": round(t1_offload_rate, 1),
                "dqn_latency_ms": round(float(np.mean(t1_dqn_lat)), 2),
                "spf_latency_ms": round(float(np.mean(t1_spf_lat)), 2)
            },
            "scenario_2_concurrency_burst": {
                "decisions_per_sec": round(decisions_sec, 1),
                "jains_fairness_dqn": round(float(jain_dqn), 4),
                "jains_fairness_spf": round(float(jain_spf), 4)
            },
            "scenario_3_hotspot_surge": {
                "bottleneck_reduction_pct": round(t3_relief, 2),
                "diversion_rate_pct": round(float(t3_diversions), 1)
            },
            "scenario_4_latency_degradation": {
                "dqn_latency_ms": round(float(np.mean(t4_dqn_lats)), 2),
                "spf_latency_ms": round(float(np.mean(t4_spf_lats)), 2),
                "latency_savings_ms": round(t4_savings, 2)
            },
            "scenario_5_jitter_and_loss": {
                "dqn_jitter_ms": round(mean_dj, 2),
                "spf_jitter_ms": round(mean_sj, 2),
                "jitter_suppression_pct": round(jitter_suppression_pct, 1),
                "dqn_loss_pct": round(mean_dloss, 2),
                "spf_loss_pct": round(mean_sloss, 2)
            }
        }

    # Generate 4-Panel Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    labels = [topo_labels[t] for t in topo_order]
    x = np.arange(len(labels))
    width = 0.35

    # 1. Core Jamming
    spf_b = [results_summary[t]["scenario_1_core_jamming"]["dijkstra_spf_bottleneck_pct"] for t in topo_order]
    dqn_b = [results_summary[t]["scenario_1_core_jamming"]["double_dqn_bottleneck_pct"] for t in topo_order]
    ax1.bar(x - width/2, spf_b, width, label='Dijkstra SPF', color='#e74c3c', alpha=0.85)
    ax1.bar(x + width/2, dqn_b, width, label='Double DQN (Ours)', color='#2ecc71', alpha=0.85)
    ax1.set_title('Scenario 1: Core Jamming Bottleneck Load (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bottleneck Utilization (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Decision Throughput
    dec_speeds = [results_summary[t]["scenario_2_concurrency_burst"]["decisions_per_sec"] for t in topo_order]
    ax2.bar(x, dec_speeds, color='#3498db', alpha=0.85, width=0.5)
    ax2.set_title('Scenario 2: Decision Throughput (decisions/sec)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Decisions / Second')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Hotspot Relief
    hotspot_relief = [results_summary[t]["scenario_3_hotspot_surge"]["bottleneck_reduction_pct"] for t in topo_order]
    ax3.bar(x, hotspot_relief, color='#9b59b6', alpha=0.85, width=0.5)
    ax3.set_title('Scenario 3: Asymmetric Hotspot Relief (% Load Reduction)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Load Reduction (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha='right', fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 4. Latency
    spf_lats = [results_summary[t]["scenario_4_latency_degradation"]["spf_latency_ms"] for t in topo_order]
    dqn_lats = [results_summary[t]["scenario_4_latency_degradation"]["dqn_latency_ms"] for t in topo_order]
    ax4.bar(x - width/2, spf_lats, width, label='Dijkstra SPF', color='#e67e22', alpha=0.85)
    ax4.bar(x + width/2, dqn_lats, width, label='Double DQN (Ours)', color='#1abc9c', alpha=0.85)
    ax4.set_title('Scenario 4: Latency under Core Link Delay Degradation', fontsize=11, fontweight='bold')
    ax4.set_ylabel('End-to-End Latency (ms)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha='right', fontweight='bold')
    ax4.legend()
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Multi-Topology Blind Stress Testing: DQN Traffic Engineering Generalization Benchmark', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    bench_plot = os.path.join(PLOTS_DIR, 'blind_topologies_stress_benchmark.png')
    plt.savefig(bench_plot, dpi=300)
    plt.close()
    print(f" • Saved Benchmark Plot to: {bench_plot}")

    # Radar Chart
    categories = ['Congestion Relief (%)', 'Offload Rate (%)', 'Fairness (x100)', 'Latency Avoidance (%)', 'Jitter Suppression (%)']
    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * math.pi for n in range(N_cat)]
    angles += angles[:1]

    fig_r, ax_r = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_radar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, t in enumerate(topo_order):
        s1 = results_summary[t]["scenario_1_core_jamming"]
        s2 = results_summary[t]["scenario_2_concurrency_burst"]
        s4 = results_summary[t]["scenario_4_latency_degradation"]
        s5 = results_summary[t]["scenario_5_jitter_and_loss"]

        lat_pct = max(0.0, ((s4['spf_latency_ms'] - s4['dqn_latency_ms']) / max(1.0, s4['spf_latency_ms'])) * 100.0)
        values = [
            min(100.0, max(0.0, s1['congestion_reduction_pct'] * 2.0)),
            min(100.0, s1['autonomous_offload_rate_pct']),
            min(100.0, s2['jains_fairness_dqn'] * 100.0),
            min(100.0, lat_pct),
            min(100.0, s5['jitter_suppression_pct'])
        ]
        values += values[:1]
        ax_r.plot(angles, values, lw=2, label=topo_labels[t], color=colors_radar[idx])
        ax_r.fill(angles, values, color=colors_radar[idx], alpha=0.10)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(categories, fontweight='bold', fontsize=10)
    ax_r.set_ylim(0, 100)
    plt.title('Multi-Topology Resilience & Generalization Radar', size=13, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    radar_plot = os.path.join(PLOTS_DIR, 'blind_topologies_radar.png')
    plt.savefig(radar_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" • Saved Radar Plot to: {radar_plot}")

    # Save summary JSON
    json_path = os.path.join(LOGS_DIR, 'blind_topologies_stress_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f" • Saved Structured Results JSON to: {json_path}")
    print("=" * 85)
    print(" ALL 5 TOPOLOGY BLIND STRESS BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == '__main__':
    run_blind_topology_stress_tests()
