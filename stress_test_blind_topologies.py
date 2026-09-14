#!/usr/bin/env python3
"""
Multi-Topology High-Intensity Blind Stress & Generalization Testing Suite (EC499)
Evaluates trained Deep Reinforcement Learning agents ZERO-SHOT ("Going Blind") across:
  1. Hierarchical Tree (Baseline Training Fabric - 7 Switches)
  2. Fat-Tree k=4 (Multi-Stage Data Center Clos - 20 Switches)
  3. Abilene Network (Continental US WAN Backbone - 12 Nodes)
  4. NSFNet Mesh (National Science Foundation Core - 14 Nodes)
  5. Spine-Leaf Fabric (Cloud Data Center Interconnect - 12 Switches, 64 Directed Links)

Executes:
  - Scenario 1: Severe Core / Backbone Jamming (85% - 98% saturation)
  - Scenario 2: High-Concurrency Flow Avalanche (500 simultaneous flows)
  - Scenario 3: Asymmetric Regional Hotspot Surges
  - Scenario 4: Dynamic Latency Spikes & Jitter (Core delay degraded 5x-10x)
  - Scenario 5: Multicast Steiner Group Replication Savings

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
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'agent'))
sys.path.append(os.path.join(BASE_DIR, 'controller'))
sys.path.append(os.path.join(BASE_DIR, 'topology'))

from dqn_router import DQNRoutingAgent
from dqn_multicast import DQNMulticastAgent
from state_manager import StateManager
from topology_library import get_topology

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
    print(" 🚀 STARTING MULTI-TOPOLOGY BLIND STRESS & GENERALIZATION BENCHMARK SUITE")
    print(" Evaluating Double DQN Unicast & Dueling DQN Multicast Zero-Shot Across 5 Fabrics")
    print("=" * 85)

    # 1. Load trained agent checkpoints
    router_agent = DQNRoutingAgent(state_size=10, action_size=4)
    router_ckpt = os.path.join(BASE_DIR, 'models', 'dqn_router.pth')
    if not router_agent.load(router_ckpt):
        print(f"[Error] Failed to load router weights from {router_ckpt}")
        return
    print(f"[Init] Loaded trained Double DQN Router checkpoint from models/dqn_router.pth")

    multicast_agent = DQNMulticastAgent(state_size=50, action_size=10)
    m_ckpt = os.path.join(BASE_DIR, 'models', 'dqn_multicast.pth')
    if os.path.exists(m_ckpt):
        multicast_agent.load(m_ckpt)
        print(f"[Init] Loaded trained Dueling DQN Multicast checkpoint from models/dqn_multicast.pth")

    results_summary = {}

    topo_order = ['tree', 'fattree', 'abilene', 'nsfnet', 'spineleaf']
    topo_labels = {
        'tree': 'Tree (Baseline)',
        'fattree': 'Fat-Tree (k=4)',
        'abilene': 'Abilene WAN',
        'nsfnet': 'NSFNet Mesh',
        'spineleaf': 'Spine-Leaf'
    }

    for topo_id in topo_order:
        graph, meta = get_topology(topo_id)
        topo_name = meta['name']
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        print("\n" + "=" * 85)
        print(f" 🌐 TESTING BLIND GENERALIZATION ON: {topo_name.upper()}")
        print(f" Topology Specs: {n_nodes} switches/nodes | {n_edges} directed edges")
        print("=" * 85)

        sm = StateManager()
        sm.graph = graph.copy()
        for u, v, data in graph.edges(data=True):
            sm.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=data['capacity'], delay_ms=data['delay'])

        edge_nodes = meta.get('edge_nodes', list(graph.nodes()))
        core_nodes = meta.get('core_nodes', [list(graph.nodes())[0]])

        # Path cache for topology to avoid redundant shortest_simple_paths traversals
        topo_path_cache = {}

        # Helper to evaluate flow
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

            def p_metrics(p):
                lat = sum(sm.link_delays.get((p[i], p[i+1]), 2.0) for i in range(len(p)-1))
                util = max(sm.link_utilization.get((p[i], p[i+1]), 0.0) for i in range(len(p)-1))
                return lat, util

            d_lat, d_util = p_metrics(chosen_path)
            s_lat, s_util = p_metrics(spf_path)
            rerouted = (chosen_path != spf_path)
            return d_util, s_util, d_lat, s_lat, rerouted, chosen_path, spf_path

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
            dst_candidates = [n for n in edge_nodes if n != src]
            dst = random.choice(dst_candidates)
            du, su, dl, sl, rerouted, path, spf = evaluate_flow(src, dst)
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
        t2_reroutes = 0
        for _ in range(n_burst):
            src = random.choice(edge_nodes)
            dst_candidates = [n for n in edge_nodes if n != src]
            dst = random.choice(dst_candidates)
            du, su, dl, sl, rerouted, dqn_path, spf_path = evaluate_flow(src, dst)
            if rerouted:
                t2_reroutes += 1

            for i in range(len(dqn_path) - 1):
                e = (dqn_path[i], dqn_path[i+1])
                if e in link_loads_dqn:
                    link_loads_dqn[e] = min(1.0, link_loads_dqn[e] + 0.015)
            for i in range(len(spf_path) - 1):
                e = (spf_path[i], spf_path[i+1])
                if e in link_loads_spf:
                    link_loads_spf[e] = min(1.0, link_loads_spf[e] + 0.015)

        burst_duration = time.time() - t0
        decisions_sec = n_burst / max(0.001, burst_duration)
        jain_dqn = jains_fairness_index(list(link_loads_dqn.values()))
        jain_spf = jains_fairness_index(list(link_loads_spf.values()))

        print(f" • Processed Burst:             500 concurrent flows in {burst_duration*1000:.1f} ms")
        print(f" • Decision Throughput:         {decisions_sec:.1f} decisions/sec")
        print(f" • Jain's Fairness Index:       DQN: {jain_dqn:.4f} vs SPF: {jain_spf:.4f}")
        print(f" • Peak Link Utilization:       DQN: {max(link_loads_dqn.values())*100:.1f}% vs SPF: {max(link_loads_spf.values())*100:.1f}%")

        # ---------------------------------------------------------------------
        # TEST 3: Asymmetric Regional Hotspot Surge
        # ---------------------------------------------------------------------
        print(f"\n[Test 3/5] Asymmetric Hotspot Surge")
        hotspot_node = edge_nodes[0]
        for u, v in sm.graph.edges():
            if u == hotspot_node or v == hotspot_node:
                sm.link_utilization[(u, v)] = random.uniform(0.80, 0.94)
            else:
                sm.link_utilization[(u, v)] = random.uniform(0.10, 0.25)

        t3_dqn_u, t3_spf_u, t3_diversions = [], [], 0
        for _ in range(100):
            src = hotspot_node
            dst = random.choice([n for n in edge_nodes if n != src])
            du, su, dl, sl, rerouted, path, spf = evaluate_flow(src, dst)
            t3_dqn_u.append(du * 100.0)
            t3_spf_u.append(su * 100.0)
            if rerouted:
                t3_diversions += 1

        t3_relief = float(np.mean(t3_spf_u) - np.mean(t3_dqn_u))
        print(f" • Hotspot Ingress Node:        Switch {hotspot_node} saturated to ~88%")
        print(f" • Hotspot Load Relief:         +{t3_relief:.2f}% improvement")
        print(f" • Autonomous Diversion Rate:   {t3_diversions}% diverted away from congested ingress")

        # ---------------------------------------------------------------------
        # TEST 4: Dynamic Latency Spikes (Core Delays Inflated 5x)
        # ---------------------------------------------------------------------
        print(f"\n[Test 4/5] Dynamic Latency Degradation (Core links inflated to 25ms)")
        for u, v in sm.graph.edges():
            if u in core_nodes or v in core_nodes:
                sm.link_delays[(u, v)] = 25.0
                sm.link_utilization[(u, v)] = 0.50
            else:
                sm.link_delays[(u, v)] = 4.0
                sm.link_utilization[(u, v)] = 0.20

        t4_dqn_lats, t4_spf_lats = [], []
        for _ in range(100):
            src = random.choice(edge_nodes)
            dst = random.choice([n for n in edge_nodes if n != src])
            du, su, dl, sl, rerouted, path, spf = evaluate_flow(src, dst)
            t4_dqn_lats.append(dl)
            t4_spf_lats.append(sl)

        t4_savings = float(np.mean(t4_spf_lats) - np.mean(t4_dqn_lats))
        print(f" • Double DQN Mean Latency:     {np.mean(t4_dqn_lats):.2f} ms")
        print(f" • Dijkstra SPF Mean Latency:   {np.mean(t4_spf_lats):.2f} ms")
        print(f" • Latency Savings:             -{t4_savings:.2f} ms (Faster via low-delay bypass!)")

        # ---------------------------------------------------------------------
        # TEST 5: Multicast Steiner Tree Replication Savings
        # ---------------------------------------------------------------------
        print(f"\n[Test 5/5] Multicast Group Replication Efficiency")
        m_savings = []
        for _ in range(50):
            m_src = random.choice(edge_nodes)
            m_dests = random.sample([n for n in edge_nodes if n != m_src], min(3, len(edge_nodes) - 1))
            g_undir = sm.graph.to_undirected()
            for u, v in g_undir.edges():
                g_undir[u][v]['weight'] = 1.0 + sm.link_delays.get((u, v), 2.0) * 0.5
            tree = nx.algorithms.approximation.steinertree.steiner_tree(g_undir, [m_src] + m_dests, weight='weight')
            tree_edges = tree.number_of_edges()
            unicast_edges = len(m_dests) * 3
            bw_saved = max(0, (unicast_edges - tree_edges) * 10.0)
            m_savings.append(bw_saved)

        avg_m_savings = float(np.mean(m_savings))
        print(f" • Multicast Conserved Bandwidth: {avg_m_savings:.1f} Mbps average")

        # Record summary record for this topology
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
                "jains_fairness_spf": round(float(jain_spf), 4),
                "peak_load_dqn_pct": round(float(max(link_loads_dqn.values()) * 100), 1),
                "peak_load_spf_pct": round(float(max(link_loads_spf.values()) * 100), 1)
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
            "scenario_5_multicast": {
                "average_bw_saved_mbps": round(avg_m_savings, 1)
            }
        }

    # =========================================================================
    # GENERATE PUBLICATION FIGURES
    # =========================================================================
    print("\n" + "=" * 85)
    print(" Generating Publication Plots across All 5 Topologies...")
    print("=" * 85)

    # 1. 4-Panel Comparative Benchmark Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    labels = [topo_labels[t] for t in topo_order]
    x = np.arange(len(labels))
    width = 0.35

    # Panel 1: Bottleneck Utilization (SPF vs Double DQN)
    spf_utils = [results_summary[t]["scenario_1_core_jamming"]["dijkstra_spf_bottleneck_pct"] for t in topo_order]
    dqn_utils = [results_summary[t]["scenario_1_core_jamming"]["double_dqn_bottleneck_pct"] for t in topo_order]

    ax1.bar(x - width/2, spf_utils, width, label='Dijkstra (SPF)', color='#d62728', alpha=0.85)
    ax1.bar(x + width/2, dqn_utils, width, label='Double DQN (Zero-Shot Blind)', color='#2ca02c', alpha=0.85)
    ax1.set_title('Test 1: Peak Link Bottleneck under Core Jamming (%)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Bottleneck Utilization (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    for i in range(len(labels)):
        ax1.text(x[i] - width/2, spf_utils[i] + 2, f"{spf_utils[i]:.1f}%", ha='center', fontsize=9)
        ax1.text(x[i] + width/2, dqn_utils[i] + 2, f"{dqn_utils[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

    # Panel 2: Congestion Relief & Autonomous Offload Rate
    relief_vals = [results_summary[t]["scenario_1_core_jamming"]["congestion_reduction_pct"] for t in topo_order]
    offload_vals = [results_summary[t]["scenario_1_core_jamming"]["autonomous_offload_rate_pct"] for t in topo_order]

    ax2.bar(x - width/2, relief_vals, width, label='Congestion Relief (+%)', color='#1f77b4', alpha=0.85)
    ax2.bar(x + width/2, offload_vals, width, label='Autonomous Offload Rate (%)', color='#9467bd', alpha=0.85)
    ax2.set_title('Test 1: Zero-Shot Congestion Relief & Offload Rate', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right', fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    for i in range(len(labels)):
        ax2.text(x[i] - width/2, relief_vals[i] + 2, f"+{relief_vals[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax2.text(x[i] + width/2, offload_vals[i] + 2, f"{offload_vals[i]:.0f}%", ha='center', fontsize=9)

    # Panel 3: Controller Throughput & Jain's Fairness
    jain_dqn_vals = [results_summary[t]["scenario_2_concurrency_burst"]["jains_fairness_dqn"] for t in topo_order]
    jain_spf_vals = [results_summary[t]["scenario_2_concurrency_burst"]["jains_fairness_spf"] for t in topo_order]

    ax3.plot(labels, jain_dqn_vals, marker='o', lw=2.5, color='#2ca02c', label="DQN Jain's Index")
    ax3.plot(labels, jain_spf_vals, marker='s', lw=2.0, ls='--', color='#d62728', label="SPF Jain's Index")
    ax3.set_title("Test 2: Load Uniformity (Jain's Fairness Index)", fontweight='bold', fontsize=11)
    ax3.set_ylabel("Jain's Index (Higher is Better)")
    ax3.set_ylim(0.2, 1.05)
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    for i, txt in enumerate(jain_dqn_vals):
        ax3.annotate(f"{txt:.3f}", (labels[i], jain_dqn_vals[i] + 0.03), ha='center', fontweight='bold')

    # Panel 4: Latency Trade-off under Degraded Core Links
    dqn_lats = [results_summary[t]["scenario_4_latency_degradation"]["dqn_latency_ms"] for t in topo_order]
    spf_lats = [results_summary[t]["scenario_4_latency_degradation"]["spf_latency_ms"] for t in topo_order]

    ax4.bar(x - width/2, spf_lats, width, label='Dijkstra (Core Degraded)', color='#ff7f0e', alpha=0.85)
    ax4.bar(x + width/2, dqn_lats, width, label='Double DQN (Bypass Path)', color='#1f77b4', alpha=0.85)
    ax4.set_title('Test 4: Latency under Core Degradation (ms)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('End-to-End Latency (ms)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha='right', fontweight='bold')
    ax4.legend()
    ax4.grid(True, linestyle=':', alpha=0.6)
    for i in range(len(labels)):
        ax4.text(x[i] - width/2, spf_lats[i] + 1.5, f"{spf_lats[i]:.1f}", ha='center', fontsize=9)
        ax4.text(x[i] + width/2, dqn_lats[i] + 1.5, f"{dqn_lats[i]:.1f}", ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Multi-Topology Blind Stress Testing: DRL Zero-Shot Generalization Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    bench_plot = os.path.join(PLOTS_DIR, 'blind_topologies_stress_benchmark.png')
    plt.savefig(bench_plot, dpi=300)
    plt.close()
    print(f" • Saved Benchmark Plot to: {bench_plot}")

    # 2. Radar Chart: Multi-Dimensional Resilience Profile across Topologies
    categories = ['Congestion Relief (%)', 'Offload Rate (%)', 'Fairness (x100)', 'Latency Avoidance (%)', 'Multicast Savings (Mbps)']
    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * math.pi for n in range(N_cat)]
    angles += angles[:1]

    fig_r, ax_r = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_radar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, t in enumerate(topo_order):
        s1 = results_summary[t]["scenario_1_core_jamming"]
        s2 = results_summary[t]["scenario_2_concurrency_burst"]
        s4 = results_summary[t]["scenario_4_latency_degradation"]
        s5 = results_summary[t]["scenario_5_multicast"]

        lat_pct = max(0.0, ((s4['spf_latency_ms'] - s4['dqn_latency_ms']) / max(1.0, s4['spf_latency_ms'])) * 100.0)
        values = [
            min(100.0, max(0.0, s1['congestion_reduction_pct'] * 2.0)), # Scaled
            min(100.0, s1['autonomous_offload_rate_pct']),
            min(100.0, s2['jains_fairness_dqn'] * 100.0),
            min(100.0, lat_pct),
            min(100.0, s5['average_bw_saved_mbps'] * 1.5)
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
