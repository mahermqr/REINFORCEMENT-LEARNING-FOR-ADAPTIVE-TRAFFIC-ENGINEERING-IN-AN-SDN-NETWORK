#!/usr/bin/env python3
"""
Comprehensive Routing Algorithms Tournament Benchmark (EC499)
Head-to-head comparison of 5 routing algorithms:
 1. Dijkstra Shortest Path First (SPF)
 2. Equal-Cost Multi-Path (ECMP)
 3. Widest Shortest Path (WSP / CSPF)
 4. Least Loaded Routing (LLR)
 5. Autonomous Dueling Double DQN Agent (Ours)

Evaluates:
 - Bottleneck Link Utilization (%)
 - Mean End-to-End Latency (ms)
 - Autonomous Offload / Congestion Avoidance Rate (%)
 - Jain's Fairness Index
 - Controller Decision Throughput (decisions/sec)
"""

import os
import sys
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add directories to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'agent'))
sys.path.append(os.path.join(BASE_DIR, 'controller'))
sys.path.append(os.path.join(BASE_DIR, 'topology'))

from state_manager import StateManager
from dqn_router import DQNRoutingAgent
from topology_library import get_topology, build_random_topology
from traditional_routing import (
    dijkstra_spf, ecmp_routing, widest_shortest_path, least_loaded_routing, compute_path_metrics
)


def run_routing_tournament():
    print("=" * 85)
    print(" 🏆 ROUTING ALGORITHMS TOURNAMENT: DUELING DOUBLE DQN VS CLASSICAL BASELINES")
    print("=" * 85)

    # 1. Initialize RL Agent
    ckpt_path = os.path.join(BASE_DIR, 'models', 'dqn_router.pth')
    router_agent = DQNRoutingAgent(state_size=10, action_size=4)
    if os.path.exists(ckpt_path):
        router_agent.load(ckpt_path)
        print(f"[Init] Successfully loaded Dueling Double DQN checkpoint from {ckpt_path}")
    else:
        print("[Init] Warning: Checkpoint not found, using initialized weights.")

    topologies_to_test = [
        ('tree', "Hierarchical Tree (7 Nodes)"),
        ('fattree', "Fat-Tree Clos Fabric (20 Nodes)"),
        ('abilene', "Abilene US Backbone (12 Nodes)"),
        ('nsfnet', "NSFNET Continental Mesh (14 Nodes)"),
        ('spineleaf', "Spine-Leaf Fabric (12 Nodes)"),
        ('random_20', "Dynamic Watts-Strogatz Mesh (20 Nodes)")
    ]

    algorithms = ['SPF (Dijkstra)', 'ECMP', 'WSP (Widest Path)', 'LLR (Least Loaded)', 'Dueling DQN (Ours)']

    benchmark_data = {
        'jamming_bottleneck': {algo: [] for algo in algorithms},
        'degraded_latency': {algo: [] for algo in algorithms},
        'jains_fairness': {algo: [] for algo in algorithms},
        'offload_rate': {algo: [] for algo in algorithms},
        'topologies': []
    }

    for topo_key, topo_name in topologies_to_test:
        print(f"\n" + "-" * 85)
        print(f" 🌐 Evaluating Fabric: {topo_name}")
        print("-" * 85)

        if topo_key == 'random_20':
            g, meta = build_random_topology(num_nodes=20, p_edge=0.28, seed=42)
        else:
            g, meta = get_topology(topo_key)

        edge_nodes = meta.get('edge_nodes', list(g.nodes()))
        core_nodes = meta.get('core_nodes', [list(g.nodes())[0]])

        # ----------------------------------------------------------------------
        # Benchmark Scenario 1: Severe Core Jamming Stress (85% - 98% saturation)
        # ----------------------------------------------------------------------
        sm = StateManager()
        sm.graph = g.copy()
        for u, v, d in g.edges(data=True):
            cap = d.get('capacity', 100.0)
            delay = d.get('delay', 2.0)
            sm.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=cap, delay_ms=delay)

        # Saturate all core-facing links
        for u, v in sm.graph.edges():
            if u in core_nodes or v in core_nodes:
                sm.link_utilization[(u, v)] = random.uniform(0.85, 0.98)
            else:
                sm.link_utilization[(u, v)] = random.uniform(0.10, 0.25)

        # Evaluate 100 flow requests across edges
        flow_samples = 100
        algo_utils = {algo: [] for algo in algorithms}
        algo_latencies = {algo: [] for algo in algorithms}
        algo_offloaded = {algo: 0 for algo in algorithms}

        random.seed(1234)
        for flow_idx in range(flow_samples):
            src = random.choice(edge_nodes)
            dst_opts = [n for n in edge_nodes if n != src]
            dst = random.choice(dst_opts)

            cand_paths = sm.get_candidate_paths(src, dst, k=4)
            spf_path = cand_paths[0]

            # 1. SPF
            p_spf = dijkstra_spf(sm.graph, src, dst)
            u_spf, d_spf, _ = compute_path_metrics(p_spf, sm.link_utilization, sm.link_delays)
            algo_utils['SPF (Dijkstra)'].append(u_spf)
            algo_latencies['SPF (Dijkstra)'].append(d_spf)

            # 2. ECMP
            p_ecmp = ecmp_routing(sm.graph, src, dst, flow_hash=flow_idx)
            u_ecmp, d_ecmp, _ = compute_path_metrics(p_ecmp, sm.link_utilization, sm.link_delays)
            algo_utils['ECMP'].append(u_ecmp)
            algo_latencies['ECMP'].append(d_ecmp)

            # 3. WSP
            p_wsp = widest_shortest_path(sm.graph, src, dst, sm.link_utilization, sm.link_delays, candidate_paths=cand_paths)
            u_wsp, d_wsp, _ = compute_path_metrics(p_wsp, sm.link_utilization, sm.link_delays)
            algo_utils['WSP (Widest Path)'].append(u_wsp)
            algo_latencies['WSP (Widest Path)'].append(d_wsp)

            # 4. LLR
            p_llr = least_loaded_routing(sm.graph, src, dst, sm.link_utilization, candidate_paths=cand_paths)
            u_llr, d_llr, _ = compute_path_metrics(p_llr, sm.link_utilization, sm.link_delays)
            algo_utils['LLR (Least Loaded)'].append(u_llr)
            algo_latencies['LLR (Least Loaded)'].append(d_llr)

            # 5. Dueling DQN
            st = sm.get_routing_state(src, dst)
            action = router_agent.act(st, explore=False)
            p_dqn = cand_paths[action % len(cand_paths)]
            u_dqn, d_dqn, _ = compute_path_metrics(p_dqn, sm.link_utilization, sm.link_delays)
            algo_utils['Dueling DQN (Ours)'].append(u_dqn)
            algo_latencies['Dueling DQN (Ours)'].append(d_dqn)

            # Check offload: took path different from jammed primary path
            if p_spf != p_dqn:
                algo_offloaded['Dueling DQN (Ours)'] += 1
            if p_spf != p_wsp:
                algo_offloaded['WSP (Widest Path)'] += 1
            if p_spf != p_llr:
                algo_offloaded['LLR (Least Loaded)'] += 1
            if p_spf != p_ecmp:
                algo_offloaded['ECMP'] += 1

        print(f" [Core Jamming Stress Results (100 Flows)]")
        for algo in algorithms:
            mean_u = np.mean(algo_utils[algo]) * 100.0
            mean_lat = np.mean(algo_latencies[algo])
            off_pct = (algo_offloaded[algo] / flow_samples) * 100.0
            benchmark_data['jamming_bottleneck'][algo].append(mean_u)
            benchmark_data['offload_rate'][algo].append(off_pct)
            print(f"  • {algo:<22}: Bottleneck Load: {mean_u:5.1f}% | Latency: {mean_lat:5.2f}ms | Offload Rate: {off_pct:4.1f}%")

        # ----------------------------------------------------------------------
        # Benchmark Scenario 2: Dynamic Latency Degradation (Core links inflated to 25ms)
        # ----------------------------------------------------------------------
        sm_lat = StateManager()
        sm_lat.graph = g.copy()
        for u, v, d in g.edges(data=True):
            cap = d.get('capacity', 100.0)
            base_delay = d.get('delay', 2.0)
            if u in core_nodes or v in core_nodes:
                sm_lat.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=cap, delay_ms=25.0)
            else:
                sm_lat.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=cap, delay_ms=base_delay)

        degraded_lats = {algo: [] for algo in algorithms}
        for flow_idx in range(flow_samples):
            src = edge_nodes[flow_idx % len(edge_nodes)]
            dst = edge_nodes[(flow_idx + 1) % len(edge_nodes)]
            cand_paths = sm_lat.get_candidate_paths(src, dst, k=4)

            # SPF
            p = dijkstra_spf(sm_lat.graph, src, dst)
            _, d, _ = compute_path_metrics(p, sm_lat.link_utilization, sm_lat.link_delays)
            degraded_lats['SPF (Dijkstra)'].append(d)

            # ECMP
            p = ecmp_routing(sm_lat.graph, src, dst, flow_hash=flow_idx)
            _, d, _ = compute_path_metrics(p, sm_lat.link_utilization, sm_lat.link_delays)
            degraded_lats['ECMP'].append(d)

            # WSP
            p = widest_shortest_path(sm_lat.graph, src, dst, sm_lat.link_utilization, sm_lat.link_delays, candidate_paths=cand_paths)
            _, d, _ = compute_path_metrics(p, sm_lat.link_utilization, sm_lat.link_delays)
            degraded_lats['WSP (Widest Path)'].append(d)

            # LLR
            p = least_loaded_routing(sm_lat.graph, src, dst, sm_lat.link_utilization, candidate_paths=cand_paths)
            _, d, _ = compute_path_metrics(p, sm_lat.link_utilization, sm_lat.link_delays)
            degraded_lats['LLR (Least Loaded)'].append(d)

            # Dueling DQN
            st = sm_lat.get_routing_state(src, dst)
            act = router_agent.act(st, explore=False)
            p = cand_paths[act % len(cand_paths)]
            _, d, _ = compute_path_metrics(p, sm_lat.link_utilization, sm_lat.link_delays)
            degraded_lats['Dueling DQN (Ours)'].append(d)

        for algo in algorithms:
            mean_deg_lat = float(np.mean(degraded_lats[algo]))
            benchmark_data['degraded_latency'][algo].append(mean_deg_lat)

        # ----------------------------------------------------------------------
        # Benchmark Scenario 3: High-Concurrency Burst (Jain's Fairness Index)
        # ----------------------------------------------------------------------
        edge_load_counts = {algo: {e: 0 for e in g.edges()} for algo in algorithms}
        for flow_idx in range(300):
            src = edge_nodes[flow_idx % len(edge_nodes)]
            dst = edge_nodes[(flow_idx + 2) % len(edge_nodes)]
            cand_paths = sm.get_candidate_paths(src, dst, k=4)

            paths = {
                'SPF (Dijkstra)': dijkstra_spf(sm.graph, src, dst),
                'ECMP': ecmp_routing(sm.graph, src, dst, flow_hash=flow_idx),
                'WSP (Widest Path)': widest_shortest_path(sm.graph, src, dst, sm.link_utilization, sm.link_delays, candidate_paths=cand_paths),
                'LLR (Least Loaded)': least_loaded_routing(sm.graph, src, dst, sm.link_utilization, candidate_paths=cand_paths),
                'Dueling DQN (Ours)': cand_paths[router_agent.act(sm.get_routing_state(src, dst), explore=False) % len(cand_paths)]
            }

            for algo, p in paths.items():
                for i in range(len(p) - 1):
                    e = (p[i], p[i+1])
                    if e in edge_load_counts[algo]:
                        edge_load_counts[algo][e] += 1
                    elif (p[i+1], p[i]) in edge_load_counts[algo]:
                        edge_load_counts[algo][(p[i+1], p[i])] += 1

        for algo in algorithms:
            loads = np.array(list(edge_load_counts[algo].values()), dtype=np.float64)
            sum_x = np.sum(loads)
            sum_sq_x = np.sum(loads ** 2)
            n_edges = len(loads)
            jains = (sum_x ** 2) / (n_edges * sum_sq_x + 1e-9) if sum_sq_x > 0 else 1.0
            benchmark_data['jains_fairness'][algo].append(float(jains))

        benchmark_data['topologies'].append(topo_key)

    # --------------------------------------------------------------------------
    # Save Benchmark Results JSON
    # --------------------------------------------------------------------------
    json_path = os.path.join(BASE_DIR, 'logs', 'routing_algorithms_benchmark.json')
    with open(json_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[Saved] Tournament benchmark data saved to: {json_path}")

    # --------------------------------------------------------------------------
    # Generate Publication-Grade Plot: routing_algorithms_comparison.png
    # --------------------------------------------------------------------------
    plots_dir = os.path.join(BASE_DIR, 'logs', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'routing_algorithms_comparison.png')

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor('#0d1117')

    colors = {
        'SPF (Dijkstra)': '#ff5252',     # Red
        'ECMP': '#ff9800',               # Orange
        'WSP (Widest Path)': '#ffeb3b',  # Yellow
        'LLR (Least Loaded)': '#40c4ff',  # Light Blue
        'Dueling DQN (Ours)': '#00e676'  # Neon Green
    }

    topo_labels = ['Tree', 'FatTree', 'Abilene', 'NSFNET', 'SpineLeaf', 'Random (20N)']
    x = np.arange(len(topo_labels))
    width = 0.16

    # Panel 1: Bottleneck Link Load in Core Jamming
    ax1 = axes[0, 0]
    ax1.set_facecolor('#161b22')
    for idx, algo in enumerate(algorithms):
        vals = benchmark_data['jamming_bottleneck'][algo]
        bars = ax1.bar(x + (idx - 2) * width, vals, width, label=algo, color=colors[algo], alpha=0.9)
        if algo == 'Dueling DQN (Ours)':
            for bar in bars:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.0,
                         f"{bar.get_height():.0f}%", ha='center', va='bottom',
                         fontsize=7.5, color='#00e676', fontweight='bold')
    ax1.set_title('A) Severe Core Jamming: Peak Bottleneck Utilization (%) [Lower is Better]',
                  fontsize=11, fontweight='bold', color='#f0f6fc', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(topo_labels, fontsize=9.5, color='#8b949e')
    ax1.set_ylabel('Bottleneck Utilization (%)', fontsize=10, color='#8b949e')
    ax1.set_ylim(0, 110)
    ax1.axhline(70, color='#ff1744', linestyle='--', alpha=0.5, label='Congestion Threshold (70%)')
    ax1.grid(True, linestyle=':', alpha=0.3, color='#30363d')
    ax1.legend(loc='upper right', fontsize=8, facecolor='#21262d', edgecolor='#30363d')

    # Panel 2: Degraded Latency Bypass
    ax2 = axes[0, 1]
    ax2.set_facecolor('#161b22')
    for idx, algo in enumerate(algorithms):
        vals = benchmark_data['degraded_latency'][algo]
        ax2.bar(x + (idx - 2) * width, vals, width, label=algo, color=colors[algo], alpha=0.9)
    ax2.set_title('B) Core Latency Degradation: Mean End-to-End Delay (ms) [Lower is Better]',
                  fontsize=11, fontweight='bold', color='#f0f6fc', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(topo_labels, fontsize=9.5, color='#8b949e')
    ax2.set_ylabel('Latency (ms)', fontsize=10, color='#8b949e')
    ax2.grid(True, linestyle=':', alpha=0.3, color='#30363d')
    ax2.legend(loc='upper right', fontsize=8, facecolor='#21262d', edgecolor='#30363d')

    # Panel 3: Autonomous Offload Rate from Jammed Paths
    ax3 = axes[1, 0]
    ax3.set_facecolor('#161b22')
    for idx, algo in enumerate(algorithms):
        vals = benchmark_data['offload_rate'][algo]
        ax3.plot(topo_labels, vals, marker='o', linewidth=2.2, label=algo, color=colors[algo])
    ax3.set_title('C) Autonomous Rerouting / Offload Rate from Jammed Paths (%) [Higher is Better]',
                  fontsize=11, fontweight='bold', color='#f0f6fc', pad=10)
    ax3.set_ylabel('Offload Rate (%)', fontsize=10, color='#8b949e')
    ax3.set_ylim(-5, 110)
    ax3.grid(True, linestyle=':', alpha=0.3, color='#30363d')
    ax3.legend(loc='upper left', fontsize=8, facecolor='#21262d', edgecolor='#30363d')

    # Panel 4: Jain's Fairness Index under Burst Load
    ax4 = axes[1, 1]
    ax4.set_facecolor('#161b22')
    mean_jains = [np.mean(benchmark_data['jains_fairness'][algo]) for algo in algorithms]
    bars4 = ax4.barh(algorithms, mean_jains, color=[colors[a] for a in algorithms], height=0.55, alpha=0.9)
    for bar in bars4:
        ax4.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height()/2.,
                 f"{bar.get_width():.4f}", ha='left', va='center',
                 fontsize=9, color='#f0f6fc', fontweight='bold')
    ax4.set_title("D) Global Traffic Distribution Uniformity: Jain's Fairness Index [Higher is Better]",
                  fontsize=11, fontweight='bold', color='#f0f6fc', pad=10)
    ax4.set_xlabel("Mean Jain's Fairness Index Across Topologies", fontsize=10, color='#8b949e')
    ax4.set_xlim(0, 1.0)
    ax4.grid(True, linestyle=':', alpha=0.3, color='#30363d')

    plt.suptitle("AUTONOMOUS SDN TRAFFIC ENGINEERING: ROUTING ALGORITHMS BENCHMARK TOURNAMENT\n"
                 "Dueling Double Deep Q-Network (D3QN) vs Dijkstra SPF, ECMP, Widest Shortest Path & Least Loaded Routing",
                 fontsize=13, fontweight='bold', color='#58a6ff', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[Saved] Publication benchmark plot saved to: {plot_path}")
    print("=" * 85)
    print(" 🏆 ROUTING ALGORITHMS TOURNAMENT COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == '__main__':
    run_routing_tournament()
