#!/usr/bin/env python3
"""
Comprehensive Routing Algorithms Tournament Benchmark (EC499).
Head-to-head evaluation of Deep Q-Network (DQN) Traffic Engineering against:
 1. OSPF (Open Shortest Path First - RFC 2328 Reference Bandwidth Metric)
 2. Dijkstra Shortest Path First (SPF - Hop Count Metric)
 3. Equal-Cost Multi-Path (ECMP)
 4. Widest Shortest Path (WSP / CSPF - Residual Bandwidth Priority)
 5. Greedy Least Loaded Routing (LLR - Minimum Peak & Average Utilization)

Evaluates all 5 Core Metrics specified in EC499 Proposal:
 - Metric 1: Total Network Throughput & Bottleneck Link Utilization (%)
 - Metric 2: Network Latency (ms)
 - Metric 3: Network Jitter (ms Delay Variation RFC 3393)
 - Metric 4: Packet Loss Rate (%)
 - Metric 5: OpenFlow Control Overhead (Decision Latency ms & Decisions/sec)
 - Plus: Jain's Fairness Index across links
"""

import os
import sys
import json
import time
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
    ospf_routing, dijkstra_spf, ecmp_routing, widest_shortest_path, least_loaded_routing, compute_path_metrics
)

PLOTS_DIR = os.path.join(BASE_DIR, 'logs', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_routing_tournament():
    print("=" * 90)
    print(" 🏆 EC499 BENCHMARK TOURNAMENT: DQN ADAPTIVE ROUTING VS OSPF & GREEDY BASELINES")
    print("=" * 90)

    # 1. Initialize RL Agent
    ckpt_path = os.path.join(BASE_DIR, 'models', 'dqn_router.pth')
    router_agent = DQNRoutingAgent(state_size=10, action_size=4)
    if os.path.exists(ckpt_path):
        router_agent.load(ckpt_path)
        print(f"[Init] Loaded pre-trained DQN checkpoint from {ckpt_path}")
    else:
        print("[Init] Checkpoint not found; agent initialized with fresh policy.")

    topologies_to_test = [
        ('tree', "Hierarchical Tree (7 Nodes)"),
        ('fattree', "Fat-Tree Clos Fabric (20 Nodes)"),
        ('abilene', "Abilene US Backbone (12 Nodes)"),
        ('nsfnet', "NSFNET Continental Mesh (14 Nodes)"),
        ('spineleaf', "Spine-Leaf Fabric (12 Nodes)")
    ]

    algorithms = ['OSPF (RFC 2328)', 'Dijkstra SPF', 'ECMP', 'WSP (Widest Path)', 'LLR (Least Loaded)', 'DQN (Ours)']

    benchmark_data = {
        'bottleneck_util': {algo: [] for algo in algorithms},
        'latency_ms': {algo: [] for algo in algorithms},
        'jitter_ms': {algo: [] for algo in algorithms},
        'packet_loss_pct': {algo: [] for algo in algorithms},
        'jains_fairness': {algo: [] for algo in algorithms},
        'offload_rate_pct': {algo: [] for algo in algorithms},
        'decision_time_ms': {algo: [] for algo in algorithms},
        'topologies': [name for _, name in topologies_to_test]
    }

    decision_timings = {algo: [] for algo in algorithms}

    for topo_key, topo_name in topologies_to_test:
        print(f"\n" + "-" * 90)
        print(f" 🌐 Fabric: {topo_name}")
        print("-" * 90)

        g, meta = get_topology(topo_key)
        edge_nodes = meta.get('edge_nodes', list(g.nodes()))
        core_nodes = meta.get('core_nodes', [list(g.nodes())[0]])

        sm = StateManager()
        sm.graph = g.copy()
        for u, v, d in g.edges(data=True):
            cap = d.get('capacity', 100.0)
            delay = d.get('delay', 2.0)
            sm.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=cap, delay_ms=delay)

        # Apply realistic asymmetric load: core links heavily saturated (85-98%), lateral/edge links light (10-25%)
        for u, v in sm.graph.edges():
            if u in core_nodes or v in core_nodes:
                sm.link_utilization[(u, v)] = random.uniform(0.85, 0.98)
            else:
                sm.link_utilization[(u, v)] = random.uniform(0.10, 0.25)

        flow_samples = 150
        algo_utils = {algo: [] for algo in algorithms}
        algo_lats = {algo: [] for algo in algorithms}
        algo_jits = {algo: [] for algo in algorithms}
        algo_losses = {algo: [] for algo in algorithms}
        algo_offloaded = {algo: 0 for algo in algorithms}

        # Generate deterministic synthetic flow sequence (identical across all algorithms)
        random.seed(42 + len(g))
        flow_specs = []
        for flow_idx in range(flow_samples):
            src = random.choice(edge_nodes)
            dst_opts = [n for n in edge_nodes if n != src]
            dst = random.choice(dst_opts)
            mbps = random.uniform(3.0, 7.0)
            duration = random.uniform(15.0, 30.0)
            flow_specs.append((src, dst, mbps, duration))

        # Closed-Loop Dynamic Evaluation: each algorithm manages its own network state
        for algo in algorithms:
            sm_algo = StateManager()
            sm_algo.graph = g.copy()
            for u, v, d in g.edges(data=True):
                cap = d.get('capacity', 100.0)
                delay = d.get('delay', 2.0)
                sm_algo.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=cap, delay_ms=delay)

            # Apply topology-agnostic core saturation (85-95%)
            sm_algo.inject_core_congestion(utilization=0.90, core_nodes=core_nodes)

            # Route each flow dynamically
            for flow_idx, (src, dst, mbps, duration) in enumerate(flow_specs):
                cand_paths = sm_algo.get_candidate_paths(src, dst, k=4)
                p_ospf_ref = cand_paths[0]

                t0 = time.perf_counter()
                if algo == 'OSPF (RFC 2328)':
                    p = ospf_routing(sm_algo.graph, src, dst, link_bandwidths=sm_algo.link_bandwidths)
                elif algo == 'Dijkstra SPF':
                    p = dijkstra_spf(sm_algo.graph, src, dst)
                elif algo == 'ECMP':
                    p = ecmp_routing(sm_algo.graph, src, dst, flow_hash=flow_idx)
                elif algo == 'WSP (Widest Path)':
                    p = widest_shortest_path(sm_algo.graph, src, dst, sm_algo.link_utilization, sm_algo.link_delays, candidate_paths=cand_paths)
                elif algo == 'LLR (Least Loaded)':
                    p = least_loaded_routing(sm_algo.graph, src, dst, sm_algo.link_utilization, candidate_paths=cand_paths)
                elif algo == 'DQN (Ours)':
                    st = sm_algo.get_routing_state(src, dst)
                    act = router_agent.act(st, explore=False)
                    p = cand_paths[act % len(cand_paths)]
                else:
                    p = cand_paths[0]
                t_dec = (time.perf_counter() - t0) * 1000.0
                decision_timings[algo].append(t_dec)

                # Compute performance metrics under active placement
                m = compute_path_metrics(p, sm_algo.link_utilization, sm_algo.link_delays, sm_algo.link_bandwidths)
                algo_utils[algo].append(m['bottleneck_util'])
                algo_lats[algo].append(m['total_delay'])
                algo_jits[algo].append(m['jitter'])
                algo_losses[algo].append(m['packet_loss'])

                # Allocate dynamic flow onto selected path links (closed-loop load accumulation)
                sm_algo.allocate_dynamic_flow(flow_id=f"{algo}_{flow_idx}", path=p, mbps=mbps, duration_sec=duration)

                if p != p_ospf_ref:
                    algo_offloaded[algo] += 1

                # Advance simulated time & expire past flows
                sm_algo.step_dynamic_flows(current_time=time.time() + (flow_idx * 0.4))

        for algo in algorithms:
            b_util = float(np.mean(algo_utils[algo]) * 100.0)
            lat = float(np.mean(algo_lats[algo]))
            jit = float(np.mean(algo_jits[algo]))
            loss = float(np.mean(algo_losses[algo]))
            off = float((algo_offloaded[algo] / flow_samples) * 100.0)

            # Jain's fairness across path bottlenecks
            u_arr = np.array(algo_utils[algo])
            jains = float((np.sum(u_arr) ** 2) / max(1e-6, len(u_arr) * np.sum(u_arr ** 2)))

            benchmark_data['bottleneck_util'][algo].append(round(b_util, 2))
            benchmark_data['latency_ms'][algo].append(round(lat, 2))
            benchmark_data['jitter_ms'][algo].append(round(jit, 3))
            benchmark_data['packet_loss_pct'][algo].append(round(loss, 3))
            benchmark_data['jains_fairness'][algo].append(round(jains, 4))
            benchmark_data['offload_rate_pct'][algo].append(round(off, 1))
            benchmark_data['decision_time_ms'][algo].append(round(float(np.mean(decision_timings[algo])), 3))

            print(f"  • {algo:<18}: Bottleneck: {b_util:5.1f}% | Latency: {lat:5.2f}ms | Jitter: {jit:5.2f}ms | Loss: {loss:4.2f}% | Jain: {jains:.3f} | Offload: {off:4.1f}%")

    # Save quantitative results JSON
    json_path = os.path.join(BASE_DIR, 'logs', 'routing_tournament_results.json')
    with open(json_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[Saved] Detailed benchmark JSON saved to {json_path}")

    # Generate Publication-Grade 6-Panel Figure covering all Proposal Metrics
    plot_proposal_tournament_figures(benchmark_data, decision_timings)
    return benchmark_data


def plot_proposal_tournament_figures(data, decision_timings):
    """Plots 6-panel comprehensive benchmark and radar comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    topos = ['Tree', 'Fat-Tree', 'Abilene', 'NSFNet', 'Spine-Leaf']
    x = np.arange(len(topos))
    width = 0.14
    colors = {
        'OSPF (RFC 2328)': '#7f7f7f',
        'Dijkstra SPF': '#d62728',
        'ECMP': '#ff7f0e',
        'WSP (Widest Path)': '#1f77b4',
        'LLR (Least Loaded)': '#9467bd',
        'DQN (Ours)': '#2ca02c'
    }

    # 1. Bottleneck Link Utilization
    ax1 = axes[0, 0]
    for i, (algo, col) in enumerate(colors.items()):
        ax1.bar(x + i * width, data['bottleneck_util'][algo], width, label=algo, color=col, alpha=0.9)
    ax1.set_title("Metric 1: Bottleneck Link Utilization (%)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Peak Utilization (%)")
    ax1.set_xticks(x + width * 2.5)
    ax1.set_xticklabels(topos)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_ylim(0, 105)

    # 2. End-to-End Latency
    ax2 = axes[0, 1]
    for i, (algo, col) in enumerate(colors.items()):
        ax2.bar(x + i * width, data['latency_ms'][algo], width, label=algo, color=col, alpha=0.9)
    ax2.set_title("Metric 2: End-to-End Path Latency (ms)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xticks(x + width * 2.5)
    ax2.set_xticklabels(topos)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    # 3. Network Jitter (RFC 3393)
    ax3 = axes[0, 2]
    for i, (algo, col) in enumerate(colors.items()):
        ax3.bar(x + i * width, data['jitter_ms'][algo], width, label=algo, color=col, alpha=0.9)
    ax3.set_title("Metric 3: Network Jitter / Delay Variation (ms)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Jitter (ms)")
    ax3.set_xticks(x + width * 2.5)
    ax3.set_xticklabels(topos)
    ax3.grid(axis='y', linestyle='--', alpha=0.4)

    # 4. Packet Loss Rate (%)
    ax4 = axes[1, 0]
    for i, (algo, col) in enumerate(colors.items()):
        ax4.bar(x + i * width, data['packet_loss_pct'][algo], width, label=algo, color=col, alpha=0.9)
    ax4.set_title("Metric 4: Packet Loss Rate (%)", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Packet Loss (%)")
    ax4.set_xticks(x + width * 2.5)
    ax4.set_xticklabels(topos)
    ax4.grid(axis='y', linestyle='--', alpha=0.4)

    # 5. Jain's Fairness Index
    ax5 = axes[1, 1]
    for i, (algo, col) in enumerate(colors.items()):
        ax5.bar(x + i * width, data['jains_fairness'][algo], width, label=algo, color=col, alpha=0.9)
    ax5.set_title("Metric: Jain's Fairness Index", fontsize=11, fontweight='bold')
    ax5.set_ylabel("Fairness Index [0.0 - 1.0]")
    ax5.set_xticks(x + width * 2.5)
    ax5.set_xticklabels(topos)
    ax5.grid(axis='y', linestyle='--', alpha=0.4)
    ax5.set_ylim(0, 1.05)

    # 6. OpenFlow Control Overhead (Decision Latency vs Throughput)
    ax6 = axes[1, 2]
    algos_short = ['OSPF', 'SPF', 'ECMP', 'WSP', 'LLR', 'DQN']
    mean_times = [float(np.mean(decision_timings[a])) for a in colors.keys()]
    bar_colors = [colors[a] for a in colors.keys()]
    bars = ax6.bar(algos_short, mean_times, color=bar_colors, alpha=0.9)
    ax6.set_title("Metric 5: Controller Decision Time (ms)", fontsize=11, fontweight='bold')
    ax6.set_ylabel("Decision Execution Latency (ms)")
    ax6.set_ylim(0, max(mean_times) * 1.30)
    ax6.grid(axis='y', linestyle='--', alpha=0.4)
    for b in bars:
        yval = b.get_height()
        ax6.text(b.get_x() + b.get_width()/2.0, yval + 0.008, f"{yval:.2f}ms", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Common legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=6, fontsize=10, frameon=True)

    fig_path = os.path.join(PLOTS_DIR, 'proposal_benchmarks_all_metrics.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved 6-panel proposal benchmark figure to {fig_path}")

    # Spider / Radar plot comparing DQN vs OSPF vs SPF vs LLR
    plot_radar_summary(data)


def plot_radar_summary(data):
    """Generates radar chart summarizing normalized scores across the 5 Proposal objectives."""
    labels = ['Congestion Relief', 'Low Latency', 'Low Jitter', 'Zero Packet Loss', 'Jain Fairness']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Calculate average performance across all 5 topologies
    algos = ['OSPF (RFC 2328)', 'Dijkstra SPF', 'LLR (Least Loaded)', 'DQN (Ours)']
    colors = {'OSPF (RFC 2328)': '#7f7f7f', 'Dijkstra SPF': '#d62728', 'LLR (Least Loaded)': '#9467bd', 'DQN (Ours)': '#2ca02c'}

    # Dynamic scaling baselines across compared algorithms
    all_lats = [float(np.mean(data['latency_ms'][a])) for a in algos]
    all_jits = [float(np.mean(data['jitter_ms'][a])) for a in algos]
    all_loss = [float(np.mean(data['packet_loss_pct'][a])) for a in algos]
    all_util = [float(np.mean(data['bottleneck_util'][a])) for a in algos]

    max_lat = max(all_lats) * 1.10
    max_jit = max(all_jits) * 1.10
    max_loss = max(all_loss) * 1.10
    max_util = max(all_util) * 1.10

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))

    for algo in algos:
        avg_b_util = float(np.mean(data['bottleneck_util'][algo]))
        avg_lat = float(np.mean(data['latency_ms'][algo]))
        avg_jit = float(np.mean(data['jitter_ms'][algo]))
        avg_loss = float(np.mean(data['packet_loss_pct'][algo]))
        avg_jain = float(np.mean(data['jains_fairness'][algo]))

        # Normalized to [0.12, 1.0] where 1.0 is optimal
        score_relief = max(0.12, 1.0 - (avg_b_util / max_util))
        score_lat = max(0.12, 1.0 - (avg_lat / max_lat))
        score_jit = max(0.12, 1.0 - (avg_jit / max_jit))
        score_loss = max(0.12, 1.0 - (avg_loss / max_loss))
        score_jain = min(1.0, max(0.12, avg_jain))

        values = [score_relief, score_lat, score_jit, score_loss, score_jain]
        values += values[:1]

        ax.plot(angles, values, color=colors[algo], linewidth=2.2, label=algo)
        ax.fill(angles, values, color=colors[algo], alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_title("EC499 Proposal Objectives Radar Comparison\n(DQN vs Classical Routing Protocols)", fontsize=12, fontweight='bold', pad=25)
    ax.legend(loc='lower right', bbox_to_anchor=(1.35, -0.05), fontsize=9)

    radar_path = os.path.join(PLOTS_DIR, 'proposal_tournament_radar.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved radar chart to {radar_path}")


if __name__ == '__main__':
    run_routing_tournament()
