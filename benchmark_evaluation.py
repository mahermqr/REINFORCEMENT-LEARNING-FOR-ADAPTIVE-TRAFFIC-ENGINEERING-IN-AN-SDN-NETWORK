#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Training & Evaluation Suite for Adaptive SDN Traffic Engineering (EC499).
Fulfills EC499 Proposal Objectives 1, 3, 4, 5 and Procedure 4:
 - Trains Double DQN Agent with Prioritized Experience Replay (PER) using 3-Phase Curriculum.
 - Simulates synthetic traffic patterns (Poisson bursts, core jamming, asymmetric loads).
 - Tracks Throughput, Bottleneck Utilization, Latency, Jitter, Packet Loss, and Loss/Reward convergence.
 - Saves trained weights in models/dqn_router.pth.
 - Generates publication-grade convergence figures in logs/plots/dqn_te_training_convergence.png.
"""

import sys
import os
import time
import json
import csv
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup path imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'agent'))
sys.path.append(os.path.join(BASE_DIR, 'controller'))
sys.path.append(os.path.join(BASE_DIR, 'topology'))

from dqn_router import DQNRoutingAgent
from state_manager import StateManager
from topology_library import ALL_TOPOLOGY_BUILDERS
from traditional_routing import dijkstra_spf, ospf_routing, compute_path_metrics

MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
PLOTS_DIR = os.path.join(LOGS_DIR, 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def build_evaluation_topology():
    """Builds the 7-switch hierarchical tree topology with cross-links."""
    g = nx.DiGraph()
    for i in range(1, 8):
        g.add_node(i)

    links = [
        (1, 2, 100.0, 2.0), (2, 1, 100.0, 2.0),
        (1, 3, 100.0, 2.0), (3, 1, 100.0, 2.0),
        (2, 4, 50.0, 3.0),  (4, 2, 50.0, 3.0),
        (2, 5, 50.0, 3.0),  (5, 2, 50.0, 3.0),
        (3, 6, 50.0, 3.0),  (6, 3, 50.0, 3.0),
        (3, 7, 50.0, 3.0),  (7, 3, 50.0, 3.0),
        # Redundant cross links
        (4, 6, 30.0, 8.0),  (6, 4, 30.0, 8.0),
        (5, 7, 30.0, 8.0),  (7, 5, 30.0, 8.0),
    ]
    for u, v, bw, lat in links:
        g.add_edge(u, v, capacity=bw, delay=lat, util=0.0)
    return g


def run_full_training(episodes=1000):
    print("=" * 85)
    print(f" 🚀 STARTING DEEP Q-NETWORK ADAPTIVE TRAFFIC ENGINEERING TRAINING ({episodes} EPISODES)")
    print("=" * 85)

    # Initialize topologies for multi-fabric curriculum exposure
    topology_instances = {}
    for tid, builder in ALL_TOPOLOGY_BUILDERS.items():
        g, meta = builder()
        t_sm = StateManager()
        t_sm.graph = g.copy()
        for u, v, data in g.edges(data=True):
            t_sm.update_link(u, v, src_port=1, dst_port=1,
                             capacity_mbps=data.get('capacity', 100.0),
                             delay_ms=data.get('delay', 2.0))
        topology_instances[tid] = {
            'sm': t_sm,
            'meta': meta,
            'edge_nodes': meta.get('edge_nodes', list(g.nodes())),
            'core_nodes': meta.get('core_nodes', [list(g.nodes())[0]])
        }

    topologies_list = list(topology_instances.keys())

    # Initialize DQN Agent with Prioritized Experience Replay
    router_agent = DQNRoutingAgent(
        state_size=10,
        action_size=4,
        lr=0.0008,
        memory_size=20000,
        epsilon_decay=1.0 # Decay managed explicitly across 3-phase curriculum
    )

    # 3-Phase Curriculum Boundaries:
    phase1_end = int(episodes * 0.25)  # Exploration Phase: 0 - 25%
    phase2_end = int(episodes * 0.75)  # Learning Phase: 25% - 75%

    history = {
        'episodes': [],
        'rewards': [],
        'losses': [],
        'dqn_bottleneck': [],
        'spf_bottleneck': [],
        'latency_ms': [],
        'jitter_ms': [],
        'packet_loss_pct': [],
        'epsilons': []
    }

    csv_path = os.path.join(LOGS_DIR, 'training_metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'reward', 'loss', 'dqn_bottleneck', 'spf_bottleneck', 'latency_ms', 'jitter_ms', 'loss_pct', 'epsilon'])

    start_time = time.time()

    for ep in range(1, episodes + 1):
        # 3-Phase Curriculum Exploration Schedule
        if ep <= phase1_end:
            router_agent.epsilon = max(0.50, 1.0 - (ep / phase1_end) * 0.50)
        elif ep <= phase2_end:
            progress = (ep - phase1_end) / (phase2_end - phase1_end)
            router_agent.epsilon = max(0.02, 0.50 - progress * 0.48)
        else:
            router_agent.epsilon = 0.01

        # 1. Multi-Fabric Sampling: Cycle through all registered network topologies
        topo_key = topologies_list[(ep - 1) % len(topologies_list)]
        t_inst = topology_instances[topo_key]
        sm = t_inst['sm']
        edge_nodes = t_inst['edge_nodes']
        core_nodes = t_inst['core_nodes']

        # 2. Select random source and destination from edge nodes
        src = random.choice(edge_nodes)
        dst_opts = [n for n in edge_nodes if n != src]
        dst = random.choice(dst_opts)

        cand_paths = sm.get_candidate_paths(src, dst, k=4)
        spf_path = cand_paths[0]

        # 3. Simulate synthetic traffic patterns:
        #  - Mode A (35%): Nominal network (all links lightly loaded 0.10 - 0.35)
        #  - Mode B (45%): Congested Primary Path (links along Path 0 saturated at 0.75 - 0.98, alternate paths clean at 0.10 - 0.35)
        #  - Mode C (20%): Core Jamming (core/aggregation transit nodes saturated)
        traffic_mode = random.choices(['nominal', 'congested_primary', 'core_jam'], weights=[0.35, 0.45, 0.20])[0]

        # Reset nominal link loads
        for u, v in sm.graph.edges():
            sm.link_utilization[(u, v)] = random.uniform(0.10, 0.30)

        if traffic_mode == 'congested_primary':
            # Saturate links along candidate path 0 (Shortest Path)
            for idx in range(len(spf_path) - 1):
                u, v = spf_path[idx], spf_path[idx + 1]
                sm.link_utilization[(u, v)] = random.uniform(0.78, 0.98)
                if (v, u) in sm.link_utilization:
                    sm.link_utilization[(v, u)] = sm.link_utilization[(u, v)]
        elif traffic_mode == 'core_jam':
            for u, v in sm.graph.edges():
                if u in core_nodes or v in core_nodes:
                    sm.link_utilization[(u, v)] = random.uniform(0.82, 0.98)

        # 4. Agent Decision
        state = sm.get_routing_state(src, dst)
        action = router_agent.act(state, explore=True)
        chosen_path = cand_paths[action % len(cand_paths)]

        # 5. Telemetry calculation
        m_spf = compute_path_metrics(spf_path, sm.link_utilization, sm.link_delays, sm.link_bandwidths)
        m_dqn = compute_path_metrics(chosen_path, sm.link_utilization, sm.link_delays, sm.link_bandwidths)

        u_spf = m_spf['bottleneck_util']
        u_dqn = m_dqn['bottleneck_util']
        hops_dqn = m_dqn['hops']
        hops_spf = m_spf['hops']
        lat_dqn = m_dqn['total_delay']
        jit_dqn = m_dqn['jitter']
        loss_dqn = m_dqn['packet_loss']

        # 6. Intelligent Traffic Engineering Multi-Objective Reward:
        if u_spf <= 0.60:
            # Case 1: Primary path is uncongested -> Prefer Shortest Path (Action 0)
            if action == 0:
                reward = 4.0 - 0.04 * lat_dqn - 0.1 * jit_dqn
            else:
                hop_diff = max(0, hops_dqn - hops_spf)
                reward = 1.0 - 1.0 * hop_diff - 0.05 * lat_dqn - 0.1 * jit_dqn
        else:
            # Case 2: Primary path is congested (u_spf > 0.60)!
            # Strongly incentivize diverting traffic to less loaded candidate paths
            if u_dqn < u_spf - 0.10:
                # Big positive reward for offloading traffic away from congestion
                relief = u_spf - u_dqn
                reward = 12.0 * relief + 4.0 * (1.0 - u_dqn) - 0.15 * jit_dqn - 1.5 * loss_dqn
            elif action == 0:
                # Severe penalty for driving traffic into the jammed primary queue
                reward = - 14.0 * (u_spf ** 2) - 0.3 * jit_dqn - 3.0 * loss_dqn
            else:
                # Alternative path chosen is also congested
                reward = - 8.0 * (u_dqn ** 2) - 0.3 * jit_dqn - 2.0 * loss_dqn

        # 7. Next state & Experience Replay
        next_state = sm.get_routing_state(src, dst)
        router_agent.remember(state, action, reward, next_state, done=False)

        # 8. Train policy network with mini-batch Double DQN update
        loss_val = router_agent.train(batch_size=32) or 0.05

        # Record history
        history['episodes'].append(ep)
        history['rewards'].append(reward)
        history['losses'].append(loss_val)
        history['dqn_bottleneck'].append(u_dqn * 100.0)
        history['spf_bottleneck'].append(u_spf * 100.0)
        history['latency_ms'].append(lat_dqn)
        history['jitter_ms'].append(jit_dqn)
        history['packet_loss_pct'].append(loss_dqn)
        history['epsilons'].append(router_agent.epsilon)

        csv_writer.writerow([ep, round(reward, 4), round(loss_val, 4), round(u_dqn * 100, 2),
                             round(u_spf * 100, 2), round(lat_dqn, 2), round(jit_dqn, 3), round(loss_dqn, 3), round(router_agent.epsilon, 3)])

        if ep % 100 == 0 or ep == episodes:
            avg_r = np.mean(history['rewards'][-100:])
            avg_loss = np.mean(history['losses'][-100:])
            avg_b_dqn = np.mean(history['dqn_bottleneck'][-100:])
            avg_b_spf = np.mean(history['spf_bottleneck'][-100:])
            avg_lat = np.mean(history['latency_ms'][-100:])
            avg_jit = np.mean(history['jitter_ms'][-100:])
            avg_loss_p = np.mean(history['packet_loss_pct'][-100:])
            relief = avg_b_spf - avg_b_dqn

            print(f"[Ep {ep:04d}/{episodes}] Fabric: {topo_key:<9} | Reward: {avg_r:6.2f} | Loss: {avg_loss:6.4f} | "
                  f"DQN Bottleneck: {avg_b_dqn:5.1f}% vs SPF: {avg_b_spf:5.1f}% (Relief: +{relief:4.1f}%) | "
                  f"Latency: {avg_lat:4.1f}ms | Jitter: {avg_jit:4.2f}ms | Loss: {avg_loss_p:4.2f}% | Eps: {router_agent.epsilon:.3f}")

    csv_file.close()
    elapsed = time.time() - start_time
    print(f"\n[Completed] {episodes} episodes completed in {elapsed:.1f} seconds.")

    # Save trained checkpoint
    save_path = os.path.join(MODELS_DIR, 'dqn_router.pth')
    router_agent.save(save_path)
    print(f"[Saved] Checkpoint successfully saved to {save_path}")

    # Generate Evaluation Results JSON
    eval_results = {
        'total_episodes': episodes,
        'training_time_seconds': round(elapsed, 1),
        'final_mean_reward': round(float(np.mean(history['rewards'][-200:])), 3),
        'final_mean_loss': round(float(np.mean(history['losses'][-200:])), 4),
        'final_dqn_bottleneck_pct': round(float(np.mean(history['dqn_bottleneck'][-200:])), 2),
        'final_spf_bottleneck_pct': round(float(np.mean(history['spf_bottleneck'][-200:])), 2),
        'congestion_relief_pct': round(float(np.mean(history['spf_bottleneck'][-200:]) - np.mean(history['dqn_bottleneck'][-200:])), 2),
        'final_latency_ms': round(float(np.mean(history['latency_ms'][-200:])), 2),
        'final_jitter_ms': round(float(np.mean(history['jitter_ms'][-200:])), 3),
        'final_packet_loss_pct': round(float(np.mean(history['packet_loss_pct'][-200:])), 3),
        'exploration_decay': '3-Phase Curriculum (Exploration -> Learning -> Exploitation)'
    }
    with open(os.path.join(LOGS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)

    # Plot Convergence Figures
    plot_convergence_dashboard(history, phase1_end, phase2_end, episodes)
    return eval_results


def plot_convergence_dashboard(history, p1_end, p2_end, total_episodes):
    """Generates 4-panel publication-grade convergence dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    window = max(10, total_episodes // 50)
    def smooth(arr):
        return np.convolve(arr, np.ones(window)/window, mode='valid')

    x_smooth = np.arange(window, total_episodes + 1)

    # 1. Episode Reward
    ax1 = axes[0, 0]
    ax1.plot(history['episodes'], history['rewards'], color='#1f77b4', alpha=0.15)
    ax1.plot(x_smooth, smooth(history['rewards']), color='#1f77b4', linewidth=2.0, label='DQN Reward (Smoothed)')
    ax1.axvline(p1_end, color='red', linestyle='--', alpha=0.7, label='Phase 1 Boundary')
    ax1.axvline(p2_end, color='green', linestyle='--', alpha=0.7, label='Phase 2 Boundary')
    ax1.set_title("DQN Reward Convergence (3-Phase Curriculum)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Reward")
    ax1.set_xlabel("Training Episodes")
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right', fontsize=8)

    # 2. Bellman Loss
    ax2 = axes[0, 1]
    ax2.plot(history['episodes'], history['losses'], color='#d62728', alpha=0.15)
    ax2.plot(x_smooth, smooth(history['losses']), color='#d62728', linewidth=2.0, label='Smooth L1 Loss')
    ax2.axvline(p1_end, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(p2_end, color='green', linestyle='--', alpha=0.7)
    ax2.set_title("Double DQN Loss Stabilization", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Training Episodes")
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=8)

    # 3. Bottleneck Congestion: DQN vs Dijkstra SPF Baseline
    ax3 = axes[1, 0]
    ax3.plot(x_smooth, smooth(history['spf_bottleneck']), color='#7f7f7f', linestyle='--', linewidth=1.8, label='SPF Baseline Load')
    ax3.plot(x_smooth, smooth(history['dqn_bottleneck']), color='#2ca02c', linewidth=2.2, label='DQN Optimized Load')
    ax3.set_title("Bottleneck Link Utilization: DQN vs SPF Baseline", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Bottleneck Utilization (%)")
    ax3.set_xlabel("Training Episodes")
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper right', fontsize=8)

    # 4. Latency, Jitter, and Packet Loss Convergence
    ax4 = axes[1, 1]
    ax4.plot(x_smooth, smooth(history['latency_ms']), color='#ff7f0e', linewidth=1.8, label='Latency (ms)')
    ax4.plot(x_smooth, smooth(history['jitter_ms']), color='#9467bd', linewidth=1.8, label='Jitter (ms)')
    ax4.plot(x_smooth, smooth(history['packet_loss_pct']), color='#e377c2', linewidth=1.8, label='Packet Loss (%)')
    ax4.set_title("Latency, Jitter & Packet Loss Dynamics", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Metric Value")
    ax4.set_xlabel("Training Episodes")
    ax4.grid(True, linestyle='--', alpha=0.4)
    ax4.legend(loc='upper right', fontsize=8)

    plt.suptitle("Deep Q-Network Adaptive Traffic Engineering: Training Progression & Metric Convergence\n(EC499 - University of Tripoli)", fontsize=13, fontweight='bold')
    plot_file = os.path.join(PLOTS_DIR, 'dqn_te_training_convergence.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Convergence dashboard saved to {plot_file}")


if __name__ == '__main__':
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_full_training(episodes=episodes)
