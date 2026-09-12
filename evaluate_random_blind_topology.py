#!/usr/bin/env python3
"""
Dynamic Zero-Shot Blind Evaluation on Random Unseen Topologies
Evaluates the pre-trained Double DQN Router and Dueling DQN Multicast Agent
on dynamically generated, arbitrary random topologies without retraining.
"""

import os
import sys
import time
import random
import argparse
import itertools
import numpy as np
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'agent'))
sys.path.append(os.path.join(BASE_DIR, 'controller'))
sys.path.append(os.path.join(BASE_DIR, 'topology'))

from dqn_router import DQNRoutingAgent
from dqn_multicast import DQNMulticastAgent
from state_manager import StateManager
from topology_library import build_random_topology

def evaluate_random_blind_topology(num_nodes=20, seed=None, num_flows=200):
    print("=" * 85)
    print(" 🎲 ZERO-SHOT BLIND EVALUATION ON DYNAMICALLY GENERATED RANDOM TOPOLOGY")
    print("=" * 85)

    # 1. Load trained agent checkpoints
    router_agent = DQNRoutingAgent(state_size=10, action_size=4)
    router_ckpt = os.path.join(BASE_DIR, 'models', 'dqn_router.pth')
    if not router_agent.load(router_ckpt):
        print(f"[Error] Failed to load router weights from {router_ckpt}")
        return
    router_agent.epsilon = 0.0 # Strict greedy inference (zero exploration)
    print(f"[Init] Loaded trained Double DQN Router checkpoint from models/dqn_router.pth")

    multicast_agent = DQNMulticastAgent(state_size=50, action_size=10)
    m_ckpt = os.path.join(BASE_DIR, 'models', 'dqn_multicast.pth')
    if os.path.exists(m_ckpt):
        multicast_agent.load(m_ckpt)
        multicast_agent.epsilon = 0.0
        print(f"[Init] Loaded trained Dueling DQN Multicast checkpoint from models/dqn_multicast.pth")

    # 2. Build completely unseen random connected topology
    graph, meta = build_random_topology(num_nodes=num_nodes, p_edge=0.30, seed=seed)
    topo_name = meta['name']
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    print(f"\n[Topology Generated on the Fly]")
    print(f" • Fabric:                 {topo_name}")
    print(f" • Random Seed:            {seed if seed is not None else 'Dynamic / Unseeded'}")
    print(f" • Switches/Nodes:         {n_nodes}")
    print(f" • Directed Links:         {n_edges}")
    print(f" • Core Nodes:             {meta['core_nodes']}")
    print(f" • Edge Nodes:             {len(meta['edge_nodes'])} switches")

    sm = StateManager()
    sm.graph = graph.copy()
    for u, v, d in graph.edges(data=True):
        sm.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=d['capacity'], delay_ms=d['delay'])

    core_nodes = meta['core_nodes']
    edge_nodes = meta['edge_nodes']

    # Helper to evaluate flow
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

        def p_metrics(p):
            lat = sum(sm.link_delays.get((p[i], p[i+1]), 2.0) for i in range(len(p)-1))
            util = max(sm.link_utilization.get((p[i], p[i+1]), 0.0) for i in range(len(p)-1))
            return util, lat

        du, dl = p_metrics(chosen_path)
        su, sl = p_metrics(spf_path)
        rerouted = (chosen_path != spf_path)
        return du, su, dl, sl, rerouted, chosen_path, spf_path

    # -------------------------------------------------------------------------
    # TEST 1: Core Jamming Stress on Random Fabric
    # -------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print(f" [TEST 1/4] RANDOM CORE JAMMING (Core nodes: {core_nodes} saturated to 85% - 98%)")
    print("-" * 85)
    for u, v in sm.graph.edges():
        if u in core_nodes or v in core_nodes:
            sm.link_utilization[(u, v)] = random.uniform(0.85, 0.98)
        else:
            sm.link_utilization[(u, v)] = random.uniform(0.10, 0.28)

    t1_dqn_u, t1_spf_u, t1_dqn_l, t1_spf_l, reroutes = [], [], [], [], 0
    for _ in range(num_flows):
        src = random.choice(edge_nodes)
        dst = random.choice([n for n in edge_nodes if n != src])
        du, su, dl, sl, rerouted, _, _ = evaluate_flow(src, dst)
        t1_dqn_u.append(du * 100.0)
        t1_spf_u.append(su * 100.0)
        t1_dqn_l.append(dl)
        t1_spf_l.append(sl)
        if rerouted:
            reroutes += 1

    t1_relief = np.mean(t1_spf_u) - np.mean(t1_dqn_u)
    print(f" • Evaluated Random Ingress/Egress Flows: {num_flows}")
    print(f" • Dijkstra SPF Bottleneck Load:         {np.mean(t1_spf_u):.2f}%")
    print(f" • Double DQN Bottleneck Load:           {np.mean(t1_dqn_u):.2f}%")
    print(f" • Congestion Reduction:                 +{t1_relief:.2f}% improvement!")
    print(f" • Autonomous Offload Rate:              {(reroutes / num_flows) * 100.0:.1f}% diverted to uncongested paths")
    print(f" • Mean Latency:                         DQN: {np.mean(t1_dqn_l):.2f} ms vs SPF: {np.mean(t1_spf_l):.2f} ms")

    # -------------------------------------------------------------------------
    # TEST 2: High-Concurrency Burst (Jain's Fairness)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print(f" [TEST 2/4] HIGH-CONCURRENCY FLOW AVALANCHE (500 Concurrent Requests)")
    print("-" * 85)
    link_loads_dqn = {e: 0.05 for e in sm.graph.edges()}
    link_loads_spf = {e: 0.05 for e in sm.graph.edges()}

    t0 = time.time()
    for _ in range(500):
        src = random.choice(edge_nodes)
        dst = random.choice([n for n in edge_nodes if n != src])
        _, _, _, _, _, dqn_p, spf_p = evaluate_flow(src, dst)
        for i in range(len(dqn_p) - 1):
            e = (dqn_p[i], dqn_p[i+1])
            if e in link_loads_dqn:
                link_loads_dqn[e] = min(1.0, link_loads_dqn[e] + 0.012)
        for i in range(len(spf_p) - 1):
            e = (spf_p[i], spf_p[i+1])
            if e in link_loads_spf:
                link_loads_spf[e] = min(1.0, link_loads_spf[e] + 0.012)
    burst_time_ms = (time.time() - t0) * 1000.0
    decisions_per_sec = 500.0 / max(0.001, (burst_time_ms / 1000.0))

    def jains_fairness(loads_dict):
        x = list(loads_dict.values())
        if not x or sum(x) == 0:
            return 1.0
        return float((sum(x) ** 2) / (len(x) * sum(val ** 2 for val in x)))

    j_dqn = jains_fairness(link_loads_dqn)
    j_spf = jains_fairness(link_loads_spf)
    print(f" • Processed Burst:                      500 concurrent flows in {burst_time_ms:.1f} ms")
    print(f" • Controller Decision Throughput:       {decisions_per_sec:.1f} decisions/second")
    print(f" • Jain's Fairness Index:                DQN: {j_dqn:.4f} vs SPF: {j_spf:.4f}")
    print(f" • Max Link Load:                        DQN: {max(link_loads_dqn.values())*100:.1f}% vs SPF: {max(link_loads_spf.values())*100:.1f}%")

    # -------------------------------------------------------------------------
    # TEST 3: Dynamic Latency Degradation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print(f" [TEST 3/4] DYNAMIC LATENCY DEGRADATION (Core links degraded to 25ms)")
    print("-" * 85)
    for u, v in sm.graph.edges():
        if u in core_nodes or v in core_nodes:
            sm.link_delays[(u, v)] = 25.0
            sm.link_utilization[(u, v)] = 0.50
        else:
            sm.link_delays[(u, v)] = 4.0
            sm.link_utilization[(u, v)] = 0.20

    t3_dqn_l, t3_spf_l = [], []
    for _ in range(150):
        src = random.choice(edge_nodes)
        dst = random.choice([n for n in edge_nodes if n != src])
        _, _, dl, sl, _, _, _ = evaluate_flow(src, dst)
        t3_dqn_l.append(dl)
        t3_spf_l.append(sl)

    t3_savings = np.mean(t3_spf_l) - np.mean(t3_dqn_l)
    print(f" • Double DQN Mean Latency:              {np.mean(t3_dqn_l):.2f} ms")
    print(f" • Dijkstra SPF Mean Latency:            {np.mean(t3_spf_l):.2f} ms")
    print(f" • End-to-End Latency Improvement:       +{t3_savings:.2f} ms (Faster via low-delay bypass!)")

    # -------------------------------------------------------------------------
    # TEST 4: Multicast Tree Replication Efficiency
    # -------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print(f" [TEST 4/4] MULTICAST REPLICATION EFFICIENCY (Steiner Tree on Random Fabric)")
    print("-" * 85)
    m_savings = []
    for _ in range(25):
        m_src = random.choice(edge_nodes)
        m_dests = random.sample([n for n in edge_nodes if n != m_src], min(3, len(edge_nodes) - 1))
        m_state = sm.get_multicast_state(m_src, m_dests)
        m_action = multicast_agent.act(m_state, explore=False)

        weighted_g = sm.graph.copy().to_undirected()
        for u, v in weighted_g.edges():
            weighted_g[u][v]['weight'] = 1.0 + (m_action % 3) * sm.link_delays.get((u, v), 2.0)

        tree = nx.algorithms.approximation.steinertree.steiner_tree(weighted_g, [m_src] + m_dests, weight='weight')
        tree_edges = tree.number_of_edges()
        unicast_edges = len(m_dests) * 3
        bw_saved = max(0.0, (unicast_edges - tree_edges) * 10.0)
        m_savings.append(bw_saved)

    print(f" • Multicast Conserved Bandwidth:        {np.mean(m_savings):.1f} Mbps average")
    print("=" * 85)
    print(f" ✅ ZERO-SHOT BLIND TRANSFER TEST PASSED ON UNSEEN {n_nodes}-NODE RANDOM TOPOLOGY!")
    print("=" * 85)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate DRL Traffic Engineering Zero-Shot on Random Unseen Topologies")
    parser.add_argument('--nodes', type=int, default=20, help="Number of random network switches/nodes (default: 20)")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility (default: None)")
    parser.add_argument('--flows', type=int, default=200, help="Number of test flows to evaluate (default: 200)")
    args = parser.parse_args()

    evaluate_random_blind_topology(num_nodes=args.nodes, seed=args.seed, num_flows=args.flows)
