"""
Traditional SDN Routing Algorithms Baseline Engine (EC499).
Implements classical network routing protocols and greedy heuristics for head-to-head benchmarking:
 1. OSPF (Open Shortest Path First - RFC 2328 with Reference Bandwidth Cost Metric)
 2. Dijkstra Shortest Path First (SPF - Hop Count Metric)
 3. Equal-Cost Multi-Path (ECMP)
 4. Widest Shortest Path (WSP / CSPF - Residual Bandwidth Priority)
 5. Greedy Least Loaded Routing (LLR - Minimum Peak & Average Utilization)
 6. Random Path Routing Baseline

Computes comprehensive network telemetry metrics required by Proposal:
 - Bottleneck Link Utilization (%)
 - Total Path Latency (ms)
 - Network Jitter (ms Delay Variation)
 - Packet Loss Rate (%)
 - Hop Count & Control Overhead
"""

import itertools
import networkx as nx
import numpy as np


def compute_path_metrics(path, link_utilization, link_delays=None, link_bandwidths=None, jitter_history=None):
    """
    Computes bottleneck utilization, total latency, network jitter, packet loss rate, and hop count.

    Mathematical Models:
      1. Latency (ms): D_total = sum(D_e) + queuing_delay(U_e)
      2. Jitter (ms): Packet Delay Variation (RFC 3393). In packet queues, queuing variation
         scales non-linearly with bottleneck link utilization:
         Jitter = base_jitter + 0.15 * total_delay * (U_bottleneck / max(0.01, 1.0 - min(0.98, U_bottleneck)))
      3. Packet Loss (%): Finite buffer queue overflow model (M/M/1/K queuing approximation).
         Loss is near 0% under low load (U <= 70%), then escalates sharply when links saturate (U > 70%).
    """
    if not path or len(path) <= 1:
        return {
            'bottleneck_util': 0.0,
            'total_delay': 0.0,
            'jitter': 0.0,
            'packet_loss': 0.0,
            'hops': 0
        }

    delays = link_delays or {}
    bws = link_bandwidths or {}
    hops = len(path) - 1
    total_delay = 0.0
    bottleneck_util = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        util = float(link_utilization.get((u, v), 0.0))
        # Ensure bidirectional or directional fallback
        if util == 0.0 and (v, u) in link_utilization:
            util = float(link_utilization[(v, u)])

        base_delay = float(delays.get((u, v), delays.get((v, u), 2.0)))

        # Queuing delay scales with link utilization: M/M/1 queue delay
        # D_e = base_delay / (1 - min(0.95, util))
        clamped_util = min(0.95, max(0.0, util))
        queuing_delay = base_delay * (0.2 + 0.8 / max(0.05, 1.0 - clamped_util))

        total_delay += queuing_delay
        bottleneck_util = max(bottleneck_util, util)

    # 1. Network Jitter (RFC 3393 Delay Variation)
    # Fluctuation in queuing delay increases drastically under heavy load
    base_jitter = 0.35 * hops
    congestion_factor = (bottleneck_util ** 2) / max(0.02, 1.0 - min(0.98, bottleneck_util))
    jitter_ms = float(base_jitter + 0.45 * total_delay * min(5.0, congestion_factor))

    # 2. Packet Loss Rate (%):
    # Under low load (util < 0.70), loss is negligible (< 0.05%)
    # Under high load (util >= 0.70), queue drops escalate exponentially up to 25-35%
    if bottleneck_util <= 0.70:
        loss_pct = float(max(0.0, bottleneck_util * 0.05))
    else:
        excess = (bottleneck_util - 0.70) / 0.30
        loss_pct = float(min(35.0, 0.05 + 30.0 * (excess ** 2.2)))

    return {
        'bottleneck_util': float(bottleneck_util),
        'total_delay': float(total_delay),
        'jitter': float(jitter_ms),
        'packet_loss': float(loss_pct),
        'hops': int(hops)
    }


def ospf_routing(graph, src, dst, link_bandwidths=None, reference_bandwidth=100.0):
    """
    Standard OSPF (Open Shortest Path First) Routing (RFC 2328).
    Computes shortest path based on standard Cisco/IETF OSPF interface cost:
      Cost(u, v) = max(1, round(Reference_Bandwidth / Capacity_Mbps))
    High-bandwidth links have lower cost; lower-bandwidth links have higher cost.
    OSPF remains static and oblivious to dynamic traffic load.
    """
    bws = link_bandwidths or {}
    ospf_graph = graph.copy()

    for u, v in ospf_graph.edges():
        cap = float(bws.get((u, v), bws.get((v, u), 100.0)))
        # OSPF Cost metric: Cost = Reference_Bandwidth / Capacity
        cost = max(1.0, round(float(reference_bandwidth) / max(1.0, cap)))
        ospf_graph[u][v]['ospf_cost'] = cost

    try:
        return list(nx.shortest_path(ospf_graph, src, dst, weight='ospf_cost'))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [src, dst]


def dijkstra_spf(graph, src, dst, weight=None):
    """
    Standard Dijkstra Shortest Path First (SPF) with hop count metric (unit edge weights).
    Pure shortest-hop path calculation.
    """
    try:
        return list(nx.shortest_path(graph, src, dst, weight=weight))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [src, dst]


def ecmp_routing(graph, src, dst, flow_hash=0):
    """
    Equal-Cost Multi-Path (ECMP).
    Discovers all shortest paths of identical minimum hop count and hashes flows across them.
    Incapable of offloading onto slightly longer but uncongested alternative paths.
    """
    try:
        all_shortest = list(nx.all_shortest_paths(graph, src, dst))
        if not all_shortest:
            return [src, dst]
        selected_idx = int(flow_hash) % len(all_shortest)
        return all_shortest[selected_idx]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [src, dst]


def widest_shortest_path(graph, src, dst, link_utilization, link_delays=None, candidate_paths=None):
    """
    Widest Shortest Path (WSP / CSPF).
    Greedy heuristic: evaluates candidate paths and selects the path with the largest
    residual bandwidth (minimum bottleneck link utilization). Breaks ties using shortest latency/hop count.
    """
    if candidate_paths is None:
        try:
            candidate_paths = list(itertools.islice(nx.shortest_simple_paths(graph, src, dst), 8))
        except Exception:
            candidate_paths = [[src, dst]]

    if not candidate_paths:
        return [src, dst]

    delays = link_delays or {}
    scored_paths = []
    for p in candidate_paths:
        metrics = compute_path_metrics(p, link_utilization, delays)
        # Score tuple: (bottleneck_util ascending, total_delay ascending, hops ascending)
        scored_paths.append((metrics['bottleneck_util'], metrics['total_delay'], metrics['hops'], p))

    scored_paths.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored_paths[0][3]


def least_loaded_routing(graph, src, dst, link_utilization, candidate_paths=None):
    """
    Greedy Least Loaded Routing (LLR).
    Evaluates candidate paths and selects the path that minimizes the blended
    maximum and average link utilization across all hops.
    """
    if candidate_paths is None:
        try:
            candidate_paths = list(itertools.islice(nx.shortest_simple_paths(graph, src, dst), 8))
        except Exception:
            candidate_paths = [[src, dst]]

    if not candidate_paths:
        return [src, dst]

    scored_paths = []
    for p in candidate_paths:
        if len(p) <= 1:
            scored_paths.append((0.0, p))
            continue
        utils = [link_utilization.get((p[i], p[i+1]), 0.0) for i in range(len(p)-1)]
        avg_util = sum(utils) / max(1, len(utils))
        max_util = max(utils) if utils else 0.0
        # Blend peak and average link load
        load_metric = 0.65 * max_util + 0.35 * avg_util
        scored_paths.append((load_metric, p))

    scored_paths.sort(key=lambda item: item[0])
    return scored_paths[0][1]


def random_routing(graph, src, dst, candidate_paths=None):
    """
    Random Routing baseline. Uniform random selection across candidate paths.
    """
    if candidate_paths is None:
        try:
            candidate_paths = list(itertools.islice(nx.shortest_simple_paths(graph, src, dst), 4))
        except Exception:
            candidate_paths = [[src, dst]]

    if not candidate_paths:
        return [src, dst]

    idx = np.random.randint(0, len(candidate_paths))
    return candidate_paths[idx]
