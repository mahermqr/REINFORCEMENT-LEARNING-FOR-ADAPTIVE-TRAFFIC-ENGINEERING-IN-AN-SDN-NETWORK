"""
Traditional SDN Routing Algorithms Baseline Engine.
Implements classical network routing algorithms for head-to-head benchmarking:
 1. Dijkstra Shortest Path First (SPF)
 2. Equal-Cost Multi-Path (ECMP)
 3. Widest Shortest Path (WSP / CSPF)
 4. Least Loaded Routing (LLR)
 5. Random Path Routing
"""

import itertools
import networkx as nx
import numpy as np


def compute_path_metrics(path, link_utilization, link_delays):
    """
    Computes bottleneck utilization, total latency, and hop count for a path.
    """
    if not path or len(path) <= 1:
        return 0.0, 0.0, 0

    hops = len(path) - 1
    total_delay = 0.0
    bottleneck_util = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        util = link_utilization.get((u, v), 0.0)
        delay = link_delays.get((u, v), 2.0)
        bottleneck_util = max(bottleneck_util, util)
        total_delay += delay

    return float(bottleneck_util), float(total_delay), int(hops)


def dijkstra_spf(graph, src, dst, weight=None):
    """
    Standard Dijkstra Shortest Path First (SPF).
    Routes strictly along the path with minimal hop count or static weight.
    Susceptible to severe link jamming under concentrated traffic.
    """
    try:
        return list(nx.shortest_path(graph, src, dst, weight=weight))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [src, dst]


def ecmp_routing(graph, src, dst, flow_hash=0):
    """
    Equal-Cost Multi-Path (ECMP).
    Discovers all shortest paths of identical minimum hop count and hashes flows across them.
    Incapable of offloading onto slightly longer but uncongested paths.
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
    Evaluates candidate paths and selects the path with the largest residual bandwidth
    (minimum bottleneck link utilization). Breaks ties using shortest latency/hop count.
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
        b_util, t_delay, hops = compute_path_metrics(p, link_utilization, delays)
        # Score tuple: (bottleneck_util ascending, total_delay ascending, hops ascending)
        scored_paths.append((b_util, t_delay, hops, p))

    scored_paths.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored_paths[0][3]


def least_loaded_routing(graph, src, dst, link_utilization, candidate_paths=None):
    """
    Least Loaded Routing (LLR).
    Selects the path that minimizes the average link utilization across all hops.
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
        # Blend average and peak load
        load_metric = 0.6 * max_util + 0.4 * avg_util
        scored_paths.append((load_metric, p))

    scored_paths.sort(key=lambda item: item[0])
    return scored_paths[0][1]


def random_routing(graph, src, dst, candidate_paths=None):
    """
    Random Routing baseline. Uniform random choice across candidate paths.
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
