#!/usr/bin/env python3
"""
Multi-Topology Graph & Telemetry Library (EC499)
Provides standardized network graph topologies for zero-shot "blind" transfer
and stress testing of Deep Reinforcement Learning SDN Traffic Engineering agents:
  1. Hierarchical Tree (7 switches, 8 hosts, redundant mesh)
  2. Fat-Tree (k=4: 20 switches, 16 hosts, multi-stage Clos fabric)
  3. Abilene / Internet2 Network (12 nodes, 30 directed links, US WAN backbone)
  4. NSFNet Core Mesh (14 nodes, 42 directed links, continental mesh)
  5. Spine-Leaf Data Center Fabric (4 Spines, 8 Leaves, 64 directed links)
"""

import networkx as nx

def build_hierarchical_tree():
    """Hierarchical 7-switch Tree with lateral cross-links."""
    g = nx.DiGraph()
    for i in range(1, 8):
        g.add_node(i, label=f"s{i}", type="core" if i == 1 else ("agg" if i in [2, 3] else "edge"))

    links = [
        (1, 2, 100.0, 2.0), (2, 1, 100.0, 2.0),
        (1, 3, 100.0, 2.0), (3, 1, 100.0, 2.0),
        (2, 4, 50.0, 3.0),  (4, 2, 50.0, 3.0),
        (2, 5, 50.0, 3.0),  (5, 2, 50.0, 3.0),
        (3, 6, 50.0, 3.0),  (6, 3, 50.0, 3.0),
        (3, 7, 50.0, 3.0),  (7, 3, 50.0, 3.0),
        # Lateral redundant cross links
        (4, 6, 30.0, 8.0),  (6, 4, 30.0, 8.0),
        (5, 7, 30.0, 8.0),  (7, 5, 30.0, 8.0),
    ]
    for u, v, bw, lat in links:
        g.add_edge(u, v, capacity=bw, delay=lat, util=0.05, loss=0.0)

    # 2D coordinates for visual canvas rendering
    positions = {
        1: {'x': 0.50, 'y': 0.16},
        2: {'x': 0.30, 'y': 0.44},
        3: {'x': 0.70, 'y': 0.44},
        4: {'x': 0.16, 'y': 0.78},
        5: {'x': 0.38, 'y': 0.78},
        6: {'x': 0.62, 'y': 0.78},
        7: {'x': 0.84, 'y': 0.78}
    }
    hosts = [
        {'id': 'h1', 'ip': '10.0.0.1', 'mac': '00:00:00:00:00:01', 'switch': 4, 'port': 1},
        {'id': 'h2', 'ip': '10.0.0.2', 'mac': '00:00:00:00:00:02', 'switch': 4, 'port': 2},
        {'id': 'h3', 'ip': '10.0.0.3', 'mac': '00:00:00:00:00:03', 'switch': 5, 'port': 1},
        {'id': 'h4', 'ip': '10.0.0.4', 'mac': '00:00:00:00:00:04', 'switch': 5, 'port': 2},
        {'id': 'h5', 'ip': '10.0.0.5', 'mac': '00:00:00:00:00:05', 'switch': 6, 'port': 1},
        {'id': 'h6', 'ip': '10.0.0.6', 'mac': '00:00:00:00:00:06', 'switch': 6, 'port': 2},
        {'id': 'h7', 'ip': '10.0.0.7', 'mac': '00:00:00:00:00:07', 'switch': 7, 'port': 1},
        {'id': 'h8', 'ip': '10.0.0.8', 'mac': '00:00:00:00:00:08', 'switch': 7, 'port': 2},
    ]
    meta = {
        'name': 'Hierarchical Tree (Baseline)',
        'id': 'tree',
        'switches': 7,
        'links': 8,
        'positions': positions,
        'hosts': hosts,
        'core_nodes': [1],
        'edge_nodes': [4, 5, 6, 7]
    }
    return g, meta


def build_fattree_k4():
    """Fat-Tree (k=4): 20 switches (4 Core, 8 Agg, 8 Edge), 32 inter-switch links."""
    g = nx.DiGraph()
    k = 4
    num_cores = (k // 2) ** 2  # 4
    num_pods = k              # 4

    # Cores: IDs 101-104
    cores = [100 + i for i in range(1, num_cores + 1)]
    for c in cores:
        g.add_node(c, label=f"c{c-100}", type="core")

    # Aggs (201-208) and Edges (301-308) in 4 Pods
    aggs = []
    edges = []
    positions = {}

    # Core coordinates
    for i, c in enumerate(cores):
        positions[c] = {'x': 0.25 + i * 0.165, 'y': 0.15}

    pod_step = 0.22
    for pod in range(num_pods):
        p_base_x = 0.12 + pod * pod_step
        pod_aggs = []
        pod_edges = []
        for a in range(k // 2):
            sw_id = 200 + (pod * 2 + a + 1)
            pod_aggs.append(sw_id)
            g.add_node(sw_id, label=f"a{sw_id-200}", type="agg")
            positions[sw_id] = {'x': p_base_x + a * 0.08, 'y': 0.45}

        for e in range(k // 2):
            sw_id = 300 + (pod * 2 + e + 1)
            pod_edges.append(sw_id)
            g.add_node(sw_id, label=f"e{sw_id-300}", type="edge")
            positions[sw_id] = {'x': p_base_x + e * 0.08, 'y': 0.78}

        aggs.extend(pod_aggs)
        edges.extend(pod_edges)

        # Intra-pod bipartite links: connect each Agg to each Edge in Pod
        for a_sw in pod_aggs:
            for e_sw in pod_edges:
                g.add_edge(a_sw, e_sw, capacity=100.0, delay=1.5, util=0.05, loss=0.0)
                g.add_edge(e_sw, a_sw, capacity=100.0, delay=1.5, util=0.05, loss=0.0)

        # Connect Core to Agg
        for i_a, a_sw in enumerate(pod_aggs):
            for j in range(k // 2):
                c_idx = i_a * (k // 2) + j
                c_sw = cores[c_idx]
                g.add_edge(c_sw, a_sw, capacity=100.0, delay=1.0, util=0.05, loss=0.0)
                g.add_edge(a_sw, c_sw, capacity=100.0, delay=1.0, util=0.05, loss=0.0)

    # 16 hosts attached to edges
    hosts = []
    for i, e_sw in enumerate(edges):
        for h_idx in range(1, 3):
            h_num = i * 2 + h_idx
            hosts.append({
                'id': f'h{h_num}',
                'ip': f'10.0.{i+1}.{h_idx}',
                'mac': f'00:00:00:00:00:{h_num:02x}',
                'switch': e_sw,
                'port': h_idx
            })

    meta = {
        'name': 'Fat-Tree Data Center Fabric (k=4)',
        'id': 'fattree',
        'switches': len(g.nodes()),
        'links': len(g.edges()) // 2,
        'positions': positions,
        'hosts': hosts,
        'core_nodes': cores,
        'edge_nodes': edges
    }
    return g, meta


def build_abilene_topology():
    """Abilene / Internet2 US Backbone Network: 12 nodes, 30 directed links."""
    g = nx.DiGraph()
    cities = {
        1: ("Seattle", 0.12, 0.20),
        2: ("Sunnyvale", 0.10, 0.50),
        3: ("Los Angeles", 0.14, 0.75),
        4: ("Salt Lake City", 0.28, 0.35),
        5: ("Denver", 0.40, 0.42),
        6: ("Kansas City", 0.55, 0.48),
        7: ("Houston", 0.54, 0.82),
        8: ("Chicago", 0.65, 0.32),
        9: ("Indianapolis", 0.68, 0.46),
        10: ("Atlanta", 0.74, 0.72),
        11: ("Washington DC", 0.86, 0.48),
        12: ("New York", 0.88, 0.30)
    }
    positions = {}
    for node_id, (name, x, y) in cities.items():
        g.add_node(node_id, label=name, type="wan_node")
        positions[node_id] = {'x': x, 'y': y}

    # Inter-city links with approximate geographical delays (ms) and 100-200 Mbps capacity
    undirected_links = [
        (1, 2, 100.0, 4.0),   # Seattle - Sunnyvale
        (1, 4, 100.0, 5.0),   # Seattle - Salt Lake City
        (2, 3, 150.0, 2.5),   # Sunnyvale - Los Angeles
        (2, 4, 100.0, 4.5),   # Sunnyvale - Salt Lake City
        (3, 7, 100.0, 8.5),   # Los Angeles - Houston
        (4, 5, 100.0, 3.5),   # Salt Lake City - Denver
        (5, 6, 150.0, 4.0),   # Denver - Kansas City
        (6, 8, 150.0, 3.5),   # Kansas City - Chicago
        (6, 9, 100.0, 3.8),   # Kansas City - Indianapolis
        (7, 10, 100.0, 5.5),  # Houston - Atlanta
        (8, 9, 150.0, 2.0),   # Chicago - Indianapolis
        (8, 12, 200.0, 6.0),  # Chicago - New York
        (9, 10, 100.0, 4.2),  # Indianapolis - Atlanta
        (10, 11, 150.0, 4.0), # Atlanta - Washington DC
        (11, 12, 200.0, 2.5)  # Washington DC - New York
    ]
    for u, v, bw, lat in undirected_links:
        g.add_edge(u, v, capacity=bw, delay=lat, util=0.05, loss=0.0)
        g.add_edge(v, u, capacity=bw, delay=lat, util=0.05, loss=0.0)

    hosts = [
        {'id': f'h{i}', 'ip': f'10.0.{i}.1', 'mac': f'00:00:00:00:00:{i:02x}', 'switch': i, 'port': 1}
        for i in range(1, 13)
    ]
    meta = {
        'name': 'Abilene Internet2 US Backbone',
        'id': 'abilene',
        'switches': 12,
        'links': 15,
        'positions': positions,
        'hosts': hosts,
        'core_nodes': [6, 8, 9], # Central crossroads
        'edge_nodes': [1, 2, 3, 7, 10, 11, 12]
    }
    return g, meta


def build_nsfnet_topology():
    """NSFNet 14-Node Continental Backbone Mesh."""
    g = nx.DiGraph()
    coords = {
        1: (0.08, 0.28), 2: (0.16, 0.42), 3: (0.22, 0.70), 4: (0.32, 0.45),
        5: (0.42, 0.55), 6: (0.50, 0.25), 7: (0.52, 0.78), 8: (0.64, 0.40),
        9: (0.66, 0.65), 10: (0.78, 0.28), 11: (0.80, 0.50), 12: (0.82, 0.75),
        13: (0.90, 0.38), 14: (0.92, 0.60)
    }
    positions = {}
    for node_id, (x, y) in coords.items():
        g.add_node(node_id, label=f"N{node_id}", type="wan_mesh")
        positions[node_id] = {'x': x, 'y': y}

    undirected_links = [
        (1, 2, 100.0, 3.5), (1, 3, 100.0, 5.0), (1, 4, 100.0, 4.0),
        (2, 3, 100.0, 3.0), (2, 8, 150.0, 7.0),
        (3, 7, 100.0, 6.0),
        (4, 5, 100.0, 2.5), (4, 9, 100.0, 5.5),
        (5, 6, 100.0, 3.0), (5, 7, 100.0, 3.5),
        (6, 8, 150.0, 3.0), (6, 10, 150.0, 4.5),
        (7, 9, 100.0, 3.5),
        (8, 11, 150.0, 3.5),
        (9, 12, 100.0, 4.0),
        (10, 11, 150.0, 3.0), (10, 13, 150.0, 2.5),
        (11, 12, 100.0, 3.5), (11, 14, 150.0, 3.0),
        (12, 14, 100.0, 2.5),
        (13, 14, 150.0, 2.0)
    ]
    for u, v, bw, lat in undirected_links:
        g.add_edge(u, v, capacity=bw, delay=lat, util=0.05, loss=0.0)
        g.add_edge(v, u, capacity=bw, delay=lat, util=0.05, loss=0.0)

    hosts = [
        {'id': f'h{i}', 'ip': f'10.0.{i}.1', 'mac': f'00:00:00:00:00:{i:02x}', 'switch': i, 'port': 1}
        for i in range(1, 15)
    ]
    meta = {
        'name': 'NSFNet 14-Node Continental Mesh',
        'id': 'nsfnet',
        'switches': 14,
        'links': 21,
        'positions': positions,
        'hosts': hosts,
        'core_nodes': [6, 8, 10, 11],
        'edge_nodes': [1, 2, 3, 7, 12, 13, 14]
    }
    return g, meta


def build_spine_leaf_fabric():
    """Spine-Leaf Clos Fabric: 4 Spines, 8 Leaves, full bipartite mesh (64 directed edges)."""
    g = nx.DiGraph()
    spines = [1, 2, 3, 4]
    leaves = [5, 6, 7, 8, 9, 10, 11, 12]
    positions = {}

    for i, s in enumerate(spines):
        g.add_node(s, label=f"Spine-{s}", type="spine")
        positions[s] = {'x': 0.25 + i * 0.165, 'y': 0.22}

    for i, l in enumerate(leaves):
        g.add_node(l, label=f"Leaf-{l-4}", type="leaf")
        positions[l] = {'x': 0.10 + i * 0.114, 'y': 0.72}

    # Bipartite interconnect: every spine connects to every leaf
    for s in spines:
        for l in leaves:
            g.add_edge(s, l, capacity=100.0, delay=1.0, util=0.05, loss=0.0)
            g.add_edge(l, s, capacity=100.0, delay=1.0, util=0.05, loss=0.0)

    hosts = [
        {'id': f'h{l-4}', 'ip': f'10.0.{l-4}.1', 'mac': f'00:00:00:00:00:{l-4:02x}', 'switch': l, 'port': 1}
        for l in leaves
    ]
    meta = {
        'name': 'Spine-Leaf Data Center Clos Fabric',
        'id': 'spineleaf',
        'switches': 12,
        'links': 32, # 32 bidirectional / 64 directed
        'positions': positions,
        'hosts': hosts,
        'core_nodes': spines,
        'edge_nodes': leaves
    }
    return g, meta


def build_random_topology(num_nodes=16, p_edge=0.25, seed=None):
    """Generates an arbitrary connected random network mesh for true blind testing."""
    import random
    if seed is not None:
        random.seed(seed)
    # Generate connected Watts-Strogatz small-world mesh
    g_undir = nx.connected_watts_strogatz_graph(n=num_nodes, k=4, p=p_edge, seed=seed)
    g = nx.DiGraph()
    for n in g_undir.nodes():
        node_id = n + 1
        g.add_node(node_id, label=f"r{node_id}", type="core" if n < 4 else "edge")

    positions = {}
    for n in range(num_nodes):
        positions[n + 1] = {
            'x': round(random.uniform(0.1, 0.9), 2),
            'y': round(random.uniform(0.1, 0.9), 2)
        }

    for u, v in g_undir.edges():
        bw = random.choice([50.0, 100.0, 150.0, 200.0])
        lat = round(random.uniform(2.0, 10.0), 1)
        g.add_edge(u + 1, v + 1, capacity=bw, delay=lat, util=0.05, loss=0.0)
        g.add_edge(v + 1, u + 1, capacity=bw, delay=lat, util=0.05, loss=0.0)

    hosts = [
        {'id': f'h{i}', 'ip': f'10.0.{i}.1', 'mac': f'00:00:00:00:00:{i:02x}', 'switch': i, 'port': 1}
        for i in range(5, num_nodes + 1)
    ]
    meta = {
        'name': f'Random Dynamic Mesh (N={num_nodes})',
        'id': 'random',
        'switches': num_nodes,
        'links': g_undir.number_of_edges(),
        'positions': positions,
        'hosts': hosts,
        'core_nodes': [1, 2, 3, 4],
        'edge_nodes': list(range(5, num_nodes + 1))
    }
    return g, meta


ALL_TOPOLOGY_BUILDERS = {
    'tree': build_hierarchical_tree,
    'fattree': build_fattree_k4,
    'abilene': build_abilene_topology,
    'nsfnet': build_nsfnet_topology,
    'spineleaf': build_spine_leaf_fabric
}

def get_topology(topo_id='tree'):
    """Returns (graph, metadata) for a given topology identifier."""
    if topo_id.lower() == 'random':
        return build_random_topology(num_nodes=20, p_edge=0.28)
    builder = ALL_TOPOLOGY_BUILDERS.get(topo_id.lower(), build_hierarchical_tree)
    return builder()

def list_available_topologies():
    """Returns a list of metadata for all registered topologies."""
    res = []
    for tid, builder in ALL_TOPOLOGY_BUILDERS.items():
        _, meta = builder()
        res.append({
            'id': tid,
            'name': meta['name'],
            'switches': meta['switches'],
            'links': meta.get('links', 0),
            'hosts': len(meta.get('hosts', []))
        })
    return res


if __name__ == '__main__':
    print("=" * 75)
    print(" 🌐 EC499 MULTI-TOPOLOGY GRAPH & TELEMETRY LIBRARY")
    print("=" * 75)
    topos = list_available_topologies()
    print(f"{'ID':<12} | {'Topology Name':<32} | {'Switches':<9} | {'Links':<7} | {'Hosts'}")
    print("-" * 75)
    for t in topos:
        print(f"{t['id']:<12} | {t['name']:<32} | {t['switches']:<9} | {t['links']:<7} | {t['hosts']}")
    print("-" * 75)
    print("Use in tournament: python3 benchmark_routing_algorithms.py")
    print("Use in stress-test: python3 stress_test_blind_topologies.py")
    print("=" * 75)

