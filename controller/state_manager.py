import numpy as np
import networkx as nx
import math
import time
import collections

class StateManager:
    """
    Centralized State and Telemetry Repository for Adaptive SDN Traffic Engineering.
    Maintains network topology, link metrics, port/flow differential rates,
    host locations, multicast group memberships, and security feature representations.
    """
    def __init__(self):
        # Topology graph: nodes are switch DPIDs, edges have 'port', 'bandwidth', 'delay', 'loss'
        self.graph = nx.DiGraph()
        
        # Link metric dictionaries: keyed by (src_dpid, dst_dpid)
        self.link_bandwidths = {}   # Capacity in Mbps (default: 100 Mbps)
        self.link_utilization = {}  # Current utilization [0.0 - 1.0]
        self.link_delays = {}       # Latency in ms (default: 2.0 ms)
        self.link_loss = {}         # Packet loss rate [0.0 - 1.0]
        
        # Host tracking tables: enables dynamic ARP/L2/L3 resolution
        self.host_ip_to_mac = {}    # ip -> mac
        self.host_locations = {}    # mac -> (dpid, port)
        self.ip_to_location = {}    # ip -> (dpid, port)
        self.mac_to_port = {}       # (dpid, mac) -> port
        
        # Server resource utilization: keyed by dpid or host_ip
        self.server_cpu = collections.defaultdict(float)  # [0.0 - 1.0]
        self.server_ram = collections.defaultdict(float)  # [0.0 - 1.0]
        
        # Telemetry history for rate calculation
        self.raw_port_stats = {}    # (dpid, port) -> {'rx_bytes', 'tx_bytes', 'rx_pkts', 'tx_pkts', 'timestamp'}
        self.port_rates = {}        # (dpid, port) -> {'rx_mbps', 'tx_mbps', 'rx_pps', 'tx_pps'}
        
        self.raw_flow_stats = {}    # (dpid, src_ip, dst_ip) -> {'bytes', 'packets', 'duration', 'timestamp'}
        self.flow_stats = {}        # (dpid, src_ip, dst_ip) -> {'mbps', 'pps', 'packets', 'bytes', 'duration'}
        
        # Traffic entropy and rate counters for DDoS detection
        self.src_ip_counter = collections.defaultdict(int)
        self.dst_ip_counter = collections.defaultdict(int)
        self.dst_port_counter = collections.defaultdict(int)
        self.last_security_sample_time = time.time()
        self.recent_packet_count = 0
        self.recent_byte_count = 0
        self.cached_security_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Link failure recovery tracking: (u, v) -> edge_attributes
        self.failed_links = {}
        
        # Rolling time-series telemetry history (up to 60 data points for live web charts)
        self.telemetry_history = collections.deque(maxlen=60)
        self.active_simulation_mode = None
        
        # Multicast group registry: group_ip -> set of (receiver_dpid, receiver_port)
        self.multicast_groups = collections.defaultdict(set)
        
        # Routing candidate path cache: (src_dpid, dst_dpid) -> list of candidate paths
        self._routing_path_cache = {}

    # --------------------------------------------------------------------------
    # Topology & Host Management
    # --------------------------------------------------------------------------

    def register_switch(self, dpid):
        """Registers a switch datapath node in the topology graph."""
        if not self.graph.has_node(dpid):
            self.graph.add_node(dpid)

    def unregister_switch(self, dpid):
        """Removes a switch datapath node from the topology graph."""
        if self.graph.has_node(dpid):
            self.graph.remove_node(dpid)
            self._routing_path_cache.clear()

    def update_link(self, src_dpid, dst_dpid, src_port, dst_port=None, capacity_mbps=100.0, delay_ms=2.0, loss=0.0):
        """Adds or updates an inter-switch directional link."""
        self.register_switch(src_dpid)
        self.register_switch(dst_dpid)
        
        self.graph.add_edge(src_dpid, dst_dpid, port=src_port, peer_port=dst_port)
        self._routing_path_cache.clear()
        self.link_bandwidths[(src_dpid, dst_dpid)] = capacity_mbps
        if (src_dpid, dst_dpid) not in self.link_delays:
            self.link_delays[(src_dpid, dst_dpid)] = delay_ms
        if (src_dpid, dst_dpid) not in self.link_loss:
            self.link_loss[(src_dpid, dst_dpid)] = loss
        if (src_dpid, dst_dpid) not in self.link_utilization:
            self.link_utilization[(src_dpid, dst_dpid)] = 0.0

    def record_host(self, ip, mac, dpid, port):
        """Records host location and mapping for dynamic packet routing."""
        # Never record a host on an inter-switch trunk port
        if self.graph.has_node(dpid):
            for _, _, data in self.graph.out_edges(dpid, data=True):
                if data.get('port') == port:
                    return  # Packet arrived on a trunk port connecting two switches, ignore

        if ip:
            self.host_ip_to_mac[ip] = mac
            self.ip_to_location[ip] = (dpid, port)
        if mac:
            self.host_locations[mac] = (dpid, port)
            self.mac_to_port[(dpid, mac)] = port

    def clear_hosts(self):
        """Clears learned host locations."""
        self.host_ip_to_mac.clear()
        self.host_locations.clear()
        self.ip_to_location.clear()
        self.mac_to_port.clear()

    def get_host_location(self, ip=None, mac=None):
        """Returns (dpid, port) for a given IP or MAC address."""
        if ip and ip in self.ip_to_location:
            return self.ip_to_location[ip]
        if mac and mac in self.host_locations:
            return self.host_locations[mac]
        if ip and ip in self.host_ip_to_mac:
            m = self.host_ip_to_mac[ip]
            return self.host_locations.get(m, (None, None))
        return (None, None)

    # --------------------------------------------------------------------------
    # Telemetry & Differential Statistics
    # --------------------------------------------------------------------------

    def update_port_stats(self, dpid, port_no, rx_bytes, tx_bytes, rx_packets, tx_packets, duration_sec):
        """Updates port stats and computes differential Mbps and PPS rates."""
        now = time.time()
        key = (dpid, port_no)
        
        if key in self.raw_port_stats:
            prev = self.raw_port_stats[key]
            dt = max(0.1, now - prev['timestamp'])
            
            delta_rx_bytes = max(0, rx_bytes - prev['rx_bytes'])
            delta_tx_bytes = max(0, tx_bytes - prev['tx_bytes'])
            delta_rx_pkts = max(0, rx_packets - prev['rx_pkts'])
            delta_tx_pkts = max(0, tx_packets - prev['tx_pkts'])
            
            # Rate in Mbps: (bytes * 8) / (1e6 * dt)
            rx_mbps = (delta_rx_bytes * 8.0) / (1e6 * dt)
            tx_mbps = (delta_tx_bytes * 8.0) / (1e6 * dt)
            rx_pps = delta_rx_pkts / dt
            tx_pps = delta_tx_pkts / dt
            
            self.port_rates[key] = {
                'rx_mbps': rx_mbps,
                'tx_mbps': tx_mbps,
                'rx_pps': rx_pps,
                'tx_pps': tx_pps
            }
            
            # Update link utilization if this port connects to an adjacent switch
            for neighbor in self.graph.successors(dpid):
                edge_data = self.graph.get_edge_data(dpid, neighbor)
                if edge_data and edge_data.get('port') == port_no:
                    capacity = self.link_bandwidths.get((dpid, neighbor), 100.0)
                    self.link_utilization[(dpid, neighbor)] = min(1.0, tx_mbps / max(1.0, capacity))
        
        self.raw_port_stats[key] = {
            'rx_bytes': rx_bytes,
            'tx_bytes': tx_bytes,
            'rx_pkts': rx_packets,
            'tx_pkts': tx_packets,
            'duration': duration_sec,
            'timestamp': now
        }

    def update_flow_stats(self, dpid, src_ip, dst_ip, packets, bytes_count, duration_sec):
        """Updates flow counters and security metrics."""
        now = time.time()
        key = (dpid, src_ip, dst_ip)
        
        delta_packets = packets
        delta_bytes = bytes_count
        
        if key in self.raw_flow_stats:
            prev = self.raw_flow_stats[key]
            dt = max(0.1, now - prev['timestamp'])
            delta_packets = max(0, packets - prev['packets'])
            delta_bytes = max(0, bytes_count - prev['bytes'])
            
            mbps = (delta_bytes * 8.0) / (1e6 * dt)
            pps = delta_packets / dt
        else:
            mbps = 0.0
            pps = 0.0
            
        self.raw_flow_stats[key] = {
            'packets': packets,
            'bytes': bytes_count,
            'duration': duration_sec,
            'timestamp': now
        }
        
        self.flow_stats[key] = {
            'mbps': mbps,
            'pps': pps,
            'packets': packets,
            'bytes': bytes_count,
            'duration': duration_sec
        }
        
        if src_ip:
            self.src_ip_counter[src_ip] += delta_packets
            self.recent_packet_count += delta_packets
            self.recent_byte_count += delta_bytes
        if dst_ip:
            self.dst_ip_counter[dst_ip] += delta_packets

    def update_link_latency(self, src_dpid, dst_dpid, delay_ms):
        """Updates measured link latency (RTT probe)."""
        self.link_delays[(src_dpid, dst_dpid)] = max(0.1, delay_ms)

    # --------------------------------------------------------------------------
    # Multicast Group Management
    # --------------------------------------------------------------------------

    def register_multicast_member(self, group_ip, dpid, port):
        """Registers a host switch-port as a listener for a multicast IP."""
        self.multicast_groups[group_ip].add((dpid, port))

    def unregister_multicast_member(self, group_ip, dpid, port):
        """Removes a listener from a multicast group."""
        if (dpid, port) in self.multicast_groups[group_ip]:
            self.multicast_groups[group_ip].remove((dpid, port))

    def get_multicast_receivers(self, group_ip):
        """Returns set of (dpid, port) listening to group_ip."""
        return self.multicast_groups.get(group_ip, set())

    # --------------------------------------------------------------------------
    # Feature Extractors for RL Agents
    # --------------------------------------------------------------------------

    def get_candidate_paths(self, src_dpid, dst_dpid, k=4):
        """
        Discovers up to k diversity-aware candidate paths between source and destination:
        - Path 0: Dijkstra Shortest Path (pure hop count)
        - Path 1: 2nd Shortest Simple Path
        - Path 2: Diverse / Low-Overlap Bypass Path (routes around primary path edges)
        - Path 3: Widest Path (minimizes peak link utilization)
        """
        if not self.graph.has_node(src_dpid) or not self.graph.has_node(dst_dpid):
            return [[src_dpid, dst_dpid]]

        cache_key = (src_dpid, dst_dpid)
        cached = self._routing_path_cache.get(cache_key)
        if cached is not None:
            return cached

        paths = []
        try:
            import itertools
            simple_paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, src_dpid, dst_dpid), 12))
            if not simple_paths:
                paths = [[src_dpid, dst_dpid]]
            else:
                p0 = simple_paths[0]
                paths.append(p0)

                # Path 1: Next shortest simple path
                if len(simple_paths) > 1:
                    paths.append(simple_paths[1])

                # Path 2: Disjoint / core-bypass path (minimizing edge overlap with p0)
                p0_edges = set((p0[idx], p0[idx+1]) for idx in range(len(p0)-1))
                p0_edges.update((p0[idx+1], p0[idx]) for idx in range(len(p0)-1))
                diverse_candidates = [p for p in simple_paths if p not in paths]
                if diverse_candidates:
                    diverse_candidates.sort(key=lambda p: sum(1 for idx in range(len(p)-1) if (p[idx], p[idx+1]) in p0_edges))
                    paths.append(diverse_candidates[0])

                # Path 3: Widest path (lowest bottleneck utilization)
                remaining = [p for p in simple_paths if p not in paths]
                if remaining:
                    remaining.sort(key=lambda p: max([self.link_utilization.get((p[idx], p[idx+1]), 0.0) for idx in range(len(p)-1)] + [0.0]))
                    paths.append(remaining[0])

                # Fill remaining slots up to k
                for p in simple_paths:
                    if len(paths) >= k:
                        break
                    if p not in paths:
                        paths.append(p)
        except Exception:
            paths = [[src_dpid, dst_dpid]]

        if not paths:
            paths = [[src_dpid, dst_dpid]]

        self._routing_path_cache[cache_key] = paths[:k]
        return self._routing_path_cache[cache_key]

    def get_routing_state(self, src_dpid, dst_dpid):
        """
        Builds a normalized 10-dimensional state vector for Unicast Double DQN Agent.
        [0]: Normalized shortest path length (hops / 10.0) in (0, 1]
        [1-4]: Bottleneck link utilization of Candidate Paths 0, 1, 2, 3 [0.0 - 1.0]
        [5-8]: Normalized end-to-end latency of Candidate Paths 0, 1, 2, 3 [0.0 - 1.0] (delay / 50.0)
        [9]: Network-wide peak link utilization [0.0 - 1.0]
        """
        state = np.zeros(10, dtype=np.float32)
        if not self.graph.has_node(src_dpid) or not self.graph.has_node(dst_dpid):
            return state

        # Discover or retrieve cached diversity-aware candidate paths (up to 4)
        candidate_paths = self.get_candidate_paths(src_dpid, dst_dpid, k=4)

        # 0: Shortest path hops normalized
        if candidate_paths and len(candidate_paths[0]) > 1:
            hops = len(candidate_paths[0]) - 1
            state[0] = min(1.0, max(0.1, hops / 10.0))
        else:
            state[0] = 0.1

        # 1-4: Bottleneck link utilization on each candidate path
        # 5-8: Normalized delay on each candidate path
        for i in range(4):
            if i < len(candidate_paths):
                p = candidate_paths[i]
                if len(p) > 1:
                    utils = [self.link_utilization.get((p[j], p[j+1]), 0.0) for j in range(len(p)-1)]
                    delays = [self.link_delays.get((p[j], p[j+1]), 2.0) for j in range(len(p)-1)]
                    state[1 + i] = min(1.0, max(0.0, float(max(utils) if utils else 0.0)))
                    state[5 + i] = min(1.0, max(0.0, float(sum(delays) / 50.0)))
                else:
                    state[1 + i] = 0.0
                    state[5 + i] = 0.04
            else:
                # Nonexistent candidate path: penalized so agent prefers available paths
                state[1 + i] = 1.0
                state[5 + i] = 1.0

        # 9: Peak network utilization
        all_utils = list(self.link_utilization.values())
        state[9] = min(1.0, max(0.0, float(max(all_utils)))) if all_utils else 0.0

        return state

    def get_multicast_state(self, src_dpid, group_destinations):
        """
        Builds a normalized 50-dimensional state vector for Multicast Dueling DQN.
        [0]: Normalized Source DPID
        [1-5]: Up to 5 normalized Destination DPIDs
        [6-8]: Global avg utilization, avg delay, avg loss
        [9-47]: Flattened link features (util, delay, loss) for up to 13 topology edges
        [48-49]: Active receiver count & total multicast load
        """
        state = np.zeros(50, dtype=np.float32)
        num_nodes = max(1, self.graph.number_of_nodes())
        
        state[0] = (src_dpid % num_nodes) / float(num_nodes)
        for i, dst in enumerate(group_destinations[:5]):
            state[i + 1] = (dst % num_nodes) / float(num_nodes)

        # Global statistics
        utils = list(self.link_utilization.values())
        delays = list(self.link_delays.values())
        losses = list(self.link_loss.values())

        state[6] = float(np.mean(utils)) if utils else 0.0
        state[7] = min(1.0, float(np.mean(delays)) / 50.0) if delays else 0.04
        state[8] = float(np.mean(losses)) if losses else 0.0

        # Sample edge features for up to 13 links
        edges = list(self.graph.edges())
        for i, edge in enumerate(edges[:13]):
            idx = 9 + (i * 3)
            state[idx] = self.link_utilization.get(edge, 0.0)
            state[idx + 1] = min(1.0, self.link_delays.get(edge, 2.0) / 50.0)
            state[idx + 2] = self.link_loss.get(edge, 0.0)

        state[48] = min(1.0, len(group_destinations) / 10.0)
        state[49] = min(1.0, float(np.sum(utils)) / 10.0) if utils else 0.0

        return state

    def update_security_window(self):
        """Called periodically by monitor loop to compute Shannon entropy, rates, and record history."""
        now = time.time()
        dt = max(0.5, now - self.last_security_sample_time)
        
        total_pkts = self.recent_packet_count
        total_bytes = self.recent_byte_count
        
        pps = total_pkts / dt
        bpp = (total_bytes / max(1, total_pkts)) if total_pkts > 0 else 0.0
        
        # Shannon Entropy
        entropy = 0.0
        if total_pkts > 0:
            for count in self.src_ip_counter.values():
                if count > 0:
                    p = count / float(total_pkts)
                    entropy -= p * math.log2(p)
                    
        max_possible_entropy = math.log2(max(2, len(self.src_ip_counter)))
        normalized_entropy = min(1.0, entropy / max(1.0, max_possible_entropy)) if total_pkts > 10 else 1.0

        num_flows = len(self.flow_stats)
        
        self.cached_security_state = np.array([
            min(1.0, pps / 5000.0),
            normalized_entropy,
            min(1.0, bpp / 1500.0),
            min(1.0, num_flows / 100.0),
            min(1.0, (pps * bpp) / 1e7)
        ], dtype=np.float32)

        # Record telemetry history for dynamic real-time dashboard charts
        total_mbps = sum(p.get('tx_mbps', 0.0) for p in self.port_rates.values())
        avg_lat = float(np.mean(list(self.link_delays.values()))) if self.link_delays else 2.0
        max_u = float(np.max(list(self.link_utilization.values()))) if self.link_utilization else 0.0
        self.telemetry_history.append({
            'timestamp': time.strftime("%H:%M:%S", time.localtime(now)),
            'throughput_mbps': round(float(total_mbps), 2),
            'packet_rate_pps': round(float(pps), 1),
            'entropy': round(float(normalized_entropy), 3),
            'avg_latency_ms': round(float(avg_lat), 2),
            'max_link_utilization_pct': round(float(max_u * 100.0), 1),
            'active_flows': num_flows,
            'is_attack': bool(normalized_entropy < 0.45 and pps > 1200)
        })

        # Reset sample window
        self.last_security_sample_time = now
        self.recent_packet_count = 0
        self.recent_byte_count = 0
        self.src_ip_counter.clear()

    def get_security_state(self):
        """Returns the latest normalized 5-dimensional security state vector."""
        if self.recent_packet_count > 0:
            self.update_security_window()
        return self.cached_security_state

    # --------------------------------------------------------------------------
    # Interactive Simulation & Telemetry Injection
    # --------------------------------------------------------------------------

    def inject_traffic_flow(self, src_ip, dst_ip, src_dpid, dst_dpid, mbps=15.0, pps=1200.0, path=None):
        """Injects a simulated flow and updates link utilization along the chosen path."""
        duration = 10.0
        bytes_count = int((mbps * 1e6 * duration) / 8.0)
        packets = int(pps * duration)
        self.update_flow_stats(src_dpid, src_ip, dst_ip, packets, bytes_count, duration)

        if path and len(path) > 1:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                cap = self.link_bandwidths.get((u, v), 100.0)
                add_util = mbps / max(1.0, cap)
                self.link_utilization[(u, v)] = min(1.0, self.link_utilization.get((u, v), 0.0) + add_util)
                port = self.graph[u][v].get('port', 1) if self.graph.has_edge(u, v) else 1
                self.port_rates[(u, port)] = {
                    'rx_mbps': mbps, 'tx_mbps': mbps, 'rx_pps': pps, 'tx_pps': pps
                }

    def inject_ddos_flood(self, attacker_ip="10.0.0.4", target_ip="10.0.0.1", pps=4800, bpp=80):
        """Injects an intense single-source volumetric flood causing entropy collapse."""
        duration = 5.0
        total_packets = int(pps * duration)
        total_bytes = int(total_packets * bpp)
        self.update_flow_stats(5, attacker_ip, target_ip, total_packets, total_bytes, duration)
        self.active_simulation_mode = "ddos_flood"
        self.update_security_window()

    def inject_core_congestion(self, utilization=0.95):
        """Saturates core switch links to trigger lateral cross-link rerouting."""
        core_edges = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 4), (4, 2), (2, 5), (5, 2), (3, 6), (6, 3), (3, 7), (7, 3)]
        for u, v in core_edges:
            if (u, v) in self.link_utilization or self.graph.has_edge(u, v):
                self.link_utilization[(u, v)] = utilization
        # Keep lateral cross-links free
        for u, v in [(4, 6), (6, 4), (5, 7), (7, 5)]:
            if (u, v) in self.link_utilization or self.graph.has_edge(u, v):
                self.link_utilization[(u, v)] = 0.18
        self.active_simulation_mode = "core_jamming"

    def reset_simulation(self):
        """Resets link utilization and clears simulated telemetry."""
        for k in self.link_utilization:
            self.link_utilization[k] = 0.05
        self.port_rates.clear()
        self.flow_stats.clear()
        self.raw_flow_stats.clear()
        self.raw_port_stats.clear()
        self.restore_link()
        self.active_simulation_mode = None
        self.cached_security_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def calculate_multivector_entropy(self):
        """
        Calculates normalized multi-vector Shannon entropy across:
          - H_src: Source IP distribution (single-source vs distributed)
          - H_dst: Destination IP distribution (targeted victim vs broad sweep)
          - H_port: Destination port distribution (service targeted vs random)
        Returns structured analysis dict.
        """
        def calc_h(counter):
            total = sum(counter.values())
            if total <= 0:
                return 1.0
            h = 0.0
            for count in counter.values():
                if count > 0:
                    p = count / float(total)
                    h -= p * math.log2(p)
            max_h = math.log2(max(2, len(counter)))
            return min(1.0, max(0.0, h / max(1.0, max_h)))

        h_src = calc_h(self.src_ip_counter)
        h_dst = calc_h(self.dst_ip_counter)
        h_port = calc_h(self.dst_port_counter)

        if h_src < 0.40 and h_dst < 0.40:
            attack_type = "targeted_single_source"
        elif h_src > 0.60 and h_dst < 0.40:
            attack_type = "distributed_reflection_flood"
        elif h_src < 0.40 and h_dst > 0.60:
            attack_type = "scanning_or_random_sweep"
        else:
            attack_type = "nominal_or_diffused"

        return {
            'h_src': round(float(h_src), 3),
            'h_dst': round(float(h_dst), 3),
            'h_port': round(float(h_port), 3),
            'attack_type': attack_type,
            'is_anomalous': bool(h_src < 0.45 or h_dst < 0.45)
        }

    def inject_link_failure(self, u, v):
        """Simulates physical fiber cut or port down between switch u and v."""
        if not self.graph.has_edge(u, v):
            return False
        data_fwd = dict(self.graph[u][v])
        data_rev = dict(self.graph[v][u]) if self.graph.has_edge(v, u) else data_fwd
        self.failed_links[(u, v)] = data_fwd
        self.failed_links[(v, u)] = data_rev
        self.graph.remove_edge(u, v)
        if self.graph.has_edge(v, u):
            self.graph.remove_edge(v, u)
        self._routing_path_cache.clear()
        self.active_simulation_mode = f"link_failure_{u}_{v}"
        return True

    def restore_link(self, u=None, v=None):
        """Restores specific or all previously severed links."""
        restored = 0
        if u is not None and v is not None:
            pairs = [(u, v), (v, u)]
        else:
            pairs = list(self.failed_links.keys())

        for edge in pairs:
            if edge in self.failed_links:
                data = self.failed_links.pop(edge)
                src, dst = edge
                self.graph.add_edge(src, dst, **data)
                restored += 1

        if restored > 0:
            self._routing_path_cache.clear()
            if not self.failed_links:
                self.active_simulation_mode = None
        return restored
