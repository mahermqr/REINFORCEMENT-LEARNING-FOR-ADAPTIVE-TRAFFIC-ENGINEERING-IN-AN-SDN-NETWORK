import numpy as np
import networkx as nx
import math
import time
import collections

class StateManager:
    """
    Centralized State and Telemetry Repository for Adaptive SDN Traffic Engineering (EC499).
    Maintains network topology, link metrics, port/flow differential rates,
    host locations, network jitter (RFC 3393), packet loss modeling, and OpenFlow control overhead.
    """
    def __init__(self):
        # Topology graph: nodes are switch DPIDs, edges have 'port', 'bandwidth', 'delay', 'loss'
        self.graph = nx.DiGraph()

        # Link metric dictionaries: keyed by (src_dpid, dst_dpid)
        self.link_bandwidths = {}   # Capacity in Mbps (default: 100 Mbps)
        self.link_utilization = {}  # Current utilization [0.0 - 1.0]
        self.base_link_delays = {}  # Static physical base latency in ms (uncongested)
        self.link_delays = {}       # Real-time queuing latency in ms
        self.link_jitter = {}       # Network Jitter / Delay Variation in ms (RFC 3393)
        self.link_loss = {}         # Packet loss rate percentage [0.0 - 100.0%]
        self._prev_delays = {}      # Previous delay sample for jitter calculation
        self.active_dynamic_flows = {} # flow_id -> {'path', 'mbps', 'expires_at'}

        # Host tracking tables: enables dynamic ARP/L2/L3 resolution
        self.host_ip_to_mac = {}    # ip -> mac
        self.host_locations = {}    # mac -> (dpid, port)
        self.ip_to_location = {}    # ip -> (dpid, port)
        self.mac_to_port = {}       # (dpid, mac) -> port

        # Telemetry history for rate calculation
        self.raw_port_stats = {}    # (dpid, port) -> {'rx_bytes', 'tx_bytes', 'rx_pkts', 'tx_pkts', 'timestamp'}
        self.port_rates = {}        # (dpid, port) -> {'rx_mbps', 'tx_mbps', 'rx_pps', 'tx_pps'}

        self.raw_flow_stats = {}    # (dpid, src_ip, dst_ip) -> {'bytes', 'packets', 'duration', 'timestamp'}
        self.flow_stats = {}        # (dpid, src_ip, dst_ip) -> {'mbps', 'pps', 'packets', 'bytes', 'duration'}

        # OpenFlow Control Overhead Accounting (Proposal Objective 5)
        self.packet_in_count = 0
        self.flow_mod_count = 0
        self.stats_request_count = 0
        self.stats_reply_count = 0
        self.control_overhead_bytes = 0
        self.controller_decision_times = collections.deque(maxlen=1000) # latencies in ms
        self.control_overhead_start_time = time.time()

        # Link failure recovery tracking: (u, v) -> edge_attributes
        self.failed_links = {}

        # Rolling time-series telemetry history (up to 60 data points for live web charts)
        self.telemetry_history = collections.deque(maxlen=60)
        self.active_simulation_mode = None

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
        self.link_bandwidths[(src_dpid, dst_dpid)] = float(capacity_mbps)
        self.base_link_delays[(src_dpid, dst_dpid)] = float(delay_ms)
        if (src_dpid, dst_dpid) not in self.link_delays:
            self.link_delays[(src_dpid, dst_dpid)] = float(delay_ms)
        if (src_dpid, dst_dpid) not in self.link_jitter:
            self.link_jitter[(src_dpid, dst_dpid)] = 0.25 # Nominal base jitter ms
        if (src_dpid, dst_dpid) not in self.link_loss:
            self.link_loss[(src_dpid, dst_dpid)] = float(loss)
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
        """Updates port stats and computes differential Mbps, PPS, link utilization, jitter, and loss."""
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
                    util = min(1.0, tx_mbps / max(1.0, capacity))
                    self.link_utilization[(dpid, neighbor)] = util

                    # Mathematical Packet Loss modeling (Objective 5)
                    # As link utilization approaches saturation, buffer queues overflow
                    if util <= 0.70:
                        loss_pct = util * 0.05
                    else:
                        excess = (util - 0.70) / 0.30
                        loss_pct = min(35.0, 0.05 + 30.0 * (excess ** 2.2))
                    self.link_loss[(dpid, neighbor)] = float(loss_pct)

        self.raw_port_stats[key] = {
            'rx_bytes': rx_bytes,
            'tx_bytes': tx_bytes,
            'rx_pkts': rx_packets,
            'tx_pkts': tx_packets,
            'duration': duration_sec,
            'timestamp': now
        }

    def update_flow_stats(self, dpid, src_ip, dst_ip, packets, bytes_count, duration_sec):
        """Updates flow counters and calculates throughput rates."""
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

    def update_link_latency(self, src_dpid, dst_dpid, delay_ms):
        """
        Updates measured link latency and calculates RFC 3393 / RFC 3550 Delay Variation (Jitter).
        Exponential moving average: J_k = J_{k-1} + (|D_k - D_{k-1}| - J_{k-1}) / 16
        """
        link = (src_dpid, dst_dpid)
        d_new = max(0.1, float(delay_ms))
        d_prev = self._prev_delays.get(link, d_new)
        self._prev_delays[link] = d_new
        self.link_delays[link] = d_new

        # Instantaneous delay variation
        d_diff = abs(d_new - d_prev)
        j_prev = self.link_jitter.get(link, 0.25)
        # RFC 3550 Jitter filter
        j_new = j_prev + (d_diff - j_prev) / 16.0

        # Account for queue congestion jitter scaling
        util = self.link_utilization.get(link, 0.0)
        if util > 0.60:
            congestion_boost = (util ** 2) / max(0.05, 1.0 - min(0.95, util))
            j_new += 0.15 * d_new * min(4.0, congestion_boost)

        self.link_jitter[link] = float(j_new)

    # --------------------------------------------------------------------------
    # Control Overhead Accounting (Proposal Objective 5)
    # --------------------------------------------------------------------------

    def record_packet_in(self, byte_size=64):
        """Records an incoming OpenFlow OFPPacketIn message."""
        self.packet_in_count += 1
        self.control_overhead_bytes += byte_size

    def record_flow_mod(self, byte_size=72):
        """Records an outgoing OpenFlow OFPFlowMod rule installation."""
        self.flow_mod_count += 1
        self.control_overhead_bytes += byte_size

    def record_stats_request(self, byte_size=56):
        """Records a stats query message."""
        self.stats_request_count += 1
        self.control_overhead_bytes += byte_size

    def record_stats_reply(self, byte_size=128):
        """Records a stats reply message."""
        self.stats_reply_count += 1
        self.control_overhead_bytes += byte_size

    def record_decision_latency(self, latency_ms):
        """Records the time taken by the controller / RL agent to infer an optimal path."""
        self.controller_decision_times.append(float(latency_ms))

    def get_control_overhead_summary(self):
        """
        Computes detailed quantitative OpenFlow control plane metrics.
        Returns:
          - packet_in_count: total PacketIn messages
          - flow_mod_count: total FlowMod installations
          - total_control_messages: PacketIn + FlowMod + Stats
          - mean_decision_latency_ms: average decision execution time
          - p95_decision_latency_ms: 95th percentile decision latency
          - decision_throughput_dps: decisions processed per second
          - total_control_overhead_kb: total control channel bytes in KB
        """
        dt = max(0.5, time.time() - self.control_overhead_start_time)
        times = list(self.controller_decision_times)
        mean_time = float(np.mean(times)) if times else 0.45
        p95_time = float(np.percentile(times, 95)) if times else 0.85
        total_msgs = self.packet_in_count + self.flow_mod_count + self.stats_request_count + self.stats_reply_count
        dps = (self.flow_mod_count / dt) if dt > 0 else 0.0

        return {
            'packet_in_count': int(self.packet_in_count),
            'flow_mod_count': int(self.flow_mod_count),
            'stats_request_count': int(self.stats_request_count),
            'stats_reply_count': int(self.stats_reply_count),
            'total_control_messages': int(total_msgs),
            'control_message_rate_mps': round(float(total_msgs / dt), 2),
            'mean_decision_latency_ms': round(mean_time, 3),
            'p95_decision_latency_ms': round(p95_time, 3),
            'decision_throughput_dps': round(float(dps), 1),
            'total_control_overhead_kb': round(float(self.control_overhead_bytes / 1024.0), 2)
        }

    # --------------------------------------------------------------------------
    # Aggregate Network Telemetry Summary (Proposal Objectives 3, 4, 5)
    # --------------------------------------------------------------------------

    def get_network_te_summary(self):
        """
        Calculates all key metrics defined in the EC499 Proposal:
          - Total Network Throughput (Mbps)
          - Bottleneck Link Utilization (%)
          - Mean Link Utilization (%)
          - Jain's Fairness Index across links
          - Mean Latency (ms)
          - Mean Jitter (ms)
          - Aggregate Packet Loss Rate (%)
          - Active Flows Count
          - Control Overhead Summary
        """
        # Throughput: aggregate of tx_mbps across ports
        total_mbps = sum(p.get('tx_mbps', 0.0) for p in self.port_rates.values())
        if total_mbps == 0.0 and self.flow_stats:
            total_mbps = sum(f.get('mbps', 0.0) for f in self.flow_stats.values())

        utils = list(self.link_utilization.values()) or [0.0]
        delays = list(self.link_delays.values()) or [2.0]
        jitters = list(self.link_jitter.values()) or [0.25]
        losses = list(self.link_loss.values()) or [0.0]

        max_u = float(np.max(utils))
        mean_u = float(np.mean(utils))

        # Jain's Fairness Index: (sum(u))^2 / (n * sum(u^2))
        u_arr = np.array(utils, dtype=np.float64)
        sum_u = np.sum(u_arr)
        sum_sq_u = np.sum(u_arr ** 2)
        n = len(u_arr)
        jains_index = float((sum_u ** 2) / max(1e-6, n * sum_sq_u)) if sum_sq_u > 0 else 1.0
        jains_index = min(1.0, max(0.0, jains_index))

        summary = {
            'total_throughput_mbps': round(float(total_mbps), 2),
            'bottleneck_utilization_pct': round(float(max_u * 100.0), 2),
            'mean_utilization_pct': round(float(mean_u * 100.0), 2),
            'jains_fairness_index': round(float(jains_index), 4),
            'mean_latency_ms': round(float(np.mean(delays)), 2),
            'mean_jitter_ms': round(float(np.mean(jitters)), 3),
            'mean_packet_loss_pct': round(float(np.mean(losses)), 3),
            'active_flows_count': len(self.flow_stats),
            'control_overhead': self.get_control_overhead_summary()
        }
        return summary

    # --------------------------------------------------------------------------
    # Feature Extractors for DQN Adaptive Routing
    # --------------------------------------------------------------------------

    def get_candidate_paths(self, src_dpid, dst_dpid, k=4):
        """
        Discovers up to k diversity-aware candidate paths between source and destination:
        - Path 0: Dijkstra Shortest Path (hop count / OSPF base)
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
            simple_paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, src_dpid, dst_dpid), 16))
            if not simple_paths:
                paths = [[src_dpid, dst_dpid]]
            else:
                p0 = simple_paths[0]
                paths.append(p0)

                # Path 1: 2nd shortest simple path
                if len(simple_paths) > 1:
                    paths.append(simple_paths[1])

                # Path 2: Edge-diverse / lateral bypass path (minimizing edge overlap with p0)
                p0_edges = set((p0[idx], p0[idx+1]) for idx in range(len(p0)-1))
                p0_edges.update((p0[idx+1], p0[idx]) for idx in range(len(p0)-1))
                diverse_candidates = [p for p in simple_paths if p not in paths]
                if diverse_candidates:
                    diverse_candidates.sort(key=lambda p: sum(1 for idx in range(len(p)-1) if (p[idx], p[idx+1]) in p0_edges))
                    paths.append(diverse_candidates[0])

                # Path 3: Node-diverse path (minimizing intermediate transit node overlap with p0)
                p0_nodes = set(p0[1:-1])
                node_diverse = [p for p in simple_paths if p not in paths]
                if node_diverse:
                    node_diverse.sort(key=lambda p: len(set(p[1:-1]) & p0_nodes))
                    paths.append(node_diverse[0])

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
        Builds a normalized 10-dimensional state vector for DQN Traffic Engineering Agent:
        [0]: Normalized shortest path length (hops / 10.0) in (0, 1]
        [1-4]: Bottleneck link utilization of Candidate Paths 0, 1, 2, 3 [0.0 - 1.0]
        [5-8]: Normalized end-to-end latency of Candidate Paths 0, 1, 2, 3 [0.0 - 1.0] (delay / 50.0)
        [9]: Network-wide peak link utilization [0.0 - 1.0]
        """
        state = np.zeros(10, dtype=np.float32)
        if not self.graph.has_node(src_dpid) or not self.graph.has_node(dst_dpid):
            return state

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
                state[1 + i] = 1.0
                state[5 + i] = 1.0

        # 9: Peak network utilization
        all_utils = list(self.link_utilization.values())
        state[9] = min(1.0, max(0.0, float(max(all_utils)))) if all_utils else 0.0

        return state

    # --------------------------------------------------------------------------
    # Interactive Simulation & Telemetry Injection
    # --------------------------------------------------------------------------

    def inject_traffic_flow(self, src_ip, dst_ip, src_dpid, dst_dpid, mbps=15.0, pps=1200.0, path=None):
        """Injects a simulated flow and updates link utilization, jitter, and loss along the chosen path."""
        duration = 10.0
        bytes_count = int((mbps * 1e6 * duration) / 8.0)
        packets = int(pps * duration)
        self.update_flow_stats(src_dpid, src_ip, dst_ip, packets, bytes_count, duration)

        if path and len(path) > 1:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                cap = self.link_bandwidths.get((u, v), 100.0)
                add_util = mbps / max(1.0, cap)
                new_u = min(1.0, self.link_utilization.get((u, v), 0.0) + add_util)
                self.link_utilization[(u, v)] = new_u

                # Update loss and jitter under new load
                if new_u <= 0.70:
                    loss_pct = new_u * 0.05
                else:
                    excess = (new_u - 0.70) / 0.30
                    loss_pct = min(35.0, 0.05 + 30.0 * (excess ** 2.2))
                self.link_loss[(u, v)] = float(loss_pct)

                curr_delay = self.link_delays.get((u, v), 2.0)
                self.update_link_latency(u, v, curr_delay * (1.0 + new_u))

                port = self.graph[u][v].get('port', 1) if self.graph.has_edge(u, v) else 1
                self.port_rates[(u, port)] = {
                    'rx_mbps': mbps, 'tx_mbps': mbps, 'rx_pps': pps, 'tx_pps': pps
                }

    # --------------------------------------------------------------------------
    # Closed-Loop Dynamic Flow Allocation (Proposal Objective 4)
    # --------------------------------------------------------------------------

    def allocate_dynamic_flow(self, flow_id, path, mbps=5.0, duration_sec=10.0):
        """
        Allocates an active unicast flow along a designated multi-hop path.
        Simulates dynamic closed-loop bandwidth consumption on link queues.
        """
        now = time.time()
        if not hasattr(self, 'active_dynamic_flows'):
            self.active_dynamic_flows = {}

        self.active_dynamic_flows[flow_id] = {
            'path': list(path),
            'mbps': float(mbps),
            'expires_at': now + float(duration_sec)
        }
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cap = self.link_bandwidths.get((u, v), 100.0)
            delta_u = mbps / max(1.0, cap)
            new_u = min(1.0, self.link_utilization.get((u, v), 0.05) + delta_u)
            self.link_utilization[(u, v)] = new_u

            # Recalculate analytical queue loss & latency
            if new_u <= 0.70:
                loss_pct = new_u * 0.05
            else:
                excess = (new_u - 0.70) / 0.30
                loss_pct = min(35.0, 0.05 + 30.0 * (excess ** 2.2))
            self.link_loss[(u, v)] = float(loss_pct)

            base_d = self.base_link_delays.get((u, v), 2.0)
            q_delay = base_d * (0.2 + 0.8 / max(0.05, 1.0 - min(0.95, new_u)))
            self.update_link_latency(u, v, q_delay)

    def release_dynamic_flow(self, flow_id):
        """Reclaims link bandwidth and relieves queue congestion upon flow completion."""
        if not hasattr(self, 'active_dynamic_flows'):
            self.active_dynamic_flows = {}
            return
        flow = self.active_dynamic_flows.pop(flow_id, None)
        if not flow:
            return
        path = flow['path']
        mbps = flow['mbps']
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cap = self.link_bandwidths.get((u, v), 100.0)
            delta_u = mbps / max(1.0, cap)
            new_u = max(0.02, self.link_utilization.get((u, v), 0.05) - delta_u)
            self.link_utilization[(u, v)] = new_u
            if new_u <= 0.70:
                loss_pct = new_u * 0.05
            else:
                excess = (new_u - 0.70) / 0.30
                loss_pct = min(35.0, 0.05 + 30.0 * (excess ** 2.2))
            self.link_loss[(u, v)] = float(loss_pct)

            base_d = self.base_link_delays.get((u, v), 2.0)
            q_delay = base_d * (0.2 + 0.8 / max(0.05, 1.0 - min(0.95, new_u)))
            self.update_link_latency(u, v, q_delay)

    def step_dynamic_flows(self, current_time=None):
        """Expires finished flows whose lifespan has ended."""
        if not hasattr(self, 'active_dynamic_flows'):
            self.active_dynamic_flows = {}
            return 0
        now = current_time or time.time()
        expired = [fid for fid, f in self.active_dynamic_flows.items() if now >= f['expires_at']]
        for fid in expired:
            self.release_dynamic_flow(fid)
        return len(expired)

    def inject_core_congestion(self, utilization=0.95, core_nodes=None, asymmetric_ratio=0.5):
        """
        Saturates core switch links to test lateral cross-link rerouting.
        Topology-Agnostic: dynamically identifies core / transit bottlenecks via
        metadata or networkx edge betweenness centrality.
        When multiple core nodes exist, saturates the primary subset (asymmetric_ratio)
        to simulate realistic elephant-flow hotspot bottlenecks rather than total gridlock.
        """
        target_edges = []
        c_nodes = core_nodes or (self.active_core_nodes if hasattr(self, 'active_core_nodes') and self.active_core_nodes else None)
        if c_nodes:
            if len(c_nodes) > 1 and asymmetric_ratio is not None and asymmetric_ratio < 1.0:
                subset_count = max(1, int(round(len(c_nodes) * asymmetric_ratio)))
                jammed_cores = set(c_nodes[:subset_count])
            else:
                jammed_cores = set(c_nodes)
            for u, v in self.graph.edges():
                if u in jammed_cores or v in jammed_cores:
                    target_edges.append((u, v))
        else:
            # Fallback: Automatic discovery via Edge Betweenness Centrality on arbitrary graph
            if self.graph.number_of_edges() > 0:
                centrality = nx.edge_betweenness_centrality(self.graph)
                sorted_edges = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                top_k = max(2, int(len(sorted_edges) * 0.25))
                target_edges = [edge for edge, _ in sorted_edges[:top_k]]
                for u, v in list(target_edges):
                    if self.graph.has_edge(v, u) and (v, u) not in target_edges:
                        target_edges.append((v, u))

        # Apply high congestion to discovered core edges
        target_set = set(target_edges)
        for u, v in self.graph.edges():
            if (u, v) in target_set:
                self.link_utilization[(u, v)] = float(utilization)
                self.link_loss[(u, v)] = 21.5
                self.link_jitter[(u, v)] = 12.8
                base_d = self.base_link_delays.get((u, v), 2.0)
                self.link_delays[(u, v)] = max(base_d * 5.0, 20.0)
            else:
                # Keep lateral / perimeter edges uncongested
                self.link_utilization[(u, v)] = 0.15
                self.link_loss[(u, v)] = 0.01
                self.link_jitter[(u, v)] = 0.45
                if (u, v) in self.link_delays:
                    base_d = self.base_link_delays.get((u, v), 2.0)
                    self.link_delays[(u, v)] = base_d

        self.active_simulation_mode = "core_jamming"
        return len(target_edges)

    def reset_simulation(self):
        """Resets link utilization and clears simulated telemetry."""
        if hasattr(self, 'active_dynamic_flows'):
            self.active_dynamic_flows.clear()
        for k in self.link_utilization:
            self.link_utilization[k] = 0.05
            self.link_loss[k] = 0.0
            self.link_jitter[k] = 0.25
        for k in self.link_delays:
            self.link_delays[k] = self.base_link_delays.get(k, 2.0)
        self._prev_delays.clear()
        self.port_rates.clear()
        self.flow_stats.clear()
        self.raw_flow_stats.clear()
        self.raw_port_stats.clear()
        self.restore_link()
        self.active_simulation_mode = None

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

    def sample_telemetry_history(self):
        """Periodic telemetry history sampler."""
        now = time.time()
        te_summary = self.get_network_te_summary()
        self.telemetry_history.append({
            'timestamp': time.strftime("%H:%M:%S", time.localtime(now)),
            'throughput_mbps': te_summary['total_throughput_mbps'],
            'bottleneck_utilization_pct': te_summary['bottleneck_utilization_pct'],
            'mean_utilization_pct': te_summary['mean_utilization_pct'],
            'avg_latency_ms': te_summary['mean_latency_ms'],
            'avg_jitter_ms': te_summary['mean_jitter_ms'],
            'packet_loss_pct': te_summary['mean_packet_loss_pct'],
            'active_flows': te_summary['active_flows_count'],
            'flow_mods': te_summary['control_overhead']['flow_mod_count']
        })

    def update_security_window(self):
        """Alias for backward compatibility."""
        self.sample_telemetry_history()
