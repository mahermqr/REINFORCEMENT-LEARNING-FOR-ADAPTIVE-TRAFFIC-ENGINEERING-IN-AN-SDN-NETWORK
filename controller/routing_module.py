import sys
import os
import time
import threading
import numpy as np

# Add agent and controller to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))
sys.path.append(os.path.dirname(__file__))
from dqn_router import DQNRoutingAgent
from ryu.lib.packet import ethernet, ipv4, ether_types

class RoutingModule:
    """
    Adaptive Unicast Routing Engine powered by Deep Q-Network (Double DQN) for SDN Traffic Engineering (EC499).
    Dynamically routes traffic across multi-hop paths to maximize throughput, balance link loads,
    minimize latency, reduce jitter, and prevent packet loss.
    """
    def __init__(self, controller, state_manager):
        self.controller = controller
        self.state_manager = state_manager
        self.logger = controller.logger

        # Initialize Double DQN Agent (10 state features, 4 candidate path actions)
        self.agent = DQNRoutingAgent(state_size=10, action_size=4, lr=0.001)
        self.prev_experience = {} # (src_dpid, dst_dpid) -> (state, action, candidate_paths)

        # Auto-load trained checkpoint if available
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dqn_router.pth')
        if os.path.exists(ckpt_path):
            self.agent.load(ckpt_path)
            self.logger.info("[RoutingModule] Loaded pre-trained Double DQN checkpoint from %s", ckpt_path)

        # Asynchronous background training worker to prevent blocking OpenFlow PacketIn processing
        self.model_lock = threading.Lock()
        self._training_active = False
        self._training_thread = None
        self.start_training_worker()

        self.logger.info("[RoutingModule] Initialized with Double DQN Agent on device: %s", self.agent.device)

    def start_training_worker(self):
        """Starts decoupled background worker thread for PyTorch policy optimization."""
        if not self._training_active:
            self._training_active = True
            self._training_thread = threading.Thread(target=self._training_loop, daemon=True)
            self._training_thread.start()
            self.logger.info("[RoutingModule] Asynchronous background training worker started.")

    def stop_training_worker(self):
        """Halts background training worker gracefully."""
        self._training_active = False

    def _training_loop(self):
        """Background thread executing periodic Double DQN policy updates."""
        while self._training_active:
            try:
                time.sleep(2.0)
                if hasattr(self.agent, 'memory') and len(self.agent.memory) >= 32:
                    with self.model_lock:
                        loss = self.agent.train(batch_size=32)
                    if loss is not None:
                        self.logger.debug("[RoutingModule-Worker] Asynchronous DQN Loss: %.4f | Epsilon: %.3f", loss, self.agent.epsilon)
            except Exception as e:
                self.logger.error("[RoutingModule-Worker] Training exception: %s", e)

    def handle_unicast(self, ev, pkt):
        """Processes unicast packets, computes optimal path via DQN, and installs multi-hop flows."""
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        eth_pkt = pkt.get_protocols(ethernet.ethernet)[0]
        src_mac = eth_pkt.src
        dst_mac = eth_pkt.dst

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        src_ip = ip_pkt.src if ip_pkt else None
        dst_ip = ip_pkt.dst if ip_pkt else None

        # Learn host location
        self.state_manager.record_host(src_ip, src_mac, dpid, in_port)

        # Lookup destination host location
        dst_dpid, dst_port = self.state_manager.get_host_location(ip=dst_ip, mac=dst_mac)

        if dst_dpid is None or dst_port is None:
            # Destination not yet learned: flood packet to discover host
            self._flood(datapath, msg, in_port)
            return

        # Case 1: Source and destination are connected to the same switch
        if dpid == dst_dpid:
            actions = [parser.OFPActionOutput(dst_port)]
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac)
            self.controller.add_flow(datapath, priority=10, match=match, actions=actions, idle_timeout=30)
            self._send_packet_out(datapath, msg, actions)
            return

        # Case 2: Multi-hop routing across SDN fabric
        t0 = time.time()
        state = self.state_manager.get_routing_state(dpid, dst_dpid)
        # Production traffic uses deterministic policy evaluation (explore=False)
        with self.model_lock:
            action = self.agent.act(state, explore=False)
        decision_time_ms = (time.time() - t0) * 1000.0
        self.state_manager.record_decision_latency(decision_time_ms)

        # Discover diversity-aware candidate paths using StateManager
        candidate_paths = self.state_manager.get_candidate_paths(dpid, dst_dpid, k=4)
        if not candidate_paths or len(candidate_paths[0]) <= 1:
            self.logger.warning("[RoutingModule] No path found between switch %s and %s. Flooding.", dpid, dst_dpid)
            self._flood(datapath, msg, in_port)
            return

        # Action selects from candidate paths (modulo candidate count)
        selected_path = candidate_paths[action % len(candidate_paths)]

        # Experience Replay Training: record transition with rigorous temporal credit assignment
        exp_key = (dpid, dst_dpid)
        if exp_key in self.prev_experience:
            prev_s, prev_a, prev_paths = self.prev_experience[exp_key]
            # Reward is calculated for previous action prev_a observed in current network state
            chosen_prev_path = prev_paths[prev_a % len(prev_paths)]
            reward = self._calculate_reward(chosen_prev_path)
            with self.model_lock:
                self.agent.remember(prev_s, prev_a, reward, state, done=False)

        self.prev_experience[exp_key] = (state, action, candidate_paths)

        self.logger.info("[RoutingModule] Routing %s -> %s via path: %s | Action: %d | Decision: %.2f ms",
                         src_ip or src_mac, dst_ip or dst_mac, selected_path, action, decision_time_ms)

        # Install OpenFlow 1.3 flow rules along the entire path
        self._install_path(selected_path, dst_port, eth_dst=dst_mac, ip_src=src_ip, ip_dst=dst_ip)
        if src_ip and dst_ip:
            self.state_manager.record_flow_path(dpid, src_ip, dst_ip, selected_path)

        # Forward the current packet out of the first hop
        first_hop = selected_path[1]
        edge_data = self.state_manager.graph.get_edge_data(dpid, first_hop)
        out_port = edge_data['port'] if edge_data and 'port' in edge_data else ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]
        self._send_packet_out(datapath, msg, actions)

    def _calculate_reward(self, path):
        """
        Formulates composite reward function based on Proposal Objectives 3 & 4:
        Maximizes throughput and link fairness, while penalizing latency, jitter, packet loss, and peak load.
        """
        hops = len(path) - 1
        total_delay = 0.0
        max_util = 0.0
        total_jitter = 0.0
        max_loss = 0.0

        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            delay = self.state_manager.link_delays.get(edge, 2.0)
            util = self.state_manager.link_utilization.get(edge, 0.0)
            jitter = self.state_manager.link_jitter.get(edge, 0.25)
            loss = self.state_manager.link_loss.get(edge, 0.0)

            total_delay += delay
            max_util = max(max_util, util)
            total_jitter += jitter
            max_loss = max(max_loss, loss)

        # Asymptotic barrier penalty when bottleneck utilization exceeds 70%
        if max_util > 0.70:
            congestion_penalty = 12.0 * ((max_util ** 1.8) / max(0.01, 1.02 - max_util))
        else:
            congestion_penalty = 1.5 * max_util

        reward = - (0.35 * hops + 0.06 * total_delay + congestion_penalty + 0.25 * total_jitter + 0.5 * max_loss)
        return float(reward)

    def _install_path(self, path, final_port, eth_dst, ip_src=None, ip_dst=None):
        """Pushes flow rules across all switches along the calculated path."""
        for i in range(len(path)):
            cur_dpid = path[i]
            dp = self.controller.datapaths.get(cur_dpid)
            if not dp:
                continue

            parser = dp.ofproto_parser

            # Determine egress port for this hop
            if i < len(path) - 1:
                next_dpid = path[i+1]
                edge_data = self.state_manager.graph.get_edge_data(cur_dpid, next_dpid)
                out_port = edge_data['port'] if edge_data else dp.ofproto.OFPP_FLOOD
            else:
                out_port = final_port

            actions = [parser.OFPActionOutput(out_port)]
            if ip_dst and ip_src:
                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=ip_src, ipv4_dst=ip_dst)
            elif ip_dst:
                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_dst=ip_dst)
            else:
                match = parser.OFPMatch(eth_dst=eth_dst)

            self.controller.add_flow(dp, priority=10, match=match, actions=actions, idle_timeout=60, hard_timeout=120)

    def calculate_wcmp_weights(self, candidate_paths, beta=4.0):
        """
        Calculates Weighted Cost Multi-Path (WCMP) distribution weights
        using Softmax over inverse bottleneck utilization and latency.
        """
        if not candidate_paths:
            return []
        if len(candidate_paths) == 1:
            return [(candidate_paths[0], 100)]

        scores = []
        for p in candidate_paths:
            max_u = 0.0
            total_d = 0.0
            for i in range(len(p) - 1):
                edge = (p[i], p[i+1])
                u = self.state_manager.link_utilization.get(edge, 0.0)
                d = self.state_manager.link_delays.get(edge, 2.0)
                max_u = max(max_u, u)
                total_d += d
            score = - (beta * max_u + 0.05 * total_d)
            scores.append(score)

        scores = np.array(scores, dtype=np.float64)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / max(1e-9, np.sum(exp_scores))
        int_weights = np.maximum(1, np.round(probs * 100.0).astype(int))
        return list(zip(candidate_paths, int_weights.tolist()))

    def install_wcmp_multipath(self, candidate_paths, final_port, eth_dst, ip_src=None, ip_dst=None, group_id=500):
        """
        Installs an OpenFlow 1.3 OFPGT_SELECT Group on the ingress switch,
        distributing flows across multiple paths according to WCMP weights.
        """
        if not candidate_paths:
            return False

        weighted_paths = self.calculate_wcmp_weights(candidate_paths)
        ingress_dpid = candidate_paths[0][0]
        dp_ingress = self.controller.datapaths.get(ingress_dpid)

        # Install intermediate downstream rules for each path
        for path, _ in weighted_paths:
            if len(path) > 2:
                self._install_path(path[1:], final_port, eth_dst, ip_src=ip_src, ip_dst=ip_dst)

        if not dp_ingress:
            return True

        ofproto = dp_ingress.ofproto
        parser = dp_ingress.ofproto_parser

        buckets = []
        for path, weight in weighted_paths:
            if len(path) > 1:
                next_dpid = path[1]
                edge_data = self.state_manager.graph.get_edge_data(ingress_dpid, next_dpid)
                out_port = edge_data['port'] if edge_data else ofproto.OFPP_FLOOD
            else:
                out_port = final_port

            bucket_actions = [parser.OFPActionOutput(out_port)]
            bucket = parser.OFPBucket(weight=int(weight), actions=bucket_actions)
            buckets.append(bucket)

        try:
            del_mod = parser.OFPGroupMod(dp_ingress, ofproto.OFPGC_DELETE, ofproto.OFPGT_SELECT, group_id)
            dp_ingress.send_msg(del_mod)
        except Exception:
            pass

        group_mod = parser.OFPGroupMod(
            dp_ingress, ofproto.OFPGC_ADD, ofproto.OFPGT_SELECT, group_id, buckets=buckets
        )
        dp_ingress.send_msg(group_mod)

        actions = [parser.OFPActionGroup(group_id)]
        if ip_dst and ip_src:
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=ip_src, ipv4_dst=ip_dst)
        elif ip_dst:
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_dst=ip_dst)
        else:
            match = parser.OFPMatch(eth_dst=eth_dst)

        self.controller.add_flow(dp_ingress, priority=15, match=match, actions=actions, idle_timeout=60)
        self.logger.info("[RoutingModule] Installed WCMP Multipath Group %d on Switch %s with %d buckets",
                         group_id, ingress_dpid, len(buckets))
        return True

    def install_fast_failover_path(self, primary_path, backup_path, final_port, eth_dst, ip_src=None, ip_dst=None, group_id=600):
        """
        Installs an OpenFlow 1.3 OFPGT_FF (Fast Failover) Group on the ingress switch.
        Switches instantaneously to backup_path in hardware if primary watch port goes down.
        """
        ingress_dpid = primary_path[0]
        dp_ingress = self.controller.datapaths.get(ingress_dpid)

        if len(primary_path) > 2:
            self._install_path(primary_path[1:], final_port, eth_dst, ip_src=ip_src, ip_dst=ip_dst)
        if len(backup_path) > 2:
            self._install_path(backup_path[1:], final_port, eth_dst, ip_src=ip_src, ip_dst=ip_dst)

        if not dp_ingress:
            return True

        ofproto = dp_ingress.ofproto
        parser = dp_ingress.ofproto_parser

        p_next = primary_path[1] if len(primary_path) > 1 else primary_path[0]
        p_edge = self.state_manager.graph.get_edge_data(ingress_dpid, p_next)
        p_port = p_edge['port'] if p_edge and 'port' in p_edge else 1

        b_next = backup_path[1] if len(backup_path) > 1 else backup_path[0]
        b_edge = self.state_manager.graph.get_edge_data(ingress_dpid, b_next)
        b_port = b_edge['port'] if b_edge and 'port' in b_edge else 2

        b1 = parser.OFPBucket(watch_port=p_port, watch_group=ofproto.OFPG_ANY,
                              actions=[parser.OFPActionOutput(p_port)])
        b2 = parser.OFPBucket(watch_port=b_port, watch_group=ofproto.OFPG_ANY,
                              actions=[parser.OFPActionOutput(b_port)])

        try:
            del_mod = parser.OFPGroupMod(dp_ingress, ofproto.OFPGC_DELETE, ofproto.OFPGT_FF, group_id)
            dp_ingress.send_msg(del_mod)
        except Exception:
            pass

        ff_mod = parser.OFPGroupMod(
            dp_ingress, ofproto.OFPGC_ADD, ofproto.OFPGT_FF, group_id, buckets=[b1, b2]
        )
        dp_ingress.send_msg(ff_mod)

        actions = [parser.OFPActionGroup(group_id)]
        if ip_dst and ip_src:
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=ip_src, ipv4_dst=ip_dst)
        elif ip_dst:
            match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_dst=ip_dst)
        else:
            match = parser.OFPMatch(eth_dst=eth_dst)

        self.controller.add_flow(dp_ingress, priority=20, match=match, actions=actions, idle_timeout=60)
        self.logger.info("[RoutingModule] Installed Fast Failover Group %d on Switch %s (Primary port %d, Backup port %d)",
                         group_id, ingress_dpid, p_port, b_port)
        return True

    def _flood(self, datapath, msg, in_port):
        """Safely floods packet to edge host ports only to avoid broadcast storms."""
        for dp in list(self.controller.datapaths.values()):
            dp_trunk_ports = set()
            if self.state_manager.graph.has_node(dp.id):
                for _, _, edge_data in self.state_manager.graph.out_edges(dp.id, data=True):
                    if 'port' in edge_data:
                        dp_trunk_ports.add(edge_data['port'])

            for p_no in dp.ports:
                if p_no <= dp.ofproto.OFPP_MAX and p_no not in dp_trunk_ports:
                    if dp.id == datapath.id and p_no == in_port:
                        continue
                    dp_parser = dp.ofproto_parser
                    actions = [dp_parser.OFPActionOutput(p_no)]
                    out = dp_parser.OFPPacketOut(
                        datapath=dp, buffer_id=dp.ofproto.OFP_NO_BUFFER,
                        in_port=dp.ofproto.OFPP_CONTROLLER, actions=actions, data=msg.data
                    )
                    dp.send_msg(out)

    def _send_packet_out(self, datapath, msg, actions):
        """Sends OFPPacketOut message."""
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        data = None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=msg.match['in_port'], actions=actions, data=data)
        datapath.send_msg(out)
