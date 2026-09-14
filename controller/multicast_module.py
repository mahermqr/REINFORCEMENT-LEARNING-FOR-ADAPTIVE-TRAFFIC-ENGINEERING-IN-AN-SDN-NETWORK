import sys
import os
import networkx as nx

# Add agent and controller to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))
sys.path.append(os.path.dirname(__file__))
from dqn_multicast import DQNMulticastAgent
from ryu.lib.packet import ipv4, ether_types

class MulticastModule:
    """
    Intelligent Multicast Routing Module powered by Dueling Double DQN.
    Constructs optimal multicast trees using RL-weighted Steiner Tree heuristics,
    configuring OpenFlow 1.3 Group Tables (ALL) for hardware-accelerated packet replication.
    """
    def __init__(self, controller, state_manager):
        self.controller = controller
        self.state_manager = state_manager
        self.logger = controller.logger

        # Initialize Dueling Double DQN Agent (50 state features, 10 action weight configurations)
        self.agent = DQNMulticastAgent(state_size=50, action_size=10, lr=0.001)
        self.group_id_counter = 100
        self.installed_trees = {} # group_ip -> tree_graph
        self.prev_experience = {}

        # Auto-load trained checkpoint if available
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dqn_multicast.pth')
        if os.path.exists(ckpt_path):
            self.agent.load(ckpt_path)
            self.logger.info("[MulticastModule] Loaded pre-trained Dueling DQN checkpoint from %s", ckpt_path)

        self.logger.info("[MulticastModule] Initialized with Dueling Double DQN on device: %s", self.agent.device)

    def handle_multicast(self, ev, pkt):
        """Builds optimal multicast tree and installs OpenFlow 1.3 Group Tables."""
        msg = ev.msg
        datapath = msg.datapath
        src_dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        group_ip = ip_pkt.dst if ip_pkt else "224.1.1.1"

        # Determine receiver switches from StateManager or default topology edge switches
        receivers = self.state_manager.get_multicast_receivers(group_ip)
        receiver_dpids = list({r[0] for r in receivers if r[0] is not None})

        # If no explicit IGMP joins yet, automatically default to leaf switches in topology
        if not receiver_dpids:
            all_nodes = list(self.state_manager.graph.nodes())
            # Select other switches as multicast receivers
            receiver_dpids = [node for node in all_nodes if node != src_dpid][:4]

        if not receiver_dpids:
            self.logger.warning("[MulticastModule] No receivers found for multicast group %s", group_ip)
            return

        # Build RL State representation
        state = self.state_manager.get_multicast_state(src_dpid, receiver_dpids)
        action = self.agent.act(state, explore=True)

        # Build weighted network graph according to RL action
        weighted_graph = self._build_weighted_graph(action)

        # Construct Steiner Multicast Tree connecting Source + All Receivers
        try:
            terminal_nodes = list(set([src_dpid] + receiver_dpids))
            tree = nx.algorithms.approximation.steinertree.steiner_tree(
                weighted_graph.to_undirected(), terminal_nodes, weight='weight'
            )
        except Exception as e:
            self.logger.error("[MulticastModule] Steiner Tree generation failed: %s. Falling back to shortest paths.", e)
            tree = self._build_fallback_tree(src_dpid, receiver_dpids)

        # Calculate reward based on total tree cost, branch replication efficiency, and delay
        reward = self._calculate_multicast_reward(tree, len(receiver_dpids))

        # Experience Replay Training
        if group_ip in self.prev_experience:
            prev_s, prev_a = self.prev_experience[group_ip]
            self.agent.remember(prev_s, prev_a, reward, state, done=False)
            loss = self.agent.train(batch_size=32)
            if loss is not None:
                self.logger.debug("[MulticastModule] Dueling DQN Loss: %.4f | Epsilon: %.3f", loss, self.agent.epsilon)

        self.prev_experience[group_ip] = (state, action)

        self.logger.info("[MulticastModule] Group %s Tree built: %d nodes, %d edges | Action: %d | Reward: %.2f",
                         group_ip, tree.number_of_nodes(), tree.number_of_edges(), action, reward)

        # Install Group Table entries and flow rules across all switches in the tree
        self._deploy_multicast_tree(tree, src_dpid, group_ip, receivers)

        # Forward current packet
        self._forward_from_ingress(datapath, msg, tree, in_port, group_ip)

    def _build_weighted_graph(self, action):
        """Applies RL action to weight links by delay, utilization, or loss."""
        g = self.state_manager.graph.copy()

        # Action modulates the weight vector:
        # Action 0-3: Delay priority
        # Action 4-6: Bandwidth utilization priority
        # Action 7-9: Balanced load
        alpha = 1.0 + (action % 3) * 0.5
        beta = 1.0 + ((action // 3) % 3) * 0.5

        for u, v in g.edges():
            delay = self.state_manager.link_delays.get((u, v), 2.0)
            util = self.state_manager.link_utilization.get((u, v), 0.0)
            loss = self.state_manager.link_loss.get((u, v), 0.0)

            # Composite edge weight
            weight = alpha * delay + beta * (util * 20.0) + (loss * 100.0) + 1.0
            g[u][v]['weight'] = weight

        return g

    def _build_fallback_tree(self, src_dpid, receivers):
        """Constructs a union of shortest paths if Steiner tree approximation encounters errors."""
        tree = nx.Graph()
        tree.add_node(src_dpid)
        for r in receivers:
            try:
                p = nx.shortest_path(self.state_manager.graph, src_dpid, r)
                nx.add_path(tree, p)
            except Exception:
                pass
        return tree

    def _calculate_multicast_reward(self, tree, num_receivers):
        """
        Multicast Reward:
        R = (Replication Savings vs Unicast) - (Tree Edge Cost + Average Delay Penalty)
        """
        tree_edges = tree.number_of_edges()
        unicast_approx_edges = num_receivers * 3 # Estimated separate unicast paths
        savings = max(0, unicast_approx_edges - tree_edges)

        reward = float(savings * 2.0 - tree_edges * 0.8)
        return reward

    def _deploy_multicast_tree(self, tree, src_dpid, group_ip, receivers):
        """Traverses the multicast tree and configures OpenFlow 1.3 Group Tables on branching nodes."""
        self.group_id_counter = (self.group_id_counter % 1000) + 1
        group_id = self.group_id_counter

        # Directed BFS tree from source
        directed_tree = nx.bfs_tree(tree, src_dpid)

        for node_dpid in directed_tree.nodes():
            dp = self.controller.datapaths.get(node_dpid)
            if not dp:
                continue

            parser = dp.ofproto_parser
            ofproto = dp.ofproto
            children = list(directed_tree.successors(node_dpid))

            out_ports = []
            for child in children:
                edge_data = self.state_manager.graph.get_edge_data(node_dpid, child)
                if edge_data and 'port' in edge_data:
                    out_ports.append(edge_data['port'])

            # Include local host receiver ports attached to this switch
            for r_dpid, r_port in receivers:
                if r_dpid == node_dpid and r_port is not None and r_port not in out_ports:
                    out_ports.append(r_port)

            if not out_ports:
                continue

            if len(out_ports) > 1:
                # Branching node: Create OpenFlow Group (OFPGT_ALL)
                buckets = []
                for p in out_ports:
                    act = [parser.OFPActionOutput(p)]
                    buckets.append(parser.OFPBucket(actions=act))

                # Install or modify group table entry
                req = parser.OFPGroupMod(dp, ofproto.OFPGC_ADD, ofproto.OFPGT_ALL, group_id, buckets)
                dp.send_msg(req)

                # Direct multicast traffic to this group
                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_dst=group_ip)
                actions = [parser.OFPActionGroup(group_id=group_id)]
                self.controller.add_flow(dp, priority=20, match=match, actions=actions, idle_timeout=60)
            else:
                # Single egress port: Standard flow rule
                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_dst=group_ip)
                actions = [parser.OFPActionOutput(out_ports[0])]
                self.controller.add_flow(dp, priority=20, match=match, actions=actions, idle_timeout=60)

    def _forward_from_ingress(self, datapath, msg, tree, in_port, group_ip):
        """Forwards packet from ingress switch."""
        parser = datapath.ofproto_parser
        src_dpid = datapath.id
        directed_tree = nx.bfs_tree(tree, src_dpid)
        children = list(directed_tree.successors(src_dpid))

        out_ports = []
        for child in children:
            edge_data = self.state_manager.graph.get_edge_data(src_dpid, child)
            if edge_data and 'port' in edge_data:
                out_ports.append(edge_data['port'])

        if not out_ports:
            return

        actions = [parser.OFPActionOutput(p) for p in out_ports]
        data = None if msg.buffer_id != datapath.ofproto.OFP_NO_BUFFER else msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
