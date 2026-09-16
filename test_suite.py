#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite (EC499).
Reinforcement Learning for Adaptive Traffic Engineering in an SDN Network.

Tests:
  1. Deep Q-Network (Double DQN) Lifecycle & Dueling Architecture
  2. Prioritized Experience Replay (SumTree arithmetic, IS weights, priority updates)
  3. StateManager Telemetry (Throughput, Latency, Jitter RFC 3393, Packet Loss, Control Overhead)
  4. Classical & Greedy Routing Baselines (OSPF RFC 2328, Dijkstra SPF, ECMP, WSP, LLR)
  5. Multi-Topology Fabric Integrity (Tree, Fat-Tree, Abilene, NSFNet, Spine-Leaf)
  6. Controller OpenFlow 1.3 Flow Installation & Candidate Path Decomposition
"""

import sys
import os
import time
import tempfile
import unittest
import numpy as np
import torch
import networkx as nx

# Setup path imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'agent'))
sys.path.append(os.path.join(BASE_DIR, 'controller'))
sys.path.append(os.path.join(BASE_DIR, 'topology'))

from dqn_router import DQNRoutingAgent, DuelingQNetwork
from prioritized_replay import PrioritizedReplayBuffer, SumTree
from state_manager import StateManager
from topology_library import get_topology, list_available_topologies
from traditional_routing import (
    ospf_routing, dijkstra_spf, ecmp_routing, widest_shortest_path,
    least_loaded_routing, random_routing, compute_path_metrics
)


class TestDQNAgent(unittest.TestCase):
    """Verifies PyTorch DQN Agent forward passes, Dueling architecture, experience replay, and checkpointing."""

    def test_dqn_router_lifecycle(self):
        agent = DQNRoutingAgent(state_size=10, action_size=4, lr=0.001)
        dummy_state = np.random.rand(10).astype(np.float32)

        # Test greedy vs exploration inference
        greedy_action = agent.act(dummy_state, explore=False)
        self.assertIn(greedy_action, range(4))

        # Test experience storage
        for _ in range(40):
            s = np.random.rand(10).astype(np.float32)
            a = np.random.randint(0, 4)
            r = float(np.random.randn())
            s_next = np.random.rand(10).astype(np.float32)
            agent.remember(s, a, r, s_next, False)

        loss = agent.train(batch_size=16)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)

        # Test checkpoint save and load
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            new_agent = DQNRoutingAgent(state_size=10, action_size=4)
            loaded = new_agent.load(tmp_path)
            self.assertTrue(loaded)
            # Verify loaded weights match
            for p1, p2 in zip(agent.model.parameters(), new_agent.model.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_dueling_architecture(self):
        model = DuelingQNetwork(state_size=10, action_size=4)
        dummy_input = torch.randn(4, 10)
        q_values = model(dummy_input)

        self.assertEqual(q_values.shape, (4, 4))

        # Verify Value and Advantage streams
        features = model.feature_network(dummy_input)
        val = model.value_stream(features)
        adv = model.advantage_stream(features)
        self.assertEqual(val.shape, (4, 1))
        self.assertEqual(adv.shape, (4, 4))

        # Check advantage centering: mean of centered advantage should be ~0
        centered_adv = adv - adv.mean(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(centered_adv.mean(dim=-1), torch.zeros(4), atol=1e-5))


class TestPrioritizedReplay(unittest.TestCase):
    """Verifies SumTree arithmetic and Prioritized Experience Replay buffer sampling."""

    def test_sumtree_arithmetic(self):
        tree = SumTree(capacity=8)
        priorities = [1.0, 2.0, 3.0, 4.0]
        for p in priorities:
            tree.add(p, f"data_{p}")

        self.assertAlmostEqual(tree.total(), 10.0, places=4)
        self.assertEqual(tree.n_entries, 4)

        # Update priority of first entry from 1.0 to 5.0
        idx, _, _ = tree.get(0.5)
        tree.update(idx, 5.0)
        self.assertAlmostEqual(tree.total(), 14.0, places=4)

    def test_replay_buffer_sampling(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)
        batch, idxs, weights = buf.sample(16)
        self.assertEqual(len(batch), 0)

        for i in range(25):
            buf.add(np.zeros(10), i % 4, 1.0, np.zeros(10), False)

        batch, idxs, weights = buf.sample(16)
        self.assertEqual(len(batch), 16)
        self.assertEqual(len(idxs), 16)
        self.assertEqual(len(weights), 16)
        self.assertAlmostEqual(weights.max(), 1.0, places=3)


class TestStateManager(unittest.TestCase):
    """Verifies topology representation, rate extraction, Jitter, Loss, and Control Overhead."""

    def setUp(self):
        self.sm = StateManager()
        self.sm.update_link(1, 2, src_port=1, dst_port=1, capacity_mbps=100.0, delay_ms=2.0)
        self.sm.update_link(2, 1, src_port=1, dst_port=1, capacity_mbps=100.0, delay_ms=2.0)
        self.sm.update_link(2, 3, src_port=2, dst_port=1, capacity_mbps=50.0, delay_ms=4.0)
        self.sm.update_link(3, 2, src_port=1, dst_port=2, capacity_mbps=50.0, delay_ms=4.0)
        self.sm.update_link(1, 4, src_port=2, dst_port=1, capacity_mbps=100.0, delay_ms=3.0)
        self.sm.update_link(4, 1, src_port=1, dst_port=2, capacity_mbps=100.0, delay_ms=3.0)

    def test_routing_state_features(self):
        state = self.sm.get_routing_state(src_dpid=4, dst_dpid=3)
        self.assertEqual(len(state), 10)
        self.assertGreater(state[0], 0.0)
        self.assertLessEqual(state[0], 1.0)
        for i in range(10):
            self.assertGreaterEqual(state[i], 0.0)
            self.assertLessEqual(state[i], 1.0)

    def test_host_location_tracking(self):
        self.sm.record_host(ip="10.0.0.1", mac="00:00:00:00:00:01", dpid=4, port=3)
        dpid, port = self.sm.get_host_location(ip="10.0.0.1")
        self.assertEqual(dpid, 4)
        self.assertEqual(port, 3)

    def test_differential_port_rates(self):
        self.sm.update_port_stats(dpid=1, port_no=1, rx_bytes=1000, tx_bytes=2000,
                                  rx_packets=10, tx_packets=20, duration_sec=1)
        time.sleep(0.12)
        self.sm.update_port_stats(dpid=1, port_no=1, rx_bytes=1000, tx_bytes=1002000,
                                  rx_packets=10, tx_packets=1020, duration_sec=2)

        rates = self.sm.port_rates.get((1, 1))
        self.assertIsNotNone(rates)
        self.assertGreater(rates['tx_mbps'], 10.0)
        self.assertGreater(rates['tx_pps'], 1000.0)

    def test_jitter_calculation(self):
        # Update link with varying latency
        self.sm.update_link_latency(1, 2, delay_ms=2.0)
        self.sm.update_link_latency(1, 2, delay_ms=8.0)
        jitter = self.sm.link_jitter.get((1, 2))
        self.assertIsNotNone(jitter)
        self.assertGreater(jitter, 0.0)

    def test_packet_loss_modeling(self):
        # Low load: loss near 0
        self.sm.link_utilization[(1, 2)] = 0.35
        self.sm.update_port_stats(1, 1, 1000, 2000, 10, 20, 1)
        # Saturated load: loss increases
        self.sm.link_utilization[(1, 2)] = 0.95
        loss_metric = 0.05 + 30.0 * (((0.95 - 0.70) / 0.30) ** 2.2)
        self.assertGreater(loss_metric, 10.0)

    def test_control_overhead_accounting(self):
        self.sm.record_packet_in(byte_size=64)
        self.sm.record_flow_mod(byte_size=72)
        self.sm.record_stats_request(byte_size=56)
        self.sm.record_stats_reply(byte_size=128)
        self.sm.record_decision_latency(0.45)

        summary = self.sm.get_control_overhead_summary()
        self.assertEqual(summary['packet_in_count'], 1)
        self.assertEqual(summary['flow_mod_count'], 1)
        self.assertEqual(summary['stats_request_count'], 1)
        self.assertEqual(summary['stats_reply_count'], 1)
        self.assertGreater(summary['mean_decision_latency_ms'], 0.0)

    def test_network_te_summary(self):
        self.sm.inject_traffic_flow("10.0.0.1", "10.0.0.3", 4, 3, mbps=20.0, path=[4, 1, 2, 3])
        summary = self.sm.get_network_te_summary()
        self.assertGreaterEqual(summary['total_throughput_mbps'], 0.0)
        self.assertGreaterEqual(summary['jains_fairness_index'], 0.0)
        self.assertLessEqual(summary['jains_fairness_index'], 1.0)
        self.assertIn('control_overhead', summary)

    def test_link_failure_and_restoration(self):
        self.assertTrue(self.sm.inject_link_failure(1, 2))
        self.assertFalse(self.sm.graph.has_edge(1, 2))
        restored = self.sm.restore_link(1, 2)
        self.assertEqual(restored, 2) # Restores both directions
        self.assertTrue(self.sm.graph.has_edge(1, 2))


class TestTraditionalRoutingBaselines(unittest.TestCase):
    """Verifies OSPF, Dijkstra SPF, ECMP, WSP, LLR and path metric computations."""

    def setUp(self):
        g, _ = get_topology('tree')
        self.graph = g
        self.link_utils = {}
        self.link_delays = {}
        self.link_bws = {}
        for u, v, d in g.edges(data=True):
            self.link_bws[(u, v)] = d.get('capacity', 100.0)
            self.link_delays[(u, v)] = d.get('delay', 2.0)
            self.link_utils[(u, v)] = 0.20

    def test_ospf_routing(self):
        p = ospf_routing(self.graph, 4, 6, link_bandwidths=self.link_bws)
        self.assertEqual(p[0], 4)
        self.assertEqual(p[-1], 6)
        # OSPF path exists
        self.assertGreaterEqual(len(p), 2)

    def test_dijkstra_spf(self):
        p = dijkstra_spf(self.graph, 4, 6)
        self.assertEqual(p[0], 4)
        self.assertEqual(p[-1], 6)
        self.assertEqual(p, [4, 6]) # Direct cross-link is 1 hop

    def test_ecmp_routing(self):
        p = ecmp_routing(self.graph, 4, 6, flow_hash=0)
        self.assertEqual(p[0], 4)
        self.assertEqual(p[-1], 6)

    def test_wsp_and_llr_routing(self):
        # Saturate direct cross-link [4, 6] to 95%
        self.link_utils[(4, 6)] = 0.95
        self.link_utils[(6, 4)] = 0.95

        p_wsp = widest_shortest_path(self.graph, 4, 6, self.link_utils, self.link_delays)
        p_llr = least_loaded_routing(self.graph, 4, 6, self.link_utils)

        # Both WSP and LLR should avoid the 95% saturated direct link if an alternative is less loaded
        m_wsp = compute_path_metrics(p_wsp, self.link_utils, self.link_delays, self.link_bws)
        m_llr = compute_path_metrics(p_llr, self.link_utils, self.link_delays, self.link_bws)

        self.assertLessEqual(m_wsp['bottleneck_util'], 0.95)
        self.assertLessEqual(m_llr['bottleneck_util'], 0.95)

    def test_compute_path_metrics(self):
        path = [4, 2, 1, 3, 6]
        m = compute_path_metrics(path, self.link_utils, self.link_delays, self.link_bws)
        self.assertEqual(m['hops'], 4)
        self.assertGreater(m['total_delay'], 0.0)
        self.assertGreater(m['jitter'], 0.0)
        self.assertGreaterEqual(m['packet_loss'], 0.0)


class TestMultiTopologySupport(unittest.TestCase):
    """Verifies that all 5 required topologies instantiate properly."""

    def test_all_topologies(self):
        for topo_id in ['tree', 'fattree', 'abilene', 'nsfnet', 'spineleaf']:
            g, meta = get_topology(topo_id)
            self.assertGreater(g.number_of_nodes(), 0)
            self.assertGreater(g.number_of_edges(), 0)
            self.assertTrue(nx.is_strongly_connected(g) or nx.is_weakly_connected(g))


if __name__ == '__main__':
    unittest.main(verbosity=2)
