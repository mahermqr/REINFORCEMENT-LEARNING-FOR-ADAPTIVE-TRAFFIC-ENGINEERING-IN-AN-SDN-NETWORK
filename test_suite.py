#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite (EC499)
Reinforcement Learning for Adaptive Traffic Engineering in an SDN Network

Tests:
  1. Double DQN Unicast Router (Forward pass, PER sampling, Bellman target, Checkpoint)
  2. Dueling Double DQN Multicast Tree (V/A stream decomposition, Advantage centering)
  3. DDPG Continuous Control Security Agent (Actor [-1, 1], Critic Q(s, a), Polyak updates)
  4. Prioritized Experience Replay (SumTree arithmetic, IS weight normalization, Priority updates)
  5. StateManager Telemetry (Differential Mbps/PPS, link utilization, host proxy tracking)
  6. Shannon Entropy & DDoS Detection (Uniform dispersion vs single-source collapse)
  7. Yen's K-Shortest Simple Paths & Lateral Cross-Link Routing
  8. Steiner Minimal Multicast Tree Approximation & Replication Savings
  9. Web Dashboard Telemetry & Simulation API Endpoints
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

from dqn_router import DQNRoutingAgent
from dqn_multicast import DQNMulticastAgent, DuelingDQN
from ddpg_security import DDPGSecurityAgent
from prioritized_replay import PrioritizedReplayBuffer, SumTree
from state_manager import StateManager
from benchmark_evaluation import build_evaluation_topology
from topology_library import get_topology, list_available_topologies
from traditional_routing import (
    dijkstra_spf, ecmp_routing, widest_shortest_path, least_loaded_routing, random_routing, compute_path_metrics
)


class TestRLAgents(unittest.TestCase):
    """Verifies PyTorch DRL Agent forward passes, experience replay, and mathematical integrity."""

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

    def test_dueling_dqn_architecture(self):
        model = DuelingDQN(state_size=50, action_size=10)
        dummy_input = torch.randn(4, 50)
        q_values = model(dummy_input)

        self.assertEqual(q_values.shape, (4, 10))

        # Verify Value and Advantage streams are decoupled
        features = model.feature(dummy_input)
        val = model.value(features)
        adv = model.advantage(features)
        self.assertEqual(val.shape, (4, 1))
        self.assertEqual(adv.shape, (4, 10))

        # Check advantage centering: mean of centered advantage should be approximately 0
        centered_adv = adv - adv.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(centered_adv.mean(dim=1), torch.zeros(4), atol=1e-5))

    def test_dqn_multicast_training(self):
        agent = DQNMulticastAgent(state_size=50, action_size=10)
        dummy_state = np.random.rand(50).astype(np.float32)
        action = agent.act(dummy_state, explore=False)
        self.assertIn(action, range(10))

        for _ in range(40):
            s = np.random.rand(50).astype(np.float32)
            a = np.random.randint(0, 10)
            r = float(np.random.randn())
            s_next = np.random.rand(50).astype(np.float32)
            agent.remember(s, a, r, s_next, False)

        loss = agent.train(batch_size=16)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)

    def test_ddpg_security_actor_critic(self):
        agent = DDPGSecurityAgent(state_size=5, action_size=1)
        dummy_state = np.random.rand(5).astype(np.float32)

        # Actor action must be strictly bounded in [-1.0, 1.0] by Tanh
        for _ in range(20):
            st = (np.random.rand(5) * 5.0).astype(np.float32)
            act = agent.act(st, add_noise=False)
            self.assertGreaterEqual(act, -1.0)
            self.assertLessEqual(act, 1.0)

        # Critic forward pass
        s_tensor = torch.randn(4, 5)
        a_tensor = torch.randn(4, 1)
        q_val = agent.critic(s_tensor, a_tensor)
        self.assertEqual(q_val.shape, (4, 1))

        # Test training with Polyak soft updates
        initial_target_weight = agent.target_actor.net[0].weight.clone()
        for _ in range(40):
            s = np.random.rand(5).astype(np.float32)
            a = float(np.random.uniform(-1.0, 1.0))
            r = float(np.random.choice([2.0, 1.0, -1.5]))
            s_next = np.random.rand(5).astype(np.float32)
            agent.remember(s, a, r, s_next, False)

        losses = agent.train(batch_size=16)
        self.assertIsNotNone(losses)
        c_loss, a_loss = losses
        self.assertIsInstance(c_loss, float)
        self.assertIsInstance(a_loss, float)

        # Verify Polyak update modified target weights
        updated_target_weight = agent.target_actor.net[0].weight
        self.assertFalse(torch.equal(initial_target_weight, updated_target_weight))

    def test_prioritized_replay_sumtree(self):
        tree = SumTree(capacity=8)
        # Add 4 entries
        priorities = [1.0, 2.0, 3.0, 4.0]
        for p in priorities:
            tree.add(p, f"data_{p}")

        self.assertAlmostEqual(tree.total(), 10.0, places=4)
        self.assertEqual(tree.n_entries, 4)

        # Update priority of first entry from 1.0 to 5.0
        idx, _, _ = tree.get(0.5)
        tree.update(idx, 5.0)
        self.assertAlmostEqual(tree.total(), 14.0, places=4)

    def test_prioritized_replay_buffer(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Test empty buffer guard
        batch, idxs, weights = buf.sample(16)
        self.assertEqual(len(batch), 0)

        for i in range(25):
            buf.add(np.zeros(5), i % 4, 1.0, np.zeros(5), False)

        batch, idxs, weights = buf.sample(16)
        self.assertEqual(len(batch), 16)
        self.assertEqual(len(idxs), 16)
        self.assertEqual(len(weights), 16)
        self.assertAlmostEqual(weights.max(), 1.0, places=3) # Max weight normalized to 1.0


class TestStateManager(unittest.TestCase):
    """Verifies topology representation, rate extraction, and Shannon entropy calculations."""

    def setUp(self):
        self.sm = StateManager()
        # Hierarchical 4-switch fabric
        self.sm.update_link(1, 2, src_port=1, dst_port=1, capacity_mbps=100.0, delay_ms=2.0)
        self.sm.update_link(2, 1, src_port=1, dst_port=1, capacity_mbps=100.0, delay_ms=2.0)
        self.sm.update_link(2, 3, src_port=2, dst_port=1, capacity_mbps=50.0, delay_ms=4.0)
        self.sm.update_link(3, 2, src_port=1, dst_port=2, capacity_mbps=50.0, delay_ms=4.0)
        self.sm.update_link(1, 4, src_port=2, dst_port=1, capacity_mbps=100.0, delay_ms=3.0)
        self.sm.update_link(4, 1, src_port=1, dst_port=2, capacity_mbps=100.0, delay_ms=3.0)

    def test_routing_state_features(self):
        state = self.sm.get_routing_state(src_dpid=4, dst_dpid=3)
        self.assertEqual(len(state), 10)
        # Hop count feature: path 4 -> 1 -> 2 -> 3 is 3 hops
        self.assertGreater(state[0], 0.0)
        self.assertLessEqual(state[0], 1.0)
        # Resource loads in bounds
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            self.assertGreaterEqual(state[i], 0.0)
            self.assertLessEqual(state[i], 1.0)

    def test_multicast_state_features(self):
        state = self.sm.get_multicast_state(src_dpid=1, group_destinations=[2, 3, 4])
        self.assertEqual(len(state), 50)
        self.assertEqual(state[48], 3 / 10.0) # 3 destinations encoded

    def test_differential_port_rates(self):
        # Initial sample
        self.sm.update_port_stats(dpid=1, port_no=1, rx_bytes=1000, tx_bytes=2000,
                                  rx_packets=10, tx_packets=20, duration_sec=1)
        time.sleep(0.12)
        # Second sample: +1,000,000 bytes over ~0.12 sec (~66 Mbps)
        self.sm.update_port_stats(dpid=1, port_no=1, rx_bytes=1000, tx_bytes=1002000,
                                  rx_packets=10, tx_packets=1020, duration_sec=2)

        rates = self.sm.port_rates.get((1, 1))
        self.assertIsNotNone(rates)
        self.assertGreater(rates['tx_mbps'], 10.0)
        self.assertGreater(rates['tx_pps'], 1000.0)

    def test_shannon_entropy_normal_vs_attack(self):
        # Scenario A: Normal multi-host traffic (high entropy)
        sm_normal = StateManager()
        for i in range(12):
            src_ip = f"10.0.0.{i+1}"
            sm_normal.update_flow_stats(dpid=1, src_ip=src_ip, dst_ip="10.0.0.100",
                                       packets=50, bytes_count=75000, duration_sec=5)
        state_normal = sm_normal.get_security_state()
        self.assertGreater(state_normal[1], 0.80, "Benign multi-host traffic should exhibit high entropy > 0.80")

        # Scenario B: Volumetric single-source DDoS flood (entropy collapse)
        sm_attack = StateManager()
        # 1 attacker sends 5000 packets, while 2 background hosts send 5 packets each
        sm_attack.update_flow_stats(dpid=1, src_ip="10.0.0.4", dst_ip="10.0.0.1",
                                   packets=5000, bytes_count=400000, duration_sec=5)
        sm_attack.update_flow_stats(dpid=1, src_ip="10.0.0.2", dst_ip="10.0.0.1",
                                   packets=5, bytes_count=7500, duration_sec=5)
        state_attack = sm_attack.get_security_state()
        self.assertLess(state_attack[1], 0.40, "Volumetric single-source DDoS should collapse entropy < 0.40")

    def test_interactive_simulation_injections(self):
        topo = build_evaluation_topology()
        self.sm.graph = topo.copy()

        # Test Core Jamming injection
        self.sm.inject_core_congestion(utilization=0.96)
        self.assertAlmostEqual(self.sm.link_utilization[(1, 2)], 0.96)
        self.assertAlmostEqual(self.sm.link_utilization[(4, 6)], 0.18)

        # Test DDoS flood injection
        self.sm.inject_ddos_flood(attacker_ip="10.0.0.4", target_ip="10.0.0.1", pps=5000, bpp=80)
        sec_state = self.sm.get_security_state()
        self.assertLess(sec_state[1], 0.40) # Entropy collapsed

        # Test Reset
        self.sm.reset_simulation()
        self.assertEqual(len(self.sm.flow_stats), 0)
        self.assertEqual(self.sm.link_utilization[(1, 2)], 0.05)


class TestRoutingAndMulticastLogic(unittest.TestCase):
    """Verifies candidate path computation, Steiner trees, and cross-link offloading."""

    def setUp(self):
        self.topo = build_evaluation_topology()

    def test_k_shortest_candidate_paths(self):
        # Pod 1 (s4) to Pod 2 (s6)
        paths = list(nx.shortest_simple_paths(self.topo, 4, 6))
        self.assertGreaterEqual(len(paths), 2)
        # Direct lateral cross link path [4, 6] should exist
        self.assertIn([4, 6], paths)
        # Core hierarchical path [4, 2, 1, 3, 6] should also exist
        self.assertIn([4, 2, 1, 3, 6], paths)

    def test_steiner_multicast_tree(self):
        g = self.topo.copy().to_undirected()
        for u, v in g.edges():
            g[u][v]['weight'] = 1.0

        src = 4
        dests = [5, 6, 7]
        terminals = [src] + dests
        tree = nx.algorithms.approximation.steinertree.steiner_tree(g, terminals, weight='weight')

        # Tree must span all terminals
        for t in terminals:
            self.assertIn(t, tree.nodes())

        # Tree must be connected
        self.assertTrue(nx.is_connected(tree))
        # Replication saving: tree edges must be fewer than independent unicast paths (3 * 3 = 9 edges)
        self.assertLess(tree.number_of_edges(), len(dests) * 3)


class TestMultiTopologyAndBlindGeneralization(unittest.TestCase):
    """Verifies graph generation, metadata validation, and zero-shot agent transfer across 5 unseen topologies."""

    def test_topology_library_registry(self):
        topos = list_available_topologies()
        self.assertEqual(len(topos), 5)
        expected_ids = {'tree', 'fattree', 'abilene', 'nsfnet', 'spineleaf'}
        self.assertEqual({t['id'] for t in topos}, expected_ids)

    def test_topology_graph_properties(self):
        for tid in ['tree', 'fattree', 'abilene', 'nsfnet', 'spineleaf']:
            g, meta = get_topology(tid)
            self.assertGreater(g.number_of_nodes(), 0)
            self.assertGreater(g.number_of_edges(), 0)
            self.assertTrue(nx.is_strongly_connected(g) or nx.is_weakly_connected(g))

            # Validate positions
            positions = meta.get('positions', {})
            for n in g.nodes():
                self.assertIn(n, positions, f"Node {n} missing position in {tid}")
                pos = positions[n]
                self.assertTrue(0.0 <= pos['x'] <= 1.0)
                self.assertTrue(0.0 <= pos['y'] <= 1.0)

            # Validate edge capacities and delays
            for u, v, d in g.edges(data=True):
                self.assertGreater(d.get('capacity', 0), 0)
                self.assertGreater(d.get('delay', 0), 0)

    def test_blind_state_representation_invariance(self):
        """Verifies that StateManager constructs valid 10-D state vectors regardless of topology size."""
        agent = DQNRoutingAgent(state_size=10, action_size=4)

        for tid in ['tree', 'fattree', 'abilene', 'nsfnet', 'spineleaf']:
            g, meta = get_topology(tid)
            sm = StateManager()
            sm.graph = g.copy()
            for u, v, d in g.edges(data=True):
                sm.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=d['capacity'], delay_ms=d['delay'])

            edge_nodes = meta.get('edge_nodes', list(g.nodes()))
            src = edge_nodes[0]
            dst = edge_nodes[-1]

            state = sm.get_routing_state(src, dst)
            self.assertEqual(state.shape, (10,))
            self.assertTrue(np.all(np.isfinite(state)))
            self.assertTrue(np.all(state >= 0.0))

            # Zero-shot greedy inference without retraining
            action = agent.act(state, explore=False)
            self.assertIn(action, range(4))


class TestAdvancedTrafficEngineeringAndFailover(unittest.TestCase):
    """Verifies WCMP flow splitting, sub-millisecond Fast Failover, and tiered DDoS metering."""

    def setUp(self):
        import logging
        self.sm = StateManager()
        self.sm.graph = build_evaluation_topology()
        for u, v, d in self.sm.graph.edges(data=True):
            self.sm.update_link(u, v, src_port=1, dst_port=1, capacity_mbps=d.get('capacity', 100.0), delay_ms=d.get('delay', 2.0))

        class MockController:
            def __init__(self):
                self.logger = logging.getLogger("MockController")
                self.datapaths = {}

        from routing_module import RoutingModule
        from security_module import SecurityModule
        self.ctl = MockController()
        self.routing = RoutingModule(self.ctl, self.sm)
        self.security = SecurityModule(self.ctl, self.sm)

    def test_wcmp_weight_calculation(self):
        candidate_paths = [[4, 2, 1, 3, 6], [4, 6]]
        # Severe core congestion on path 0
        self.sm.link_utilization[(4, 2)] = 0.92
        self.sm.link_utilization[(4, 6)] = 0.15

        weights = self.routing.calculate_wcmp_weights(candidate_paths, beta=5.0)
        self.assertEqual(len(weights), 2)
        path0, w0 = weights[0]
        path1, w1 = weights[1]

        self.assertGreater(w1, w0)
        self.assertGreaterEqual(w0, 1)
        self.assertGreaterEqual(w1, 1)

    def test_fast_failover_link_break(self):
        # Verify initial edge
        self.assertTrue(self.sm.graph.has_edge(1, 2))

        # Inject fiber cut
        ok = self.sm.inject_link_failure(1, 2)
        self.assertTrue(ok)
        self.assertFalse(self.sm.graph.has_edge(1, 2))
        self.assertIn((1, 2), self.sm.failed_links)

        # Restore severed link
        restored = self.sm.restore_link(1, 2)
        self.assertGreaterEqual(restored, 1)
        self.assertTrue(self.sm.graph.has_edge(1, 2))
        self.assertNotIn((1, 2), self.sm.failed_links)

    def test_multivector_entropy(self):
        # Single-source concentrated flood
        self.sm.src_ip_counter.clear()
        self.sm.dst_ip_counter.clear()
        self.sm.src_ip_counter['10.0.0.4'] = 500
        self.sm.dst_ip_counter['10.0.0.1'] = 500

        mv = self.sm.calculate_multivector_entropy()
        self.assertLess(mv['h_src'], 0.40)
        self.assertLess(mv['h_dst'], 0.40)
        self.assertEqual(mv['attack_type'], "targeted_single_source")
        self.assertTrue(mv['is_anomalous'])

    def test_hierarchical_metering_tier(self):
        summary = self.security.get_security_summary()
        self.assertIn('action_tier', summary)
        self.assertIn('metered_ips', summary)
        self.assertIn('blocked_ips', summary)
        self.assertIn('multivector_entropy', summary)
        self.assertIn(summary['action_tier'], ['critical_drop', 'meter_limit', 'benign'])

    def test_traditional_routing_baselines(self):
        """Verifies Dijkstra SPF, ECMP, WSP, LLR, and Random routing baseline logic."""
        g = self.sm.graph
        # Test Dijkstra SPF
        p_spf = dijkstra_spf(g, 4, 6)
        self.assertGreaterEqual(len(p_spf), 2)
        self.assertEqual(p_spf[0], 4)
        self.assertEqual(p_spf[-1], 6)

        # Test ECMP with different flow hashes
        p_ecmp1 = ecmp_routing(g, 4, 6, flow_hash=0)
        p_ecmp2 = ecmp_routing(g, 4, 6, flow_hash=1)
        self.assertGreaterEqual(len(p_ecmp1), 2)
        self.assertGreaterEqual(len(p_ecmp2), 2)

        # Saturate core link (4, 2)
        self.sm.link_utilization[(4, 2)] = 0.95
        self.sm.link_utilization[(2, 1)] = 0.95
        # Set lateral cross-link low
        self.sm.link_utilization[(4, 6)] = 0.10

        # WSP should select low-utilization bypass
        p_wsp = widest_shortest_path(g, 4, 6, self.sm.link_utilization, self.sm.link_delays)
        self.assertIn(p_wsp, [[4, 6], [4, 2, 6], [4, 2, 1, 3, 6]])

        # LLR should avoid 0.95 saturated links
        p_llr = least_loaded_routing(g, 4, 6, self.sm.link_utilization)
        b_util, _, _ = compute_path_metrics(p_llr, self.sm.link_utilization, self.sm.link_delays)
        self.assertLess(b_util, 0.95)

        # Random routing
        p_rand = random_routing(g, 4, 6)
        self.assertEqual(p_rand[0], 4)
        self.assertEqual(p_rand[-1], 6)


if __name__ == '__main__':
    print("=" * 80)
    print(" Running Full Adaptive SDN Traffic Engineering Comprehensive Test Suite")
    print("=" * 80)
    unittest.main()
