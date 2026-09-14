#!/usr/bin/env python3
"""
Unified SDN Topology & Traffic Evaluation Suite for Reinforcement Learning
Features:
  - Multi-Topology Support:
      1. Hierarchical Tree Topology (7 switches, 8 hosts, redundant mesh links)
      2. Fat-Tree Topology (k=4: 4 Core, 8 Aggregation, 8 Edge switches, 16 hosts)
  - Adaptive background unicast traffic generator (iperf)
  - Multicast streaming simulation (UDP multicast group 224.1.1.1)
  - High-rate DDoS flooding attack simulation (iperf UDP flood)
"""

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

import time
import threading
import argparse

class TreeTopology(Topo):
    """
    Hierarchical Tree Topology:
      - 1 Core switch (s1)
      - 2 Aggregation switches (s2, s3)
      - 4 Edge switches (s4, s5, s6, s7)
      - 8 End hosts (h1 - h8)
    """
    def build(self):
        info("*** Creating Core, Aggregation, and Edge Switches\n")
        s1 = self.addSwitch('s1', dpid='0000000000000001')
        s2 = self.addSwitch('s2', dpid='0000000000000002')
        s3 = self.addSwitch('s3', dpid='0000000000000003')
        s4 = self.addSwitch('s4', dpid='0000000000000004')
        s5 = self.addSwitch('s5', dpid='0000000000000005')
        s6 = self.addSwitch('s6', dpid='0000000000000006')
        s7 = self.addSwitch('s7', dpid='0000000000000007')

        # Inter-switch links with bandwidth and delay attributes
        info("*** Creating Inter-switch Links\n")
        self.addLink(s1, s2, bw=100, delay='2ms')
        self.addLink(s1, s3, bw=100, delay='2ms')
        self.addLink(s2, s4, bw=50, delay='3ms')
        self.addLink(s2, s5, bw=50, delay='3ms')
        self.addLink(s3, s6, bw=50, delay='3ms')
        self.addLink(s3, s7, bw=50, delay='3ms')

        # Redundant mesh cross-links to provide alternate routing paths for DQN Unicast
        self.addLink(s4, s6, bw=30, delay='8ms')
        self.addLink(s5, s7, bw=30, delay='8ms')

        # Hosts attached to edge switches
        info("*** Creating Hosts and Host Links\n")
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04') # Attacker host
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.0.0.6/24', mac='00:00:00:00:00:06')
        h7 = self.addHost('h7', ip='10.0.0.7/24', mac='00:00:00:00:00:07')
        h8 = self.addHost('h8', ip='10.0.0.8/24', mac='00:00:00:00:00:08')

        self.addLink(h1, s4, bw=100, delay='1ms')
        self.addLink(h2, s4, bw=100, delay='1ms')
        self.addLink(h3, s5, bw=100, delay='1ms')
        self.addLink(h4, s5, bw=100, delay='1ms')
        self.addLink(h5, s6, bw=100, delay='1ms')
        self.addLink(h6, s6, bw=100, delay='1ms')
        self.addLink(h7, s7, bw=100, delay='1ms')
        self.addLink(h8, s7, bw=100, delay='1ms')


class FatTreeTopology(Topo):
    """
    Fat-Tree Topology (k=4):
      - 4 Core switches
      - 4 Pods (each with 2 Aggregation and 2 Edge switches)
      - 16 End hosts
    """
    def build(self, k=4):
        info("*** Creating Fat-Tree (k=4) Switches and Links\n")
        cores = []
        for i in range(1, (k//2)**2 + 1):
            cores.append(self.addSwitch(f'c{i}', dpid=f'000000000000010{i}'))

        agg_switches = []
        edge_switches = []
        host_id = 1

        for pod in range(k):
            pod_aggs = []
            pod_edges = []
            for a in range(k//2):
                sw_num = pod * 2 + a + 1
                pod_aggs.append(self.addSwitch(f'a{sw_num}', dpid=f'000000000000020{sw_num}'))
            for e in range(k//2):
                sw_num = pod * 2 + e + 1
                pod_edges.append(self.addSwitch(f'e{sw_num}', dpid=f'000000000000030{sw_num}'))

            # Connect Agg to Edge within Pod
            for agg in pod_aggs:
                for edge in pod_edges:
                    self.addLink(agg, edge, bw=100, delay='2ms')

            # Connect Core to Agg
            for i, agg in enumerate(pod_aggs):
                for j in range(k//2):
                    core_idx = i * (k//2) + j
                    self.addLink(cores[core_idx], agg, bw=100, delay='1ms')

            # Add hosts to Edge switches
            for edge in pod_edges:
                for h_idx in range(k//2):
                    host = self.addHost(f'h{host_id}', ip=f'10.0.{pod+1}.{host_id}/24',
                                        mac=f'00:00:00:00:00:{host_id:02x}')
                    self.addLink(host, edge, bw=100, delay='1ms')
                    host_id += 1


def start_unicast_traffic(net):
    """Generates normal background unicast traffic."""
    info("\n[Traffic] Starting Background Unicast Traffic (iperf streams)...\n")
    try:
        h1 = net.get('h1')
        h2 = net.get('h2')
        h3 = net.get('h3')
        h5 = net.get('h5')
        h6 = net.get('h6')

        h1.cmd('iperf -s -u -p 5001 &')
        h5.cmd('iperf -s -u -p 5002 &')

        h2.cmd('iperf -c %s -u -p 5001 -b 15M -t 35 &' % h1.IP())
        h3.cmd('iperf -c %s -u -p 5002 -b 10M -t 35 &' % h5.IP())
        h6.cmd('iperf -c %s -u -p 5001 -b 8M -t 35 &' % h1.IP())
        info("[Traffic] Background unicast streams active.\n")
    except Exception as e:
        info(f"[Traffic] Unicast generation: {e}\n")


def start_multicast_traffic(net):
    """Generates multicast traffic stream to group 224.1.1.1."""
    info("\n[Traffic] Starting Multicast Streaming to group 224.1.1.1...\n")
    try:
        h2 = net.get('h2')
        h5 = net.get('h5')
        h7 = net.get('h7')
        h8 = net.get('h8')

        h5.cmd('iperf -s -u -B 224.1.1.1 -p 5005 &')
        h7.cmd('iperf -s -u -B 224.1.1.1 -p 5005 &')
        h8.cmd('iperf -s -u -B 224.1.1.1 -p 5005 &')

        h2.cmd('iperf -c 224.1.1.1 -u -p 5005 -b 12M -t 30 &')
        info("[Traffic] Multicast stream transmitting to {h5, h7, h8}.\n")
    except Exception as e:
        info(f"[Traffic] Multicast generation: {e}\n")


def simulate_ddos_attack(net):
    """Simulates a high-rate DDoS flood from attacker host h4 towards target h1."""
    info("\n[Security] Launching Simulated DDoS Flooding Attack from h4 -> h1...\n")
    try:
        attacker = net.get('h4')
        target = net.get('h1')
        attacker.cmd('iperf -c %s -u -p 5001 -b 80M -t 25 &' % target.IP())
        info("[Security] DDoS Flooding attack active (80 Mbps UDP flood).\n")
    except Exception as e:
        info(f"[Security] DDoS simulation: {e}\n")


def run_network(topo_choice='tree'):
    setLogLevel('info')

    if topo_choice == 'fattree':
        info("*** Selected Topology: FatTree (k=4: 20 switches, 16 hosts)\n")
        topo = FatTreeTopology(k=4)
    else:
        info("*** Selected Topology: Hierarchical Tree (7 switches, 8 hosts, redundant mesh)\n")
        topo = TreeTopology()

    info("*** Initializing Mininet Network with Remote Ryu Controller on 127.0.0.1:6633\n")
    net = Mininet(
        topo=topo,
        switch=OVSSwitch,
        controller=RemoteController('c0', ip='127.0.0.1', port=6633),
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )

    net.start()
    info("\n*** Network Started successfully! Waiting 5s for LLDP discovery...\n")
    time.sleep(5)

    info("*** Performing Initial Host Ping Test...\n")
    try:
        net.ping([net.get('h1'), net.get('h2')])
    except Exception:
        pass

    # Launch Traffic Scenarios in background threads
    t1 = threading.Thread(target=start_unicast_traffic, args=(net,))
    t2 = threading.Thread(target=start_multicast_traffic, args=(net,))
    t3 = threading.Thread(target=simulate_ddos_attack, args=(net,))

    t1.start()
    time.sleep(2)
    t2.start()
    time.sleep(3)
    t3.start()

    info("\n*** Dropping into interactive Mininet CLI. Type 'exit' to stop.\n")
    CLI(net)

    info("*** Stopping Network\n")
    net.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive SDN Traffic Engineering Mininet Runner")
    parser.add_argument('--topo', choices=['tree', 'fattree'], default='tree', help='Topology selection (tree or fattree)')
    args = parser.parse_args()
    run_network(args.topo)
