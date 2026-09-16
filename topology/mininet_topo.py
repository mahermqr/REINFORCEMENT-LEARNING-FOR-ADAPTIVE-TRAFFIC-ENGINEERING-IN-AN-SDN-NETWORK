#!/usr/bin/env python3
"""
Unified SDN Topology & Traffic Evaluation Suite for Reinforcement Learning
Features:
  - Multi-Topology Support:
      1. Hierarchical Tree Topology (7 switches, 8 hosts, redundant mesh links)
      2. Fat-Tree Topology (k=4: 4 Core, 8 Aggregation, 8 Edge switches, 16 hosts)
  - Adaptive background unicast traffic generator (iperf mice flows)
  - Bursty elephant flows to stress-test core link utilization and adaptive rerouting
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

        # Hosts attached to edge switches (2 hosts per edge switch)
        info("*** Creating Hosts and Host Links\n")
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
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
    Standard Fat-Tree Topology (k=4):
    - 4 Core Switches (c1..c4)
    - 4 Pods, each with 2 Aggregation (a1..a8) and 2 Edge (e1..e8) switches
    - 16 End Hosts (h1..h16)
    """
    def build(self, k=4):
        num_cores = (k // 2) ** 2
        num_pods = k
        num_aggr_per_pod = k // 2
        num_edge_per_pod = k // 2
        num_hosts_per_edge = k // 2

        core_switches = []
        for i in range(1, num_cores + 1):
            dpid = f"{i:016x}"
            sw = self.addSwitch(f'c{i}', dpid=dpid)
            core_switches.append(sw)

        host_id = 1
        dpid_counter = 100
        for pod in range(num_pods):
            aggr_switches = []
            edge_switches = []

            for a in range(num_aggr_per_pod):
                dpid_counter += 1
                sw = self.addSwitch(f'a{pod}_{a}', dpid=f"{dpid_counter:016x}")
                aggr_switches.append(sw)

            for e in range(num_edge_per_pod):
                dpid_counter += 1
                sw = self.addSwitch(f'e{pod}_{e}', dpid=f"{dpid_counter:016x}")
                edge_switches.append(sw)

            for e_idx, e_sw in enumerate(edge_switches):
                for a_sw in aggr_switches:
                    self.addLink(e_sw, a_sw, bw=50, delay='2ms')

                for h in range(num_hosts_per_edge):
                    host_ip = f"10.{pod}.{e_idx}.{h+2}/24"
                    host_mac = f"00:00:00:{pod:02x}:{e_idx:02x}:{h+2:02x}"
                    host = self.addHost(f'h{host_id}', ip=host_ip, mac=host_mac)
                    self.addLink(host, e_sw, bw=100, delay='1ms')
                    host_id += 1

            for a_idx, a_sw in enumerate(aggr_switches):
                for c in range(k // 2):
                    core_idx = a_idx * (k // 2) + c
                    self.addLink(a_sw, core_switches[core_idx], bw=100, delay='2ms')


def start_unicast_traffic(net):
    """Generates continuous background unicast mice flows (HTTP/RPC traffic)."""
    info("\n[Traffic] Starting Background Mice Flows (Web/RPC lightweight streams)...\n")
    try:
        h1 = net.get('h1')
        h2 = net.get('h2')
        h3 = net.get('h3')
        h5 = net.get('h5')
        h6 = net.get('h6')

        h1.cmd('iperf -s -u -p 5001 &')
        h5.cmd('iperf -s -u -p 5002 &')

        h2.cmd('iperf -c %s -u -p 5001 -b 15M -t 45 &' % h1.IP())
        h3.cmd('iperf -c %s -u -p 5002 -b 10M -t 45 &' % h5.IP())
        h6.cmd('iperf -c %s -u -p 5001 -b 8M -t 45 &' % h1.IP())
        info("[Traffic] Background mice flows active.\n")
    except Exception as e:
        info(f"[Traffic] Background traffic error: {e}\n")


def start_elephant_flows(net):
    """Generates high-bandwidth bursty elephant flows to induce core link congestion."""
    info("\n[Traffic] Starting Bursty Elephant Flows (Heavy cross-pod transfers)...\n")
    try:
        h4 = net.get('h4')
        h8 = net.get('h8')

        h8.cmd('iperf -s -u -p 5004 &')
        time.sleep(1)
        # High-rate burst to saturate primary core link and trigger DQN rerouting
        h4.cmd('iperf -c %s -u -p 5004 -b 45M -t 35 &' % h8.IP())
        info("[Traffic] Elephant flow active (h4 -> h8 @ 45 Mbps).\n")
    except Exception as e:
        info(f"[Traffic] Elephant flow error: {e}\n")


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

    # Launch Synthetic Traffic Scenarios in background threads
    t1 = threading.Thread(target=start_unicast_traffic, args=(net,))
    t2 = threading.Thread(target=start_elephant_flows, args=(net,))

    t1.start()
    time.sleep(3)
    t2.start()

    info("\n*** Dropping into interactive Mininet CLI. Type 'exit' to stop.\n")
    CLI(net)

    info("*** Stopping Network\n")
    net.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive SDN Traffic Engineering Mininet Runner")
    parser.add_argument('--topo', choices=['tree', 'fattree'], default='tree', help='Topology selection (tree or fattree)')
    args = parser.parse_args()
    run_network(args.topo)
