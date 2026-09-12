# Reinforcement Learning for Adaptive Traffic Engineering in an SDN Network (EC499)

A high-performance SDN Traffic Engineering and Security platform built on **Ryu OpenFlow 1.3** and **PyTorch**. It utilizes three online Deep Reinforcement Learning agents to adaptively optimize unicast routing, multicast group delivery, and real-time DDoS attack mitigation inside a Mininet-emulated network fabric.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │            Mininet Emulated Network          │
                      │   (Hierarchical Tree Topo + Unicast/Multicast│
                      │         iperf + High-Rate DDoS Flood)        │
                      └──────────────────────┬───────────────────────┘
                                             │ OpenFlow 1.3
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │              Ryu Main Controller             │
                      │   • Topology Discovery (LLDP Link Probing)   │
                      │   • Intelligent ARP & Host Tracking Proxy    │
                      │   • Port & Flow Differential Rate Polling    │
                      │   • Active Controller-Switch Echo Latency    │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────┴───────────────────────┐
                      │                 StateManager                 │
                      │   • NetworkX Graph & Link Metric Tables      │
                      │   • Differential Mbps & PPS Rate Engine      │
                      │   • Real-Time Shannon Entropy Calculator     │
                      │   • Normalized RL Feature Extractors         │
                      └──────┬───────────────┼───────────────┬───────┘
                             │               │               │
                             ▼               ▼               ▼
                      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                      │SecurityMod  │ │MulticastMod │ │ RoutingMod  │
                      │(DDPG Agent) │ │(Dueling DQN)│ │ (Double DQN)│
                      └─────────────┘ └─────────────┘ └─────────────┘
```

---

## Reinforcement Learning Agent Specifications

| Domain | Module | RL Algorithm | State Space | Action Space | Reward Formulation | OpenFlow Enforcement |
|---|---|---|---|---|---|---|
| **Unicast Routing** | `RoutingModule` | **Double DQN** | 10-dim (Hop count, avg/max link utilization, latency, CPU/RAM, flow rates) | 4 discrete actions (K-shortest candidate paths) | $R = - (0.3 \cdot \text{Hops} + 0.05 \cdot \text{Delay} + \text{CongestionPen}) + \text{Advantage}$ | Multi-hop `OFPFlowMod` installed on every switch along computed path |
| **Multicast Delivery** | `MulticastModule` | **Dueling Double DQN** | 50-dim (Source ID, Receiver IDs, link utilization/delay matrix) | 10 discrete actions (Steiner Tree weight heuristics) | $R = 0.6 \cdot \text{BWSavings} - 0.7 \cdot \text{TreeEdges}$ | OpenFlow Group Tables (`OFPGT_ALL` multi-bucket replication) |
| **DDoS Security** | `SecurityModule` | **DDPG (Actor-Critic)** | 5-dim (Packet rate PPS, Shannon Entropy, Byte/Packet ratio, Active flows) | Continuous $\mu(s) \in [-1.0, 1.0]$ | $+2.0$ (True Pos), $+1.0$ (True Neg), $-1.5$ (Miss/False Alarm) | Priority 100 hardware `OFPFlowMod` DROP rule matching attacker IP |

---

## Performance Summary (10,000-Episode Converged Benchmark)

| Metric | Shortest Path First (SPF) Baseline | DRL Multi-Agent System | Net Improvement |
|---|---|---|---|
| **Core Jamming Bottleneck Load** | $97.6\%$ | **$53.5\%$** | **$+44.1\%$ Congestion Reduction** |
| **Degraded Link Latency** | $80.0\text{ ms}$ | **$28.8\text{ ms}$** | **$+51.2\text{ ms}$ Faster Dynamic Bypass** |
| **DDoS Detection Accuracy** | Heuristic: $\approx 85\%$ | **$100.0\%$** | **Zero False Drops ($0.0\%$ FP)** |
| **Multicast Bandwidth Conserved** | 0 Mbps (Unicast Streams) | **$58.6\text{ Mbps}$** | Core Link Conservation |
| **Controller Decision Speed** | N/A | **$2,300 - 2,800$ decisions/sec** | Sub-millisecond inference |

---

## Project Structure

```
├── agent/
│   ├── dqn_router.py                   # Double DQN with LayerNorm, target network, experience replay
│   ├── dqn_multicast.py                # Dueling DQN separating Value V(s) and Advantage A(s, a)
│   ├── ddpg_security.py                # DDPG Actor-Critic with continuous action space & Polyak updates
│   ├── prioritized_replay.py           # Proportional SumTree Prioritized Experience Replay Buffer
│   └── __init__.py                     # Agent package exports
│
├── controller/
│   ├── main_controller.py              # Ryu app entry point — OpenFlow 1.3 pipeline & stats monitor
│   ├── routing_module.py               # Unicast routing logic with Double DQN agent
│   ├── multicast_module.py             # Multicast routing logic with Dueling DQN & Steiner tree
│   ├── security_module.py              # DDoS detection & mitigation with DDPG continuous control
│   ├── state_manager.py                # Centralized state: differential Mbps/PPS rates, entropy, topology
│   ├── traditional_routing.py          # Classical baselines: Dijkstra SPF, ECMP, Widest Path (WSP), LLR
│   ├── web_dashboard.py                # REST telemetry API server (port 8080)
│   └── static/index.html               # Real-time topology canvas web UI
│
├── topology/
│   ├── mininet_topo.py                 # Hierarchical Tree Topo + automated unicast, multicast, and DDoS traffic
│   ├── topology_library.py             # Multi-topology generator (Tree, Fat-Tree, Abilene, NSFNet, Spine-Leaf, Random)
│   └── traffic_generator.sh            # Standalone traffic generation script (iperf background & DDoS flood)
│
├── models/
│   ├── dqn_router.pth                  # Trained PyTorch Double DQN Unicast weights
│   ├── dqn_multicast.pth               # Trained PyTorch Dueling DQN Multicast weights
│   └── ddpg_security.pth               # Trained PyTorch DDPG Continuous Security weights
│
├── logs/
│   ├── evaluation_results.json         # Latest benchmark evaluation JSON summary
│   ├── training_metrics.csv            # 10,000-episode training metric records
│   ├── ryu_controller.log              # Ryu controller execution & OpenFlow event logs
│   └── plots/                          # Publication-grade performance figures
│
├── docs/
│   └── Project_Report_EC499.md         # Comprehensive senior design technical project report
│
├── archive/
│   └── backup_3000_episodes/           # Archived checkpoints and logs from intermediate runs
│
├── benchmark_evaluation.py             # Full multi-agent curriculum training pipeline (1,000 to 10,000 episodes)
├── benchmark_routing_algorithms.py     # Rigorous comparative benchmark against SPF, ECMP, WSP, and LLR
├── evaluate_random_blind_topology.py   # Zero-shot evaluation engine on dynamically generated random topologies
├── run_random_blind_test.sh            # Quick launcher for blind topology evaluation
├── stress_test_blind_topologies.py     # Cross-topology stress test suite across 6 standard network fabrics
├── stress_test_load_balancer.py        # High-concurrency routing avalanche and core jamming stress tests
├── simulate_live_traffic.py            # Standalone simulation driver (No root / Mininet required)
├── test_suite.py                       # Full 21-test unit & integration validation suite
├── run_system.sh                       # Single-command launcher for Ryu + Mininet + Dashboard
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Requirements & Installation

> **Environment**: Linux (Ubuntu 20.04 or 22.04 LTS recommended). Requires Open vSwitch and Mininet.

### 1. System Packages
```bash
sudo apt-get update
sudo apt-get install -y openvswitch-switch mininet net-tools iperf hping3 python3-pip
```

### 2. Python Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Platform

### Automated Single-Command Startup (Ryu + Mininet + Web Dashboard)
```bash
chmod +x run_system.sh
./run_system.sh
```
* Access the Web Dashboard at: **`http://localhost:8080`**

### Running Without Root (Autonomous Simulation Driver)
```bash
python3 simulate_live_traffic.py --mode auto --interval 4.0
```

---

## Training & Benchmarking

### Multi-Agent Training Pipeline
```bash
# 1,000 episodes (~30 seconds)
python3 benchmark_evaluation.py 1000

# 10,000 episodes (~3.2 minutes, full curriculum)
python3 benchmark_evaluation.py 10000
```

### Comparative Routing Benchmark (DQN vs. SPF, ECMP, WSP, LLR)
```bash
python3 benchmark_routing_algorithms.py
```

### Zero-Shot Blind Topology Generalization Test
```bash
chmod +x run_random_blind_test.sh
./run_random_blind_test.sh
```

### Running the Test Suite
```bash
python3 test_suite.py
```

---

## Technical Documentation & Report

* **Full Senior Design Technical Report**: Detailed architecture, DRL formulations, convergence analysis, and security verification are documented in [docs/Project_Report_EC499.md](docs/Project_Report_EC499.md).
* **Publication Figures**: All high-resolution performance plots are available in [`logs/plots/`](logs/plots/).
