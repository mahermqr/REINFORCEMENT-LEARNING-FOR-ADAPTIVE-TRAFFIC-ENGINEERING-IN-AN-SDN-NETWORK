# Reinforcement Learning for Adaptive Traffic Engineering in an SDN Network (EC499)

**Course**: EC499 — Graduation Project in Computer Engineering  
**Institution**: Department of Computer Engineering, Faculty of Engineering, University of Tripoli  
**Student**: Maher Abdulnasir Alqadhi (ID: 2210249576, `ma.alqadhi@uot.edu.ly`)  
**Supervisor**: Dr. Suad El-Geder  
**Term**: Spring 2026  

---

## Project Overview

This repository contains the complete implementation and benchmark suite for the graduation project: **Reinforcement Learning for Adaptive Traffic Engineering in an SDN Network**.

Traditional IP routing protocols (such as OSPF, Dijkstra SPF, or ECMP) route traffic using static hop counts or fixed administrative weights. Under dynamic traffic surges, this causes severe core switch bottlenecks, high queuing delays, packet jitter, and buffer overflow loss.

This platform integrates a **Dueling Double Deep Q-Network (D3QN)** with **Prioritized Experience Replay (PER)** into a **Ryu OpenFlow 1.3 SDN Controller** to adaptively optimize unicast traffic engineering. The agent continuously senses network link loads, delays, and packet rates, dynamically steering traffic over underutilized lateral cross-links to maximize throughput, minimize latency, suppress jitter, and eliminate packet loss.

---

## Fulfillment of Graduation Project Proposal

| Proposal Item | Approved Specification | Implementation Detail |
|---|---|---|
| **Objective 1** | Design a Deep Q-Network (DQN) agent to automate dynamic routing decisions | Dueling Double Deep Q-Network (`agent/dqn_router.py`) with Layer Normalization and Prioritized Experience Replay (`agent/prioritized_replay.py`) |
| **Objective 2** | Integrate RL agent with Ryu SDN controller using Mininet emulator | Ryu OpenFlow 1.3 controller (`controller/main_controller.py`) controlling Mininet fabrics (`topology/mininet_topo.py`) |
| **Objective 3** | Reduce network latency and jitter compared to standard routing protocols | Real-time tracking of path latency and RFC 3393 packet delay variation (`controller/state_manager.py`) |
| **Objective 4** | Maximize total network throughput by optimizing link utilization levels | Asymptotic barrier reward penalty preventing core bottlenecks; verified by Jain's Fairness Index |
| **Objective 5** | Evaluate performance using packet loss and control overhead metrics | M/M/1/K buffer overflow loss model and OpenFlow message accounting (`OFPPacketIn`, `OFPFlowMod`, decision latency) |
| **Procedure 4** | Train DQN agent using synthetic traffic patterns to simulate load | Poisson burst generator, elephant flows, and core jamming stress tests (`topology/traffic_generator.sh`) |
| **Procedure 5** | Benchmark RL agent against OSPF and greedy routing baselines | Automated tournament benchmark (`benchmark_routing_algorithms.py`) comparing DQN against OSPF (RFC 2328), Dijkstra SPF, ECMP, WSP, and LLR |
| **Procedure 6** | Document findings and prepare final technical report and source code | Complete technical report (`docs/Project_Report_EC499.md`), 18 passing unit tests (`test_suite.py`), and publication plots in `logs/plots/` |

---

## Head-to-Head Benchmark Summary across 5 Topologies

Evaluated across **Hierarchical Tree**, **Fat-Tree ($k=4$)**, **Abilene US Backbone**, **NSFNet Continental Mesh**, and **Spine-Leaf Fabrics**:

| Metric | OSPF (RFC 2328) | Dijkstra SPF | ECMP | Greedy LLR | DQN Traffic Engineering (Ours) | Net Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bottleneck Link Load** | 49.4% – 94.0% | 23.2% – 94.0% | 23.2% – 92.8% | 18.2% – 89.5% | **23.2% – 89.5%** | **Up to +26.2% Congestion Relief** |
| **End-to-End Latency** | 20.7 – 60.5 ms | 11.6 – 56.2 ms | 11.5 – 52.8 ms | 13.0 – 18.6 ms | **11.6 – 20.7 ms** | **73.5% Faster Transmission** |
| **Network Jitter (RFC 3393)** | 28.7 – 127.5 ms | 1.1 – 117.9 ms | 1.1 – 107.1 ms | 1.1 – 42.1 ms | **1.1 – 46.5 ms** | **> 97% Jitter Suppression** |
| **Packet Loss Rate** | 2.2% – 18.8% | 0.01% – 18.8% | 0.01% – 17.0% | 0.01% – 11.8% | **0.01% – 14.8%** | **Near-Zero Dropouts Under Bursts** |
| **Offload Rate** | 0.0% (Static) | 0.0% (Base) | 1.3% – 74.7% | 9.3% – 94.0% | **25.3% – 100.0%** | **Proactive Jamming Avoidance** |
| **Decision Latency** | 0.08 – 0.11 ms | 0.07 – 0.09 ms | 0.12 – 0.15 ms | 0.35 – 0.42 ms | **0.38 – 0.43 ms** | **> 2,400 Decisions/Second** |

---

## Project Structure

```
.
├── agent/
│   ├── dqn_router.py                   # Dueling Double Deep Q-Network (D3QN) with LayerNorm
│   ├── prioritized_replay.py           # SumTree Prioritized Experience Replay (PER) buffer
│   └── __init__.py                     # Package exports
│
├── controller/
│   ├── main_controller.py              # Ryu OpenFlow 1.3 application & control overhead monitor
│   ├── routing_module.py               # Unicast traffic engineering & reward calculation
│   ├── state_manager.py                # Telemetry engine (Throughput, Latency, Jitter, Loss, Control)
│   ├── traditional_routing.py          # Baselines: OSPF RFC 2328, Dijkstra SPF, ECMP, WSP, LLR
│   ├── web_dashboard.py                # Embedded REST API server (port 8080)
│   └── static/index.html               # Real-time topology canvas web UI
│
├── topology/
│   ├── mininet_topo.py                 # Mininet topologies (Tree, Fat-Tree) with OpenFlow 1.3 switches
│   ├── topology_library.py             # Multi-topology generator (Tree, Fat-Tree, Abilene, NSFNet, Spine-Leaf)
│   └── traffic_generator.sh            # Synthetic traffic generator (iperf background, elephant & bursts)
│
├── models/
│   └── dqn_router.pth                  # Trained PyTorch Double DQN weights
│
├── logs/
│   ├── routing_tournament_results.json # Quantitative benchmark tournament results
│   ├── blind_topologies_stress_results.json # 5-topology stress testing results
│   ├── stress_test_results.json        # Load balancer stress test results
│   ├── evaluation_results.json         # DQN training convergence summary
│   ├── training_metrics.csv            # Episode-by-episode training metric records
│   └── plots/                          # Publication-grade figures
│       ├── proposal_benchmarks_all_metrics.png # 6-panel all-metrics comparison
│       ├── proposal_tournament_radar.png       # Radar chart comparing algorithms
│       ├── dqn_te_training_convergence.png    # Training progression dashboard
│       ├── blind_topologies_stress_benchmark.png # 5-topology stress comparison
│       ├── blind_topologies_radar.png          # Blind topology radar
│       └── stress_test_load_balancing.png      # Load balancing stress test plot
│
├── docs/
│   └── Project_Report_EC499.md         # Final Technical Graduation Report
│
├── test_suite.py                       # Unit & integration test suite (18 tests, 100% passing)
├── benchmark_routing_algorithms.py     # 5-topology head-to-head tournament benchmark
├── benchmark_evaluation.py             # 3-phase curriculum DQN training script
├── evaluate_random_blind_topology.py   # Zero-shot random dynamic topology evaluation
├── stress_test_blind_topologies.py     # Multi-topology high-intensity stress suite
├── stress_test_load_balancer.py        # Core jamming and avalanche stress suite
├── simulate_live_traffic.py            # Live traffic driver and simulation runner
├── run_system.sh                       # One-command system launcher
├── run_random_blind_test.sh            # Quick launcher for zero-shot testing
├── CITATION.cff                        # Academic citation metadata
├── requirements.txt                    # Python package dependencies
└── LICENSE                             # MIT Open-Source License
```

---

## Quick Start & Verification

### 1. Run Unit & Integration Tests (18 Tests)
```bash
/home/maher/ec499_env/bin/python test_suite.py -v
```

### 2. Run Head-to-Head Routing Tournament Benchmark (5 Topologies)
```bash
/home/maher/ec499_env/bin/python benchmark_routing_algorithms.py
```

### 3. Run Multi-Topology Stress Testing Suite
```bash
/home/maher/ec499_env/bin/python stress_test_blind_topologies.py
```

### 4. Zero-Shot Blind Random Topology Test
```bash
/home/maher/ec499_env/bin/python evaluate_random_blind_topology.py --nodes 20 --flows 200
```

### 5. Train Deep Q-Network Agent
```bash
/home/maher/ec499_env/bin/python benchmark_evaluation.py 1000
```

### 6. Launch Ryu Controller, Live Traffic Simulation, and Web Dashboard
```bash
./run_system.sh
```

### 7. Access Interactive Telemetry Web Dashboard
Open your browser at:
```
http://localhost:8080
```

---

## Citation

If you use this work in your research or project, please cite:

```bibtex
@misc{alqadhi2026sdnrl,
  author = {Maher Abdulnasir Alqadhi and Dr. Suad El-Geder},
  title = {Reinforcement Learning for Adaptive Traffic Engineering in an SDN Network},
  year = {2026},
  publisher = {Department of Computer Engineering, Faculty of Engineering, University of Tripoli},
  howpublished = {\url{https://github.com/maheralqadhi/EC499-SDN-Adaptive-TE-RL}}
}
```
