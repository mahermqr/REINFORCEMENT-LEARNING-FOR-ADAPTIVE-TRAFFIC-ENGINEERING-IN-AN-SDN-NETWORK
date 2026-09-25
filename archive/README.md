# Archive: Legacy Artifacts & Historical Logs

This directory contains deprecated, historical, or superseded artifacts from earlier development iterations of the **EC499 Adaptive SDN Traffic Engineering** platform.

---

## Archived Items

| Item | Original Path | Date Archived | Reason for Archival |
| :--- | :--- | :--- | :--- |
| `legacy_logs/routing_algorithms_benchmark.json` | `logs/routing_algorithms_benchmark.json` | 2026-09-25 | Early baseline benchmark log (dated Sep 14, 2026). Superseded by the comprehensive 5-topology head-to-head tournament results in `logs/routing_tournament_results.json`. |
| `legacy_logs/ryu_controller.log` | `logs/ryu_controller.log` | 2026-09-25 | Historical runtime controller trace (dated Sep 18, 2026) capturing an aborted launch (`Errno 98: Address already in use`). Kept for historical debugging reference. |

---

## Active Equivalents in Repository

- **Routing Tournament Metrics**: Active results are located at [`logs/routing_tournament_results.json`](../logs/routing_tournament_results.json).
- **Stress Test Metrics**: Active results are located at [`logs/stress_test_results.json`](../logs/stress_test_results.json) and [`logs/blind_topologies_stress_results.json`](../logs/blind_topologies_stress_results.json).
- **Training Progression**: Episode metrics and rewards are logged to [`logs/training_metrics.csv`](../logs/training_metrics.csv).
- **Controller Execution**: Fresh controller execution outputs are generated at runtime by [`run_system.sh`](../run_system.sh).
