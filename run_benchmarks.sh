#!/usr/bin/env bash
# ==============================================================================
# Run All Routing Benchmarks and Stress Tests (EC499)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
elif [ -f "/home/maher/ec499_env/bin/python" ]; then
    PYTHON_BIN="/home/maher/ec499_env/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo " 1. Running 5-Topology Head-to-Head Tournament Benchmark..."
echo "======================================================================"
"$PYTHON_BIN" benchmark_routing_algorithms.py

echo "======================================================================"
echo " 2. Running High-Intensity Load Balancer Stress Test..."
echo "======================================================================"
"$PYTHON_BIN" stress_test_load_balancer.py

echo "======================================================================"
echo " 3. Running Multi-Topology Blind Stress Benchmark..."
echo "======================================================================"
"$PYTHON_BIN" stress_test_blind_topologies.py

echo "======================================================================"
echo " All Benchmarks and Stress Tests Completed Successfully!"
echo " Results and plots saved to logs/ and logs/plots/"
echo "======================================================================"
