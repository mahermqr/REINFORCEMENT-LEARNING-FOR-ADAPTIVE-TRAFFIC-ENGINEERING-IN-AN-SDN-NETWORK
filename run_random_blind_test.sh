#!/usr/bin/env bash
# ==============================================================================
# Autonomous Zero-Shot Random Blind Topologies Evaluation Runner
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

echo "================================================================================"
echo " 🎲 RUNNING TRUE BLIND ZERO-SHOT TEST SUITE ON UNSEEN RANDOM TOPOLOGIES"
echo "================================================================================"

for NODES in 16 20 25 30; do
    echo ""
    echo ">>> Evaluating on Random Connected Mesh with N=$NODES nodes..."
    "$PYTHON_BIN" evaluate_random_blind_topology.py --nodes "$NODES"
done

echo ""
echo "================================================================================"
echo " ✅ ALL RANDOM BLIND TOPOLOGY TESTS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
