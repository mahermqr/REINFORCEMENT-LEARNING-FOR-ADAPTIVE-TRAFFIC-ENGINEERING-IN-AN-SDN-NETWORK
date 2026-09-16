#!/usr/bin/env bash
# ==============================================================================
# Adaptive SDN Traffic Engineering & Multi-Agent Reinforcement Learning Launcher
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/home/maher/ec499_env"

echo "======================================================================"
echo " Starting Adaptive SDN Traffic Engineering Platform (EC499)"
echo " Ryu OpenFlow 1.3 Controller + Deep Q-Network (DQN) Routing Engine"
echo " Web Dashboard: http://localhost:8080"
echo "======================================================================"

# Activate environment (check local venv, configured system env, or PATH python)
if [ -d "$BASE_DIR/venv" ]; then
    echo "[1/3] Activating local virtual environment ($BASE_DIR/venv)..."
    source "$BASE_DIR/venv/bin/activate"
    PYTHON_EXEC="$BASE_DIR/venv/bin/python"
    RYU_EXEC="$BASE_DIR/venv/bin/ryu-manager"
elif [ -d "/home/maher/ec499_env" ]; then
    echo "[1/3] Activating virtual environment (/home/maher/ec499_env)..."
    source "/home/maher/ec499_env/bin/activate"
    PYTHON_EXEC="/home/maher/ec499_env/bin/python"
    RYU_EXEC="/home/maher/ec499_env/bin/ryu-manager"
else
    PYTHON_EXEC="python3"
    RYU_EXEC="ryu-manager"
fi

# Clean up stale mininet and ovs ports if sudo is available
if sudo -n true 2>/dev/null; then
    echo "[2/3] Cleaning stale Mininet and OpenFlow state..."
    sudo mn -c 2>/dev/null || true
else
    echo "[2/3] Skipping stale Mininet cleanup (passwordless sudo not active)."
fi

mkdir -p "$BASE_DIR/logs"

echo "[3/3] Starting Ryu OpenFlow 1.3 Controller in background..."
cd "$BASE_DIR/controller"
PYTHONPATH="$BASE_DIR:$BASE_DIR/agent:$BASE_DIR/controller" "$RYU_EXEC" main_controller.py --observe-links > "$BASE_DIR/logs/ryu_controller.log" 2>&1 &
RYU_PID=$!
echo "Ryu Controller started (PID: $RYU_PID). Waiting 3s for initialization..."
sleep 3

# Trap exit to kill Ryu upon script termination
cleanup() {
    echo -e "\nShutting down controller and network..."
    kill $RYU_PID 2>/dev/null || true
    if sudo -n true 2>/dev/null; then
        sudo mn -c 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT INT TERM

echo ""
echo "======================================================================"
echo " Web Dashboard is live at: http://localhost:8080"
echo "======================================================================"

# Check if Mininet can run with sudo
if sudo -n true 2>/dev/null; then
    echo "Starting Mininet Unified Topology & Traffic Simulator (Root Mode)..."
    cd "$BASE_DIR/topology"
    sudo "$PYTHON_EXEC" mininet_topo.py
else
    echo "Interactive sudo requires authentication for Mininet."
    echo "Starting Autonomous SDN Simulation Driver & Web Telemetry Stream..."
    cd "$BASE_DIR"
    "$PYTHON_EXEC" simulate_live_traffic.py --mode auto --interval 4.0
fi
