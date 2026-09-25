#!/usr/bin/env bash
# ==============================================================================
# Run Unit and Integration Test Suite (EC499)
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

exec "$PYTHON_BIN" test_suite.py "$@"
