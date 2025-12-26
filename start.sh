#!/bin/bash

# Start script for Anomaly Detection System
# This script handles conda environment setup and starts the main application

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="main.py"
LOG_FILE="$SCRIPT_DIR/output.log"
PID_FILE="$SCRIPT_DIR/anomaly_detection.pid"

# Change to script directory
cd "$SCRIPT_DIR"

# Check if the script is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Anomaly Detection System is already running with PID $OLD_PID"
        exit 1
    else
        # PID file exists but process is not running, remove stale PID file
        rm -f "$PID_FILE"
    fi
fi

# Setup conda environment
echo "Setting up conda environment..."
# Deactivate conda twice (in case we're in a nested conda environment)
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
# Activate the anomaly environment
conda activate anomaly

# Verify conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "anomaly" ]; then
    echo "ERROR: Failed to activate conda environment 'anomaly'"
    echo "Please ensure the environment exists: conda env list"
    exit 1
fi

echo "Conda environment 'anomaly' activated successfully"
echo "Starting Anomaly Detection System..."

# Delete existing log file to start fresh (before starting the process)
if [ -f "$LOG_FILE" ]; then
    rm -f "$LOG_FILE"
    echo "Deleted existing log file: $LOG_FILE"
fi

# Start the application with nohup and redirect output
nohup python "$SCRIPT_NAME" >> "$LOG_FILE" 2>&1 &
APP_PID=$!

# Save PID to file
echo $APP_PID > "$PID_FILE"

# Wait a moment to check if the process started successfully
sleep 2

if ps -p "$APP_PID" > /dev/null 2>&1; then
    echo "Anomaly Detection System started successfully!"
    echo "  PID: $APP_PID"
    echo "  Log file: $LOG_FILE"
    echo "  PID file: $PID_FILE"
    echo "  To view logs: tail -f $LOG_FILE"
else
    echo "ERROR: Failed to start Anomaly Detection System"
    rm -f "$PID_FILE"
    exit 1
fi

