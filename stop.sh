#!/bin/bash

# Stop script for Anomaly Detection System
# This script gracefully stops the running application

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="main.py"
PID_FILE="$SCRIPT_DIR/anomaly_detection.pid"

# Change to script directory
cd "$SCRIPT_DIR"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "PID file not found. Checking for running processes..."
    # Try to find the process by script name
    PIDS=$(pgrep -f "$SCRIPT_NAME")
    if [ -z "$PIDS" ]; then
        echo "Anomaly Detection System is not running"
        exit 0
    else
        echo "Found running processes: $PIDS"
        read -p "Do you want to kill these processes? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$PIDS" | xargs kill -TERM 2>/dev/null
            sleep 3
            # Force kill if still running
            REMAINING=$(pgrep -f "$SCRIPT_NAME")
            if [ -n "$REMAINING" ]; then
                echo "Force killing remaining processes..."
                echo "$REMAINING" | xargs kill -9 2>/dev/null
            fi
            echo "Anomaly Detection System stopped"
        else
            echo "Aborted"
            exit 1
        fi
        exit 0
    fi
fi

# Read PID from file
PID=$(cat "$PID_FILE")

# Check if process is still running
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Process with PID $PID is not running"
    rm -f "$PID_FILE"
    exit 0
fi

echo "Stopping Anomaly Detection System (PID: $PID)..."

# Send SIGTERM for graceful shutdown
kill -TERM "$PID" 2>/dev/null

# Wait for process to stop (max 30 seconds)
TIMEOUT=30
ELAPSED=0
while ps -p "$PID" > /dev/null 2>&1 && [ $ELAPSED -lt $TIMEOUT ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $((ELAPSED % 5)) -eq 0 ]; then
        echo "Waiting for process to stop... (${ELAPSED}s/${TIMEOUT}s)"
    fi
done

# Check if process is still running
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Process did not stop gracefully. Force killing..."
    kill -9 "$PID" 2>/dev/null
    sleep 1
fi

# Verify process is stopped
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Anomaly Detection System stopped successfully"
    rm -f "$PID_FILE"
    
    # Also check for any remaining processes with the same script name
    REMAINING=$(pgrep -f "$SCRIPT_NAME")
    if [ -n "$REMAINING" ]; then
        echo "Warning: Found additional processes: $REMAINING"
        echo "You may want to check and stop them manually"
    fi
else
    echo "ERROR: Failed to stop process $PID"
    exit 1
fi

