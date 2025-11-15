#!/bin/bash

echo "Testing Meeting Recorder API..."
echo ""

# Health check
echo "1. Health Check:"
curl -s http://localhost:5000/health | python3 -m json.tool
echo ""
echo ""

# Start recording
echo "2. Starting Recording:"
curl -s -X POST http://localhost:5000/start | python3 -m json.tool
echo ""
echo ""

# Wait 10 seconds
echo "3. Recording for 10 seconds... (speak now!)"
sleep 10
echo ""

# Status check
echo "4. Checking Status:"
curl -s http://localhost:5000/status | python3 -m json.tool
echo ""
echo ""

# Stop recording
echo "5. Stopping Recording:"
curl -s -X POST http://localhost:5000/stop | python3 -m json.tool
echo ""
echo ""

echo "Test completed!"
