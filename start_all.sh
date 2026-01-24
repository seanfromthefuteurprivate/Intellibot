#!/bin/bash

echo "Starting WSB Snake Trading Engine..."
python -m wsb_snake.main &
PYTHON_PID=$!
echo "Python engine started with PID: $PYTHON_PID"

echo "Starting web server..."
npm run dev
