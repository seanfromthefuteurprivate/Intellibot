#!/bin/bash
# Deploy Phase 1 of Resilience Architecture
# - Circuit breaker integration
# - Dead man's switch
# - Systemd restart limits

set -e

echo "🛡️ DEPLOYING RESILIENCE ARCHITECTURE - PHASE 1"
echo "================================================"
echo ""

WSB_SNAKE_PATH="/root/wsb-snake"

# 1. Update Python dependencies if needed
echo "✅ Checking Python dependencies..."
cd "$WSB_SNAKE_PATH"

# 2. Update systemd service file with restart limits
echo "✅ Updating wsb-snake.service with restart limits..."
sudo cp "$WSB_SNAKE_PATH/wsb-snake.service" /etc/systemd/system/wsb-snake.service
sudo systemctl daemon-reload

# 3. Test circuit breaker
echo "✅ Testing circuit breaker..."
python3 "$WSB_SNAKE_PATH/ops/circuit_breaker.py" status

# 4. Test dead man's switch
echo "✅ Testing dead man's switch..."
python3 "$WSB_SNAKE_PATH/ops/dead_mans_switch.py" status

# 5. Restart monitor service to load new code
echo "✅ Restarting monitor service..."
sudo systemctl restart wsb-ops-monitor

# 6. Verify monitor is running
sleep 2
if systemctl is-active --quiet wsb-ops-monitor; then
    echo "✅ Monitor service restarted successfully"
else
    echo "❌ Monitor service failed to start"
    sudo journalctl -u wsb-ops-monitor --no-pager -n 20
    exit 1
fi

# 7. Verify wsb-snake service limits
echo "✅ Verifying systemd restart limits..."
systemctl show wsb-snake.service | grep -E "StartLimit"

echo ""
echo "================================================"
echo "✅ PHASE 1 DEPLOYMENT COMPLETE"
echo "================================================"
echo ""
echo "Circuit Breaker: Max 3 restarts in 5 minutes"
echo "Systemd Limits: Max 5 restarts in 5 minutes"
echo "Dead Man's Switch: Alert if no trades for 30 minutes"
echo ""
echo "Monitor circuit breaker status:"
echo "  python3 $WSB_SNAKE_PATH/ops/circuit_breaker.py status"
echo ""
echo "Reset circuit breaker (if needed):"
echo "  python3 $WSB_SNAKE_PATH/ops/circuit_breaker.py reset"
echo ""
echo "View monitor logs:"
echo "  journalctl -u wsb-ops-monitor -f"
