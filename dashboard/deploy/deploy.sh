#!/bin/bash
#
# WSB Snake Dashboard Deployment Script
# Deploys to 157.245.240.99
#

set -e

# Configuration
SERVER="157.245.240.99"
SSH_USER="root"
REMOTE_DIR="/root/Intellibot"
SERVICE_NAME="wsb-dashboard"

echo "========================================"
echo "WSB Snake Dashboard Deployment"
echo "========================================"
echo "Target: ${SSH_USER}@${SERVER}"
echo "Remote Directory: ${REMOTE_DIR}"
echo ""

# SSH command wrapper
SSH_CMD="ssh ${SSH_USER}@${SERVER}"

echo "[1/7] Pulling latest code..."
$SSH_CMD "cd ${REMOTE_DIR} && git pull origin main"

echo ""
echo "[2/7] Installing Python dependencies..."
$SSH_CMD "cd ${REMOTE_DIR} && pip3 install -r requirements.txt"

echo ""
echo "[3/7] Building React frontend..."
$SSH_CMD "cd ${REMOTE_DIR}/dashboard/frontend && npm install && npm run build"

echo ""
echo "[4/7] Copying systemd service file..."
$SSH_CMD "cp ${REMOTE_DIR}/dashboard/deploy/wsb-dashboard.service /etc/systemd/system/"

echo ""
echo "[5/7] Copying nginx configuration..."
$SSH_CMD "cp ${REMOTE_DIR}/dashboard/deploy/wsb-dashboard.nginx /etc/nginx/sites-available/${SERVICE_NAME}"
$SSH_CMD "ln -sf /etc/nginx/sites-available/${SERVICE_NAME} /etc/nginx/sites-enabled/${SERVICE_NAME}"
$SSH_CMD "nginx -t"

echo ""
echo "[6/7] Reloading systemd and restarting services..."
$SSH_CMD "systemctl daemon-reload"
$SSH_CMD "systemctl enable ${SERVICE_NAME}"
$SSH_CMD "systemctl restart ${SERVICE_NAME}"
$SSH_CMD "systemctl reload nginx"

echo ""
echo "[7/7] Checking service status..."
echo ""
echo "--- Dashboard API Status ---"
$SSH_CMD "systemctl status ${SERVICE_NAME} --no-pager"
echo ""
echo "--- Nginx Status ---"
$SSH_CMD "systemctl status nginx --no-pager"

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo "Dashboard should be accessible at: http://${SERVER}"
echo ""
