#!/bin/bash
# =============================================================================
# VM HARDENING SCRIPT - SSH Security + SSHD Watchdog
# =============================================================================
# Run this script on the DigitalOcean droplet to:
# 1. Harden SSH configuration
# 2. Set up UFW firewall rules
# 3. Install SSHD watchdog cron job
#
# Usage: sudo bash scripts/vm-hardening.sh
# =============================================================================

set -e

echo "=========================================="
echo "  VM HARDENING SCRIPT"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (sudo)"
    exit 1
fi

SSHD_CONFIG="/etc/ssh/sshd_config"
BACKUP_DIR="/root/backups/ssh"

# =============================================================================
# 1. SSH HARDENING
# =============================================================================
echo ""
echo "[1/3] SSH HARDENING"
echo "-------------------"

# Create backup
mkdir -p "$BACKUP_DIR"
cp "$SSHD_CONFIG" "$BACKUP_DIR/sshd_config.$(date +%Y%m%d_%H%M%S).bak"
echo "  - Backed up sshd_config"

# Apply hardening settings
echo "  - Applying SSH hardening..."

# ClientAliveInterval: Send keepalive every 60 seconds
if grep -q "^ClientAliveInterval" "$SSHD_CONFIG"; then
    sed -i 's/^ClientAliveInterval.*/ClientAliveInterval 60/' "$SSHD_CONFIG"
else
    echo "ClientAliveInterval 60" >> "$SSHD_CONFIG"
fi
echo "    ClientAliveInterval 60"

# ClientAliveCountMax: Disconnect after 3 missed keepalives (3 min timeout)
if grep -q "^ClientAliveCountMax" "$SSHD_CONFIG"; then
    sed -i 's/^ClientAliveCountMax.*/ClientAliveCountMax 3/' "$SSHD_CONFIG"
else
    echo "ClientAliveCountMax 3" >> "$SSHD_CONFIG"
fi
echo "    ClientAliveCountMax 3"

# Disable root password login (key-based only)
if grep -q "^PermitRootLogin" "$SSHD_CONFIG"; then
    sed -i 's/^PermitRootLogin.*/PermitRootLogin prohibit-password/' "$SSHD_CONFIG"
else
    echo "PermitRootLogin prohibit-password" >> "$SSHD_CONFIG"
fi
echo "    PermitRootLogin prohibit-password"

# Disable password authentication (keys only)
if grep -q "^PasswordAuthentication" "$SSHD_CONFIG"; then
    sed -i 's/^PasswordAuthentication.*/PasswordAuthentication no/' "$SSHD_CONFIG"
else
    echo "PasswordAuthentication no" >> "$SSHD_CONFIG"
fi
echo "    PasswordAuthentication no"

# Disable empty passwords
if grep -q "^PermitEmptyPasswords" "$SSHD_CONFIG"; then
    sed -i 's/^PermitEmptyPasswords.*/PermitEmptyPasswords no/' "$SSHD_CONFIG"
else
    echo "PermitEmptyPasswords no" >> "$SSHD_CONFIG"
fi
echo "    PermitEmptyPasswords no"

# Set max auth tries
if grep -q "^MaxAuthTries" "$SSHD_CONFIG"; then
    sed -i 's/^MaxAuthTries.*/MaxAuthTries 3/' "$SSHD_CONFIG"
else
    echo "MaxAuthTries 3" >> "$SSHD_CONFIG"
fi
echo "    MaxAuthTries 3"

# Validate config before restarting
echo "  - Validating sshd_config..."
if sshd -t; then
    echo "    Config valid!"
    systemctl restart sshd
    echo "  - SSHD restarted with new config"
else
    echo "ERROR: Invalid sshd_config! Restoring backup..."
    cp "$BACKUP_DIR/sshd_config."*.bak "$SSHD_CONFIG"
    exit 1
fi

# =============================================================================
# 2. UFW FIREWALL
# =============================================================================
echo ""
echo "[2/3] UFW FIREWALL"
echo "------------------"

# Install UFW if not present
if ! command -v ufw &> /dev/null; then
    echo "  - Installing UFW..."
    apt-get update && apt-get install -y ufw
fi

# Enable UFW and configure rules
echo "  - Configuring UFW rules..."

# Allow SSH (CRITICAL - do this first!)
ufw allow 22/tcp comment 'SSH'
echo "    Allow 22/tcp (SSH)"

# Allow Guardian API
ufw allow 8888/tcp comment 'Guardian API'
echo "    Allow 8888/tcp (Guardian API)"

# Allow Dashboard
ufw allow 8080/tcp comment 'Dashboard'
echo "    Allow 8080/tcp (Dashboard)"

# Allow HTTPS (for outbound API calls)
ufw allow out 443/tcp comment 'HTTPS outbound'
echo "    Allow 443/tcp outbound (HTTPS)"

# Enable UFW (will prompt if interactive)
echo "  - Enabling UFW..."
ufw --force enable
echo "  - UFW enabled"

# Show status
echo ""
ufw status verbose

# =============================================================================
# 3. SSHD WATCHDOG CRON
# =============================================================================
echo ""
echo "[3/3] SSHD WATCHDOG CRON"
echo "------------------------"

CRON_JOB="*/5 * * * * systemctl is-active sshd || systemctl restart sshd"
CRON_FILE="/etc/cron.d/sshd-watchdog"

echo "  - Installing SSHD watchdog cron job..."
echo "# SSHD Watchdog - Restart sshd if it dies" > "$CRON_FILE"
echo "# Installed by vm-hardening.sh on $(date)" >> "$CRON_FILE"
echo "$CRON_JOB" >> "$CRON_FILE"
chmod 644 "$CRON_FILE"
echo "    Cron job: $CRON_JOB"
echo "    Written to: $CRON_FILE"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=========================================="
echo "  HARDENING COMPLETE"
echo "=========================================="
echo ""
echo "SSH Hardening Applied:"
echo "  - ClientAliveInterval 60 (keepalive every 60s)"
echo "  - ClientAliveCountMax 3 (disconnect after 3min idle)"
echo "  - PermitRootLogin prohibit-password (keys only)"
echo "  - PasswordAuthentication no"
echo "  - MaxAuthTries 3"
echo ""
echo "Firewall (UFW) Rules:"
echo "  - Port 22 (SSH) - ALLOWED"
echo "  - Port 8888 (Guardian API) - ALLOWED"
echo "  - Port 8080 (Dashboard) - ALLOWED"
echo ""
echo "SSHD Watchdog:"
echo "  - Cron job checks every 5 minutes"
echo "  - Auto-restarts sshd if it dies"
echo ""
echo "IMPORTANT: Test SSH connection in a NEW terminal before closing this one!"
echo "=========================================="
