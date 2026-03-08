# IBKR IB Gateway Setup Status

**Date:** 2026-03-08
**Instance:** i-03f3a7c46ec809a43 (us-east-1)
**Path:** /home/ubuntu/wsb-snake

---

## Summary

| Component | Status |
|-----------|--------|
| IB Gateway Installation | ✅ Installed |
| IBC (IB Controller) | ✅ Installed |
| Java 21 | ✅ Installed |
| Xvfb | ✅ Installed |
| X11 Fonts | ✅ Installed |
| Gateway Startup | ❌ Fails (exit code 1112) |
| API Port 4002 | ❌ Not listening |

**Current Blocker:** IB Gateway cannot display login dialog on Xvfb virtual display.

---

## Installation Details

### IB Gateway
- **Version:** 10.37 (1037)
- **Location:** `/home/ubuntu/Jts/`
- **Jars:** `/home/ubuntu/Jts/jars/` (NOT in `ibgateway/1037` subdirectory)
- **Main jar:** `total-2024.jar`

### IBC (IB Controller)
- **Version:** 3.19.0
- **Location:** `/home/ubuntu/ibc/`
- **Jar:** `/home/ubuntu/ibc/IBC.jar`
- **Config:** `/opt/ibc/config.ini`

### Java
- **Version:** OpenJDK 21.0.10
- **Location:** `/usr/lib/jvm/java-21-openjdk-amd64`

### Credentials
- **Username:** DUP172459 (paper trading account)
- **Password:** Stored in `/home/ubuntu/wsb-snake/.env` as `IBKR_PASSWORD`
- **Mode:** paper

---

## What Works

1. **Java classpath correctly configured:**
   ```
   -cp /home/ubuntu/ibc/IBC.jar:/home/ubuntu/Jts/jars/*
   ```

2. **IBC loads and parses config:**
   ```
   IBC: version: 3.19.0
   IBC: using default settings provider: ini file is /opt/ibc/config.ini
   IBC: TWS Settings directory is: /var/snap/amazon-ssm-agent/12322
   ```

3. **Gateway classes load (no ClassNotFoundException):**
   ```
   IBC: Starting Gateway
   IBC: Getting config dialog
   IBC: Creating config dialog future
   ```

4. **Xvfb starts correctly:**
   ```
   pgrep Xvfb → 986057
   ```

---

## What Fails

### Error
```
IBC: IBC closing after TWS/Gateway failed to display login dialog
IBC: Exiting with exit code=1112
```

### Root Cause
IB Gateway's Swing UI cannot render on Xvfb virtual framebuffer. The login dialog window never appears, so IBC times out after 60 seconds.

### Attempted Fixes (All Failed)

| Fix | Result |
|-----|--------|
| `--add-opens` JVM flags for Java 21 module access | Fixed module errors, still no dialog |
| `env DISPLAY=:1` inline | Still no dialog |
| `Xvfb :1 -ac` (disable access control) | Still no dialog |
| Install X11 fonts (xfonts-base, fonts-dejavu, etc.) | Still no dialog |
| Install xterm | Installed but didn't help |

---

## Working Startup Command

This is the most complete command we tested (still fails at login dialog):

```bash
# Kill existing processes
pkill -f 'java.*IbcGateway' 2>/dev/null || true

# Start Xvfb with access control disabled
Xvfb :1 -ac -screen 0 1024x768x24 &
sleep 2

# Load environment
set -a && . /home/ubuntu/wsb-snake/.env && set +a

# Start IB Gateway
env DISPLAY=:1 java \
  -cp /home/ubuntu/ibc/IBC.jar:/home/ubuntu/Jts/jars/* \
  -Xmx512m \
  -Djava.awt.headless=false \
  --add-opens=java.desktop/javax.swing=ALL-UNNAMED \
  --add-opens=java.desktop/java.awt=ALL-UNNAMED \
  --add-opens=java.desktop/java.awt.event=ALL-UNNAMED \
  --add-opens=java.desktop/javax.swing.event=ALL-UNNAMED \
  --add-opens=java.desktop/javax.swing.plaf.basic=ALL-UNNAMED \
  --add-opens=java.desktop/javax.swing.table=ALL-UNNAMED \
  --add-opens=java.desktop/sun.awt=ALL-UNNAMED \
  --add-opens=java.desktop/sun.swing=ALL-UNNAMED \
  --add-opens=java.base/java.lang=ALL-UNNAMED \
  --add-opens=java.base/java.util=ALL-UNNAMED \
  --add-opens=java.base/java.lang.reflect=ALL-UNNAMED \
  ibcalpha.ibc.IbcGateway \
  /opt/ibc/config.ini \
  $IBKR_USERNAME \
  $IBKR_PASSWORD \
  paper \
  > /tmp/ibgateway.log 2>&1 &
```

---

## IBC Config File

Location: `/opt/ibc/config.ini`

```ini
LogToConsole=yes
FIXLoginId=
FIXPassword=
FIXPasswordEncrypted=no
TradingMode=paper
IbAutoClosedown=no
AcceptIncomingConnectionAction=accept
AcceptNonBrokerageAccountWarning=yes
ExistingSessionDetectedAction=primary
OverrideTwsApiPort=4002
DismissPasswordExpiryWarning=yes
DismissNSEComplianceNotice=yes
AllowBlindTrading=yes
ReadOnlyLogin=no
```

---

## Possible Fixes to Try

### 1. Use VNC Instead of Xvfb
Xvfb may lack proper GLX/rendering support for Swing. Try x11vnc or TigerVNC:
```bash
apt-get install -y tigervnc-standalone-server
vncserver :1 -geometry 1024x768 -depth 24
export DISPLAY=:1
```

### 2. Use Xvnc (VNC-based virtual display)
```bash
Xvnc :1 -geometry 1024x768 -depth 24 -SecurityTypes None &
```

### 3. Try Java 11 Instead of Java 21
IB Gateway may have better compatibility with Java 11:
```bash
apt-get install -y openjdk-11-jre
update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java
```

### 4. Check for Missing GTK/X11 Libraries
```bash
apt-get install -y libgtk-3-0 libcanberra-gtk3-module libatk1.0-0 libatk-bridge2.0-0
```

### 5. Use IBKR's Official Docker Image
IBKR provides a containerized version that may handle display issues:
- https://github.com/IBC-Alpha/IBC/wiki/Docker-Container

### 6. Try TWS Instead of Gateway
TWS (Trader Workstation) has better UI compatibility:
```bash
java -cp /home/ubuntu/ibc/IBC.jar:/home/ubuntu/Jts/jars/* \
  ibcalpha.ibc.IbcTws /opt/ibc/config.ini $IBKR_USERNAME $IBKR_PASSWORD paper
```

### 7. Run on a Desktop EC2 Instance
Use an EC2 instance with NICE DCV or Ubuntu Desktop AMI that has proper graphics support.

---

## Alternative Data Sources

Since IBKR is blocked, consider these alternatives for option bar data:

| Source | Stock Bars | Option Bars | Notes |
|--------|------------|-------------|-------|
| TastyTrade | ✅ Works | ❌ 0 bars | Quotes work, candles don't |
| Polygon | ❌ 0 bars | ❌ 0 bars | Needs Options addon ($) |
| Alpaca | ❌ 0 bars | ❌ 0 bars | May need subscription |
| IBKR | ❓ Blocked | ❓ Blocked | Gateway won't start |

---

## Log Files

- Gateway log: `/tmp/ibgateway.log`
- IBC settings dir: `/var/snap/amazon-ssm-agent/12322/`
- jts.ini: `/var/snap/amazon-ssm-agent/12322/jts.ini`

---

## Next Steps

1. **Try VNC-based display** (Xvnc or TigerVNC) instead of Xvfb
2. **Downgrade to Java 11** if VNC doesn't work
3. **Consider IBKR Docker image** as a cleaner solution
4. **Contact IBKR support** about headless Gateway deployment

---

## References

- IBC GitHub: https://github.com/IBC-Alpha/IBC
- IBC Wiki: https://github.com/IBC-Alpha/IBC/wiki
- IBKR API: https://www.interactivebrokers.com/campus/ibkr-api-page/ibkr-api-home/
- ib_async Python library: https://github.com/ib-api-reloaded/ib_async
