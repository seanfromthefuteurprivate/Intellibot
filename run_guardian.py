#!/usr/bin/env python3
"""
VM Guardian Agent - Entry point.

Starts the guardian agent and API server for VM monitoring and auto-healing.
Uses Claude Opus 4.6 via DigitalOcean Gradient AI for intelligent diagnostics.

Usage:
    python run_guardian.py

Environment Variables:
    DO_MODEL_ACCESS_KEY: DigitalOcean Model Access Key for Claude 4.6
    GUARDIAN_API_PORT: API server port (default: 8888)
    GUARDIAN_API_TOKEN: Optional auth token for API access
"""

import os
import sys
import signal
import logging
import time
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("guardian")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

# Critical environment variables that MUST be set
CRITICAL_ENV_VARS = [
    # None required for Guardian itself - it's a monitoring tool
    # But we warn about missing trading credentials
]

# Recommended environment variables (warn if missing)
RECOMMENDED_ENV_VARS = [
    ("GUARDIAN_API_TOKEN", "API authentication disabled - anyone can access"),
    ("DO_MODEL_ACCESS_KEY", "AI advisor disabled"),
]

# All config values to log on startup (with redaction for secrets)
CONFIG_VARS = [
    ("GUARDIAN_API_PORT", "8888", False),
    ("GUARDIAN_API_TOKEN", "", True),  # Redact
    ("DO_MODEL_ACCESS_KEY", "", True),  # Redact
    ("DO_MODEL_ID", "llama3.3-70b-instruct", False),
    ("ALPACA_API_KEY", "", True),  # Redact
    ("ALPACA_SECRET_KEY", "", True),  # Redact
    ("ALPACA_LIVE_TRADING", "false", False),
    ("TELEGRAM_BOT_TOKEN", "", True),  # Redact
    ("TELEGRAM_CHAT_ID", "", True),  # Redact
    ("POLYGON_API_KEY", "", True),  # Redact
]


def redact_secret(value: str) -> str:
    """Redact a secret value, showing only first/last 2 chars."""
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "****"
    return f"{value[:2]}...{value[-2:]}"


def validate_config() -> bool:
    """
    Validate configuration on startup.

    Returns True if all critical vars are set, False otherwise.
    Logs warnings for recommended vars.
    """
    logger.info("=" * 60)
    logger.info("       CONFIG VALIDATION")
    logger.info("=" * 60)

    all_critical_present = True

    # Check critical vars
    for var in CRITICAL_ENV_VARS:
        value = os.environ.get(var, "")
        if not value:
            logger.error(f"CRITICAL: Missing required env var: {var}")
            all_critical_present = False
        else:
            logger.info(f"  [OK] {var} is set")

    # Check recommended vars
    for var, warning in RECOMMENDED_ENV_VARS:
        value = os.environ.get(var, "")
        if not value:
            logger.warning(f"  [WARN] {var} not set - {warning}")
        else:
            logger.info(f"  [OK] {var} is set")

    return all_critical_present


def log_startup_config():
    """Log all configuration values on startup (redacting secrets)."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("       STARTUP CONFIGURATION")
    logger.info("=" * 60)

    for var, default, is_secret in CONFIG_VARS:
        value = os.environ.get(var, default)
        display_value = redact_secret(value) if is_secret else (value or "(default)")
        logger.info(f"  {var}: {display_value}")

    # Log system info
    logger.info("")
    logger.info("System Info:")
    try:
        # Get git commit
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=project_root
        )
        git_commit = result.stdout.strip() if result.returncode == 0 else "unknown"
        logger.info(f"  Git Commit: {git_commit}")
    except:
        logger.info(f"  Git Commit: unknown")

    try:
        # Get Python version
        import platform
        logger.info(f"  Python: {platform.python_version()}")
        logger.info(f"  Platform: {platform.platform()}")
    except:
        pass

    logger.info("=" * 60)


def main():
    """Main entry point for the guardian agent."""
    logger.info("=" * 60)
    logger.info("       VM GUARDIAN AGENT - Starting")
    logger.info("=" * 60)

    # Validate configuration FIRST
    if not validate_config():
        logger.error("=" * 60)
        logger.error("  STARTUP ABORTED: Missing critical environment variables")
        logger.error("  Please set all required variables and try again.")
        logger.error("=" * 60)
        sys.exit(1)

    # Log all config values (redacted)
    log_startup_config()

    # Check for API key
    api_key = os.getenv("DO_MODEL_ACCESS_KEY")
    if api_key:
        logger.info("AI Advisor: ENABLED (Claude Opus 4.6 via DigitalOcean)")
    else:
        logger.warning("AI Advisor: DISABLED (set DO_MODEL_ACCESS_KEY to enable)")

    # Import after path setup
    from wsb_snake.guardian.agent import VMGuardian, start_guardian
    from wsb_snake.guardian.api_server import start_api_server

    # Start guardian
    guardian = start_guardian()

    # Start API server
    api_port = int(os.getenv("GUARDIAN_API_PORT", "8888"))
    api_server = start_api_server(port=api_port)

    logger.info("")
    logger.info("Guardian is now running:")
    logger.info(f"  - Health monitoring: Every 60 seconds")
    logger.info(f"  - Auto-restart: ENABLED")
    logger.info(f"  - API server: http://0.0.0.0:{api_port}")
    logger.info("")
    logger.info("API Endpoints:")
    logger.info(f"  GET  /health   - Comprehensive health check (services, system, errors)")
    logger.info(f"  GET  /status   - Full status report")
    logger.info(f"  GET  /services - Service status")
    logger.info(f"  GET  /logs     - Service logs")
    logger.info(f"  POST /restart  - Restart a service")
    logger.info(f"  POST /deploy   - Pull git & restart")
    logger.info(f"  POST /diagnose - AI-powered diagnosis")
    logger.info(f"  POST /exec     - Execute whitelisted commands (safe only)")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Handle shutdown
    def shutdown_handler(signum, frame):
        logger.info("\nShutdown signal received...")
        guardian.stop()
        api_server.stop()
        logger.info("Guardian stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()
