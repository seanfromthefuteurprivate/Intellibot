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


def main():
    """Main entry point for the guardian agent."""
    logger.info("=" * 60)
    logger.info("       VM GUARDIAN AGENT - Starting")
    logger.info("=" * 60)

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
    logger.info(f"  GET  /health   - Quick health check")
    logger.info(f"  GET  /status   - Full status report")
    logger.info(f"  GET  /services - Service status")
    logger.info(f"  GET  /logs     - Service logs")
    logger.info(f"  POST /restart  - Restart a service")
    logger.info(f"  POST /deploy   - Pull git & restart")
    logger.info(f"  POST /diagnose - AI-powered diagnosis")
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
