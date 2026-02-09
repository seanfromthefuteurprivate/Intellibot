"""
Guardian API Server - HTTP interface for remote control when SSH fails.

Exposes endpoints for:
- Health status
- Service restart
- Git sync
- AI diagnostics
- Command execution
"""

import os
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

from wsb_snake.guardian.agent import VMGuardian, get_guardian
from wsb_snake.guardian.health_monitor import get_health_monitor
from wsb_snake.guardian.ai_advisor import get_ai_advisor

logger = logging.getLogger(__name__)

# Configuration
API_PORT = int(os.getenv("GUARDIAN_API_PORT", "8888"))
API_TOKEN = os.getenv("GUARDIAN_API_TOKEN", "")  # Optional auth token


class GuardianAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Guardian API."""

    guardian: VMGuardian = None

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"API: {args[0]}")

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def _check_auth(self) -> bool:
        """Check authorization if token is configured."""
        if not API_TOKEN:
            return True  # No auth required

        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return token == API_TOKEN

        # Also check query param
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if "token" in params:
            return params["token"][0] == API_TOKEN

        return False

    def _get_body(self) -> Dict[str, Any]:
        """Parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length).decode()
        return json.loads(body) if body else {}

    def do_GET(self):
        """Handle GET requests."""
        if not self._check_auth():
            self._send_json({"error": "Unauthorized"}, 401)
            return

        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/" or path == "/health":
                self._handle_health()
            elif path == "/status":
                self._handle_status()
            elif path == "/services":
                self._handle_services()
            elif path == "/git":
                self._handle_git_status()
            elif path == "/logs":
                self._handle_logs(parsed)
            else:
                self._send_json({"error": "Not found"}, 404)
        except Exception as e:
            logger.error(f"API error: {e}")
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        """Handle POST requests."""
        if not self._check_auth():
            self._send_json({"error": "Unauthorized"}, 401)
            return

        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._get_body()

            if path == "/restart":
                self._handle_restart(body)
            elif path == "/pull":
                self._handle_pull()
            elif path == "/deploy":
                self._handle_deploy()
            elif path == "/diagnose":
                self._handle_diagnose(body)
            elif path == "/command":
                self._handle_command(body)
            else:
                self._send_json({"error": "Not found"}, 404)
        except Exception as e:
            logger.error(f"API error: {e}")
            self._send_json({"error": str(e)}, 500)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    # ========== ENDPOINT HANDLERS ==========

    def _handle_health(self):
        """GET /health - Quick health check."""
        guardian = get_guardian()
        status = guardian.get_status()

        self._send_json({
            "status": "ok",
            "guardian_running": status["running"],
            "overall_health": status["overall_health"],
            "timestamp": datetime.now().isoformat()
        })

    def _handle_status(self):
        """GET /status - Full status report."""
        guardian = get_guardian()
        status = guardian.get_status()
        self._send_json(status)

    def _handle_services(self):
        """GET /services - Service status."""
        monitor = get_health_monitor()
        services = {}

        for service_name in monitor.MONITORED_SERVICES:
            svc = monitor.check_service(service_name)
            services[service_name] = {
                "status": svc.status,
                "running": svc.running,
                "active": svc.active,
                "pid": svc.pid,
                "memory_mb": svc.memory_mb,
                "uptime_seconds": svc.uptime_seconds
            }

        self._send_json({"services": services})

    def _handle_git_status(self):
        """GET /git - Git repository status."""
        monitor = get_health_monitor()
        git_status = monitor.check_git_status()
        self._send_json({"git": git_status})

    def _handle_logs(self, parsed):
        """GET /logs?service=NAME&lines=N - Get service logs."""
        params = parse_qs(parsed.query)
        service = params.get("service", ["wsb-snake"])[0]
        lines = int(params.get("lines", ["50"])[0])

        monitor = get_health_monitor()
        logs = monitor.get_service_logs(service, lines)

        self._send_json({
            "service": service,
            "lines": lines,
            "logs": logs
        })

    def _handle_restart(self, body: Dict):
        """POST /restart - Restart a service."""
        service = body.get("service", "wsb-snake")

        guardian = get_guardian()
        success, message = guardian.restart_service(service)

        self._send_json({
            "success": success,
            "message": message,
            "service": service
        })

    def _handle_pull(self):
        """POST /pull - Pull git updates."""
        monitor = get_health_monitor()
        success, message = monitor.pull_git_updates()

        self._send_json({
            "success": success,
            "message": message
        })

    def _handle_deploy(self):
        """POST /deploy - Pull and restart all services."""
        guardian = get_guardian()
        results = guardian.pull_and_restart()
        self._send_json(results)

    def _handle_diagnose(self, body: Dict):
        """POST /diagnose - AI diagnosis."""
        description = body.get("issue", body.get("description", ""))

        if not description:
            self._send_json({"error": "Missing 'issue' in request body"}, 400)
            return

        guardian = get_guardian()
        diagnosis = guardian.diagnose_issue(description)

        self._send_json({
            "diagnosis": diagnosis,
            "ai_powered": guardian.advisor.is_available()
        })

    def _handle_command(self, body: Dict):
        """POST /command - Execute shell command (dangerous!)."""
        command = body.get("command", "")

        if not command:
            self._send_json({"error": "Missing 'command' in request body"}, 400)
            return

        # Safety: Only allow certain commands
        allowed_prefixes = [
            "systemctl",
            "git -C /root/wsb-snake",
            "journalctl",
            "cat /proc",
            "free",
            "df",
            "uptime",
            "ps aux",
            "pm2"
        ]

        is_allowed = any(command.startswith(prefix) for prefix in allowed_prefixes)
        if not is_allowed:
            self._send_json({
                "error": "Command not in allowlist",
                "allowed_prefixes": allowed_prefixes
            }, 403)
            return

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd="/root/wsb-snake"
            )

            self._send_json({
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            })
        except subprocess.TimeoutExpired:
            self._send_json({"error": "Command timed out"}, 504)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)


class GuardianAPIServer:
    """
    HTTP API server for the Guardian agent.

    Runs in a background thread and provides HTTP access
    to guardian functionality when SSH is unavailable.
    """

    def __init__(self, port: int = API_PORT, guardian: Optional[VMGuardian] = None):
        """
        Initialize the API server.

        Args:
            port: Port to listen on (default 8888)
            guardian: VMGuardian instance to use
        """
        self.port = port
        self.guardian = guardian or get_guardian()
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        # Set guardian on handler class
        GuardianAPIHandler.guardian = self.guardian

    def start(self):
        """Start the API server in a background thread."""
        if self.server:
            logger.warning("API server already running")
            return

        self.server = HTTPServer(("0.0.0.0", self.port), GuardianAPIHandler)
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
            name="GuardianAPI"
        )
        self._thread.start()

        logger.info(f"Guardian API server started on port {self.port}")
        logger.info(f"Auth: {'REQUIRED' if API_TOKEN else 'DISABLED'}")

    def stop(self):
        """Stop the API server."""
        if self.server:
            self.server.shutdown()
            self.server = None
            logger.info("Guardian API server stopped")


# Singleton instance
_api_server: Optional[GuardianAPIServer] = None


def get_api_server() -> GuardianAPIServer:
    """Get or create the API server singleton."""
    global _api_server
    if _api_server is None:
        _api_server = GuardianAPIServer()
    return _api_server


def start_api_server(port: int = API_PORT) -> GuardianAPIServer:
    """Start the API server."""
    server = get_api_server()
    server.port = port
    server.start()
    return server
