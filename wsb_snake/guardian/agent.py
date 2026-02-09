"""
VM Guardian Agent - Main orchestrator for system monitoring and auto-healing.

Runs as a background service that:
1. Monitors system health every 60 seconds
2. Auto-restarts failed services
3. Pulls git updates on command
4. Provides AI-powered diagnostics via Claude Opus 4.6
5. Sends Telegram alerts for critical issues
"""

import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

from wsb_snake.guardian.health_monitor import HealthMonitor, HealthReport, get_health_monitor
from wsb_snake.guardian.ai_advisor import AIAdvisor, get_ai_advisor

logger = logging.getLogger(__name__)

# Configuration
HEALTH_CHECK_INTERVAL = 60  # seconds
AUTO_RESTART_ENABLED = True
AUTO_RESTART_COOLDOWN = 300  # 5 minutes between auto-restarts
MAX_AUTO_RESTARTS = 3  # Max restarts before alerting


@dataclass
class ServiceRestartTracker:
    """Track service restart attempts."""
    service_name: str
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    consecutive_failures: int = 0


@dataclass
class GuardianState:
    """Current state of the guardian agent."""
    running: bool = False
    last_health_check: Optional[datetime] = None
    last_report: Optional[HealthReport] = None
    restart_trackers: Dict[str, ServiceRestartTracker] = field(default_factory=dict)
    alerts_sent: List[Dict[str, Any]] = field(default_factory=list)


class VMGuardian:
    """
    Main guardian agent for VM monitoring and management.

    Features:
    - Continuous health monitoring
    - Auto-restart of failed services
    - AI-powered diagnostics
    - Git sync management
    - Telegram alerting
    """

    def __init__(
        self,
        health_monitor: Optional[HealthMonitor] = None,
        ai_advisor: Optional[AIAdvisor] = None,
        auto_restart: bool = AUTO_RESTART_ENABLED
    ):
        """
        Initialize the guardian agent.

        Args:
            health_monitor: HealthMonitor instance (uses singleton if None)
            ai_advisor: AIAdvisor instance (uses singleton if None)
            auto_restart: Whether to automatically restart failed services
        """
        self.monitor = health_monitor or get_health_monitor()
        self.advisor = ai_advisor or get_ai_advisor()
        self.auto_restart = auto_restart

        self.state = GuardianState()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info("VMGuardian initialized")
        logger.info(f"Auto-restart: {'ENABLED' if auto_restart else 'DISABLED'}")
        logger.info(f"AI Advisor: {'AVAILABLE' if self.advisor.is_available() else 'NOT CONFIGURED'}")

    def start(self):
        """Start the guardian monitoring loop."""
        if self.state.running:
            logger.warning("Guardian already running")
            return

        self.state.running = True
        self._stop_event.clear()

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="VMGuardian-Monitor"
        )
        self._monitor_thread.start()

        logger.info("VMGuardian started - monitoring active")

    def stop(self):
        """Stop the guardian monitoring loop."""
        logger.info("Stopping VMGuardian...")
        self._stop_event.set()
        self.state.running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)

        logger.info("VMGuardian stopped")

    def _monitoring_loop(self):
        """Main monitoring loop - runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self._run_health_check()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            # Wait for next check interval
            self._stop_event.wait(HEALTH_CHECK_INTERVAL)

    def _run_health_check(self):
        """Run a single health check cycle."""
        logger.debug("Running health check...")

        report = self.monitor.generate_report()
        self.state.last_report = report
        self.state.last_health_check = datetime.now()

        # Log status
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}.get(
            report.overall_status, "❓"
        )
        logger.info(f"{status_emoji} Health: {report.overall_status.upper()}")

        if report.issues:
            for issue in report.issues:
                logger.warning(f"  - {issue}")

        # Handle critical issues
        if report.overall_status == "critical":
            self._handle_critical_issues(report)

    def _handle_critical_issues(self, report: HealthReport):
        """Handle critical issues detected in health report."""
        for service_name, service in report.services.items():
            if service.status == "failed" or not service.running:
                self._handle_failed_service(service_name, service)

    def _handle_failed_service(self, service_name: str, service):
        """Handle a failed service - potentially auto-restart."""
        logger.warning(f"Service {service_name} is DOWN (status: {service.status})")

        # Initialize tracker if needed
        if service_name not in self.state.restart_trackers:
            self.state.restart_trackers[service_name] = ServiceRestartTracker(
                service_name=service_name
            )

        tracker = self.state.restart_trackers[service_name]

        # Check if we should auto-restart
        if not self.auto_restart:
            logger.info(f"Auto-restart disabled - not restarting {service_name}")
            return

        # Check cooldown
        if tracker.last_restart:
            cooldown_remaining = (
                tracker.last_restart + timedelta(seconds=AUTO_RESTART_COOLDOWN)
            ) - datetime.now()

            if cooldown_remaining.total_seconds() > 0:
                logger.info(
                    f"Restart cooldown active for {service_name} "
                    f"({int(cooldown_remaining.total_seconds())}s remaining)"
                )
                return

        # Check max restarts
        if tracker.restart_count >= MAX_AUTO_RESTARTS:
            logger.error(
                f"Max auto-restarts ({MAX_AUTO_RESTARTS}) reached for {service_name} - "
                f"manual intervention required"
            )
            self._send_critical_alert(
                f"Service {service_name} failed {MAX_AUTO_RESTARTS} times - needs manual fix"
            )
            return

        # Attempt restart
        logger.info(f"Auto-restarting {service_name} (attempt {tracker.restart_count + 1})")

        success, message = self.monitor.restart_service(service_name)

        if success:
            tracker.restart_count += 1
            tracker.last_restart = datetime.now()
            tracker.consecutive_failures = 0
            logger.info(f"Service {service_name} restarted successfully")

            # Get AI diagnosis if available
            if self.advisor.is_available():
                logs = self.monitor.get_service_logs(service_name, lines=50)
                diagnosis = self.advisor.analyze_logs(logs, service_name)
                if diagnosis.success:
                    logger.info(f"AI Diagnosis: {diagnosis.content[:200]}")
        else:
            tracker.consecutive_failures += 1
            logger.error(f"Failed to restart {service_name}: {message}")

            # Get AI help for persistent failures
            if self.advisor.is_available() and tracker.consecutive_failures >= 2:
                logs = self.monitor.get_service_logs(service_name, lines=100)
                status_output, _, _ = self.monitor._run_command(
                    ["systemctl", "status", f"{service_name}.service"]
                )
                diagnosis = self.advisor.diagnose_service_failure(
                    service_name, status_output, logs
                )
                if diagnosis.success:
                    logger.info(f"AI Recovery Advice: {diagnosis.content[:500]}")

    def _send_critical_alert(self, message: str):
        """Send a critical alert via Telegram."""
        try:
            from wsb_snake.notifications.telegram_channels import send_signal

            alert_message = f"🚨 **VM GUARDIAN ALERT**\n\n{message}\n\n_Automatic recovery failed_"
            send_signal(alert_message)

            self.state.alerts_sent.append({
                "message": message,
                "sent_at": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    # ========== PUBLIC API ==========

    def get_status(self) -> Dict[str, Any]:
        """Get current guardian status."""
        report = self.state.last_report

        return {
            "running": self.state.running,
            "last_check": self.state.last_health_check.isoformat() if self.state.last_health_check else None,
            "overall_health": report.overall_status if report else "unknown",
            "issues": report.issues if report else [],
            "services": {
                name: {
                    "status": svc.status,
                    "running": svc.running,
                    "uptime_seconds": svc.uptime_seconds,
                    "memory_mb": svc.memory_mb
                }
                for name, svc in (report.services.items() if report else {})
            },
            "system": {
                "cpu_percent": report.system.cpu_percent if report else 0,
                "memory_percent": report.system.memory_percent if report else 0,
                "disk_percent": report.system.disk_percent if report else 0,
            } if report else {},
            "git": report.git_status if report else {},
            "ai_advisor_available": self.advisor.is_available(),
            "auto_restart_enabled": self.auto_restart
        }

    def force_health_check(self) -> HealthReport:
        """Force an immediate health check."""
        return self.monitor.generate_report()

    def restart_service(self, service_name: str) -> tuple:
        """
        Manually restart a service.

        Args:
            service_name: Name of the service to restart

        Returns:
            (success: bool, message: str)
        """
        return self.monitor.restart_service(service_name)

    def pull_and_restart(self) -> Dict[str, Any]:
        """
        Pull latest code and restart services.

        Returns:
            Dict with results of each operation
        """
        results = {
            "git_pull": {"success": False, "message": ""},
            "service_restarts": {}
        }

        # Pull git
        success, message = self.monitor.pull_git_updates()
        results["git_pull"] = {"success": success, "message": message}

        if not success:
            return results

        # Restart services
        for service_name in self.monitor.MONITORED_SERVICES:
            success, message = self.monitor.restart_service(service_name)
            results["service_restarts"][service_name] = {
                "success": success,
                "message": message
            }

        return results

    def diagnose_issue(self, description: str) -> str:
        """
        Use AI to diagnose an issue.

        Args:
            description: Description of the issue

        Returns:
            AI diagnosis or error message
        """
        if not self.advisor.is_available():
            return "AI advisor not configured - set DO_MODEL_ACCESS_KEY"

        # Gather context
        report = self.monitor.generate_report()

        system_info = {
            "services": {
                name: {"status": svc.status, "running": svc.running}
                for name, svc in report.services.items()
            },
            "system": {
                "cpu": report.system.cpu_percent,
                "memory": report.system.memory_percent,
                "disk": report.system.disk_percent
            },
            "git": report.git_status,
            "issues": report.issues
        }

        response = self.advisor.analyze_trading_issue(
            issue_description=description,
            trade_logs=None
        )

        if response.success:
            return response.content
        else:
            return f"AI diagnosis failed: {response.error}"


# Singleton instance
_guardian: Optional[VMGuardian] = None


def get_guardian() -> VMGuardian:
    """Get or create the guardian singleton."""
    global _guardian
    if _guardian is None:
        _guardian = VMGuardian()
    return _guardian


def start_guardian():
    """Start the guardian agent."""
    guardian = get_guardian()
    guardian.start()
    return guardian
