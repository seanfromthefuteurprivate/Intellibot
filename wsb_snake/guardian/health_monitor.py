"""
Health Monitor - Checks system and service health.

Monitors:
- Service status (wsb-snake, wsb-dashboard)
- System resources (CPU, memory, disk)
- Process health
- Git sync status
"""

import os
import subprocess
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ServiceHealth:
    """Health status of a systemd service."""
    name: str
    active: bool
    running: bool
    status: str  # "running", "stopped", "failed", "unknown"
    uptime_seconds: Optional[int] = None
    pid: Optional[int] = None
    memory_mb: Optional[float] = None
    last_check: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    load_average: tuple  # 1, 5, 15 min
    uptime_hours: float
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class HealthReport:
    """Complete health report."""
    system: SystemHealth
    services: Dict[str, ServiceHealth]
    git_status: Dict[str, Any]
    overall_status: str  # "healthy", "warning", "critical"
    issues: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class HealthMonitor:
    """
    Monitors system and service health.

    Provides real-time health checks and historical tracking.
    """

    # Services to monitor
    MONITORED_SERVICES = ["wsb-snake", "wsb-dashboard"]

    # Thresholds
    CPU_WARNING = 80
    CPU_CRITICAL = 95
    MEMORY_WARNING = 80
    MEMORY_CRITICAL = 95
    DISK_WARNING = 80
    DISK_CRITICAL = 95

    # Default repo path - configurable via WSB_SNAKE_PATH env var
    DEFAULT_REPO_PATH = os.getenv("WSB_SNAKE_PATH", "/home/ubuntu/wsb-snake")

    def __init__(self, repo_path: str = None):
        """
        Initialize health monitor.

        Args:
            repo_path: Path to the git repository (defaults to WSB_SNAKE_PATH env var or /home/ubuntu/wsb-snake)
        """
        self.repo_path = repo_path or self.DEFAULT_REPO_PATH
        self.last_report: Optional[HealthReport] = None

    def _run_command(self, cmd: List[str], timeout: int = 10) -> tuple:
        """Run a shell command and return (stdout, stderr, returncode)."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", -1
        except Exception as e:
            return "", str(e), -1

    def check_service(self, service_name: str) -> ServiceHealth:
        """
        Check health of a systemd service.

        Args:
            service_name: Name of the service (e.g., "wsb-snake")

        Returns:
            ServiceHealth with current status
        """
        full_name = f"{service_name}.service"

        # Check if active
        stdout, stderr, rc = self._run_command(
            ["systemctl", "is-active", full_name]
        )
        is_active = stdout.strip() == "active"

        # Get detailed status
        stdout, stderr, rc = self._run_command(
            ["systemctl", "show", full_name,
             "--property=ActiveState,SubState,MainPID,MemoryCurrent,ExecMainStartTimestamp"]
        )

        props = {}
        for line in stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                props[key] = value

        active_state = props.get("ActiveState", "unknown")
        sub_state = props.get("SubState", "unknown")
        pid = int(props.get("MainPID", 0)) or None
        memory_bytes = int(props.get("MemoryCurrent", 0) or 0)
        memory_mb = memory_bytes / (1024 * 1024) if memory_bytes else None

        # Determine status
        if active_state == "active" and sub_state == "running":
            status = "running"
            running = True
        elif active_state == "failed":
            status = "failed"
            running = False
        elif active_state == "inactive":
            status = "stopped"
            running = False
        else:
            status = "unknown"
            running = False

        # Calculate uptime
        uptime_seconds = None
        start_time_str = props.get("ExecMainStartTimestamp", "")
        if start_time_str and start_time_str != "n/a":
            try:
                # Parse systemd timestamp format
                # Example: "Sat 2026-02-08 10:30:00 UTC"
                from dateutil import parser
                start_time = parser.parse(start_time_str.split()[-2] + " " + start_time_str.split()[-1])
                uptime_seconds = int((datetime.now() - start_time).total_seconds())
            except:
                pass

        return ServiceHealth(
            name=service_name,
            active=is_active,
            running=running,
            status=status,
            uptime_seconds=uptime_seconds,
            pid=pid,
            memory_mb=memory_mb,
            error_message=stderr if not running else None
        )

    def check_system(self) -> SystemHealth:
        """
        Check system resource usage.

        Returns:
            SystemHealth with current metrics
        """
        # CPU usage
        cpu_percent = 0.0
        stdout, _, _ = self._run_command(
            ["grep", "-o", "^[^ ]*", "/proc/loadavg"]
        )
        try:
            load_1 = float(stdout.strip())
            # Get CPU count for percentage calculation
            stdout2, _, _ = self._run_command(["nproc"])
            cpu_count = int(stdout2.strip()) if stdout2.strip() else 1
            cpu_percent = (load_1 / cpu_count) * 100
        except:
            pass

        # Load average
        load_average = (0.0, 0.0, 0.0)
        stdout, _, _ = self._run_command(["cat", "/proc/loadavg"])
        try:
            parts = stdout.strip().split()
            load_average = (float(parts[0]), float(parts[1]), float(parts[2]))
        except:
            pass

        # Memory usage
        memory_percent = 0.0
        memory_used_mb = 0.0
        memory_total_mb = 0.0
        stdout, _, _ = self._run_command(["free", "-m"])
        try:
            lines = stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                memory_total_mb = float(parts[1])
                memory_used_mb = float(parts[2])
                memory_percent = (memory_used_mb / memory_total_mb) * 100 if memory_total_mb > 0 else 0
        except:
            pass

        # Disk usage
        disk_percent = 0.0
        disk_used_gb = 0.0
        disk_total_gb = 0.0
        stdout, _, _ = self._run_command(["df", "-BG", "/"])
        try:
            lines = stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                disk_total_gb = float(parts[1].rstrip("G"))
                disk_used_gb = float(parts[2].rstrip("G"))
                disk_percent = float(parts[4].rstrip("%"))
        except:
            pass

        # Uptime
        uptime_hours = 0.0
        stdout, _, _ = self._run_command(["cat", "/proc/uptime"])
        try:
            uptime_seconds = float(stdout.split()[0])
            uptime_hours = uptime_seconds / 3600
        except:
            pass

        return SystemHealth(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            load_average=load_average,
            uptime_hours=uptime_hours
        )

    def check_git_status(self) -> Dict[str, Any]:
        """
        Check git repository status.

        Returns:
            Dict with branch, commit, and sync status
        """
        result = {
            "branch": "unknown",
            "commit": "unknown",
            "behind_remote": 0,
            "uncommitted_changes": False,
            "last_pull": None
        }

        if not os.path.exists(self.repo_path):
            return result

        # Current branch
        stdout, _, rc = self._run_command(
            ["git", "-C", self.repo_path, "branch", "--show-current"]
        )
        if rc == 0:
            result["branch"] = stdout.strip()

        # Current commit
        stdout, _, rc = self._run_command(
            ["git", "-C", self.repo_path, "rev-parse", "--short", "HEAD"]
        )
        if rc == 0:
            result["commit"] = stdout.strip()

        # Check for uncommitted changes
        stdout, _, rc = self._run_command(
            ["git", "-C", self.repo_path, "status", "--porcelain"]
        )
        if rc == 0:
            result["uncommitted_changes"] = len(stdout.strip()) > 0

        # Check if behind remote (requires fetch first)
        self._run_command(["git", "-C", self.repo_path, "fetch", "--quiet"])
        stdout, _, rc = self._run_command(
            ["git", "-C", self.repo_path, "rev-list", "--count", "HEAD..origin/main"]
        )
        if rc == 0:
            try:
                result["behind_remote"] = int(stdout.strip())
            except:
                pass

        return result

    def get_service_logs(self, service_name: str, lines: int = 100) -> str:
        """
        Get recent logs for a service.

        Args:
            service_name: Name of the service
            lines: Number of log lines to retrieve

        Returns:
            Log content as string
        """
        stdout, _, _ = self._run_command(
            ["journalctl", "-u", f"{service_name}.service", "-n", str(lines), "--no-pager"]
        )
        return stdout

    def generate_report(self) -> HealthReport:
        """
        Generate a complete health report.

        Returns:
            HealthReport with all checks
        """
        # Check all services
        services = {}
        for service_name in self.MONITORED_SERVICES:
            services[service_name] = self.check_service(service_name)

        # Check system
        system = self.check_system()

        # Check git
        git_status = self.check_git_status()

        # Identify issues
        issues = []
        overall_status = "healthy"

        # Check service issues
        for name, svc in services.items():
            if svc.status == "failed":
                issues.append(f"Service {name} has FAILED")
                overall_status = "critical"
            elif not svc.running:
                issues.append(f"Service {name} is not running")
                if overall_status != "critical":
                    overall_status = "warning"

        # Check system issues
        if system.cpu_percent >= self.CPU_CRITICAL:
            issues.append(f"CPU usage CRITICAL: {system.cpu_percent:.1f}%")
            overall_status = "critical"
        elif system.cpu_percent >= self.CPU_WARNING:
            issues.append(f"CPU usage high: {system.cpu_percent:.1f}%")
            if overall_status == "healthy":
                overall_status = "warning"

        if system.memory_percent >= self.MEMORY_CRITICAL:
            issues.append(f"Memory usage CRITICAL: {system.memory_percent:.1f}%")
            overall_status = "critical"
        elif system.memory_percent >= self.MEMORY_WARNING:
            issues.append(f"Memory usage high: {system.memory_percent:.1f}%")
            if overall_status == "healthy":
                overall_status = "warning"

        if system.disk_percent >= self.DISK_CRITICAL:
            issues.append(f"Disk usage CRITICAL: {system.disk_percent:.1f}%")
            overall_status = "critical"
        elif system.disk_percent >= self.DISK_WARNING:
            issues.append(f"Disk usage high: {system.disk_percent:.1f}%")
            if overall_status == "healthy":
                overall_status = "warning"

        # Check git sync
        if git_status["behind_remote"] > 0:
            issues.append(f"Git is {git_status['behind_remote']} commits behind origin/main")
            if overall_status == "healthy":
                overall_status = "warning"

        report = HealthReport(
            system=system,
            services=services,
            git_status=git_status,
            overall_status=overall_status,
            issues=issues
        )

        self.last_report = report
        return report

    def restart_service(self, service_name: str) -> tuple:
        """
        Restart a systemd service.

        Args:
            service_name: Name of the service

        Returns:
            (success: bool, message: str)
        """
        logger.info(f"Restarting service: {service_name}")

        stdout, stderr, rc = self._run_command(
            ["sudo", "systemctl", "restart", f"{service_name}.service"],
            timeout=30
        )

        if rc == 0:
            logger.info(f"Service {service_name} restarted successfully")
            return True, f"Service {service_name} restarted"
        else:
            logger.error(f"Failed to restart {service_name}: {stderr}")
            return False, f"Failed to restart: {stderr}"

    def pull_git_updates(self) -> tuple:
        """
        Pull latest code from git.

        Returns:
            (success: bool, message: str)
        """
        logger.info("Pulling git updates...")

        stdout, stderr, rc = self._run_command(
            ["git", "-C", self.repo_path, "pull", "origin", "main"],
            timeout=60
        )

        if rc == 0:
            logger.info(f"Git pull successful: {stdout[:100]}")
            return True, stdout.strip()
        else:
            logger.error(f"Git pull failed: {stderr}")
            return False, stderr.strip()


# Singleton instance
_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get or create the health monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor()
    return _monitor
