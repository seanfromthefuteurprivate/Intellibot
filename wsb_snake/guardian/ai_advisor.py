"""
AI Advisor - Claude Opus 4.6 integration via DigitalOcean Gradient AI.

Provides intelligent system diagnostics, log analysis, and recommendations.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# DigitalOcean Gradient AI Configuration
DO_INFERENCE_URL = "https://inference.do-ai.run/v1"
# Model priority: Try Claude first, fall back to Llama
DO_MODEL_ID = os.getenv("DO_MODEL_ID", "llama3.3-70b-instruct")
# Available models (in order of preference):
# - anthropic-claude-opus-4.6 (requires higher tier)
# - anthropic-claude-3.7-sonnet (requires higher tier)
# - llama3.3-70b-instruct (free tier - default)
# - deepseek-r1-distill-llama-70b (free tier)


@dataclass
class AIResponse:
    """Response from Claude Opus 4.6."""
    content: str
    model: str
    tokens_used: int
    success: bool
    error: Optional[str] = None


class AIAdvisor:
    """
    AI-powered advisor using Claude Opus 4.6 via DigitalOcean.

    Provides intelligent analysis of:
    - System health and logs
    - Service failures and recovery recommendations
    - Code issues and fixes
    - Trading system diagnostics
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI Advisor.

        Args:
            api_key: DigitalOcean Model Access Key. Falls back to env var.
        """
        self.api_key = api_key or os.getenv("DO_MODEL_ACCESS_KEY")
        if not self.api_key:
            logger.warning("No DO_MODEL_ACCESS_KEY found - AI advisor disabled")

        self.base_url = DO_INFERENCE_URL
        self.model_id = DO_MODEL_ID
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        # System context for all requests
        wsb_path = os.getenv("WSB_SNAKE_PATH", "/home/ubuntu/wsb-snake")
        self.system_context = f"""You are the VM Guardian Agent for WSB Snake, a 0DTE options trading system.

Your role:
1. Monitor system health and diagnose issues
2. Analyze logs and identify root causes
3. Recommend fixes and recovery actions
4. Help with service restarts and deployments

The system runs on an AWS EC2 instance with:
- wsb-snake.service: Main trading engine
- wsb-dashboard.service: Dashboard API on port 8080
- Python 3.x with venv at {wsb_path}/venv
- Git repo at {wsb_path}

Be concise. Provide actionable commands when possible."""

    def is_available(self) -> bool:
        """Check if AI advisor is configured and available."""
        return bool(self.api_key)

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> AIResponse:
        """
        Call Claude Opus 4.6 API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Creativity (0-1, lower = more focused)

        Returns:
            AIResponse with result or error
        """
        if not self.api_key:
            return AIResponse(
                content="",
                model=self.model_id,
                tokens_used=0,
                success=False,
                error="API key not configured"
            )

        try:
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": self.system_context},
                    *messages
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

            return AIResponse(
                content=content,
                model=self.model_id,
                tokens_used=tokens,
                success=True
            )

        except requests.exceptions.Timeout:
            return AIResponse(
                content="",
                model=self.model_id,
                tokens_used=0,
                success=False,
                error="API timeout"
            )
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_msg = e.response.json().get("error", {}).get("message", str(e))
            except:
                pass
            return AIResponse(
                content="",
                model=self.model_id,
                tokens_used=0,
                success=False,
                error=f"API error: {error_msg}"
            )
        except Exception as e:
            return AIResponse(
                content="",
                model=self.model_id,
                tokens_used=0,
                success=False,
                error=f"Unexpected error: {str(e)}"
            )

    def analyze_logs(self, logs: str, service_name: str = "wsb-snake") -> AIResponse:
        """
        Analyze service logs and identify issues.

        Args:
            logs: Log content to analyze
            service_name: Name of the service

        Returns:
            AIResponse with analysis and recommendations
        """
        messages = [{
            "role": "user",
            "content": f"""Analyze these logs from {service_name} service and identify:
1. Any errors or warnings
2. Root cause of issues
3. Recommended fix (with commands if applicable)

LOGS:
```
{logs[-8000:]}  # Last 8000 chars to fit context
```

Respond with:
- ISSUE: Brief description
- CAUSE: Root cause
- FIX: Commands or steps to resolve"""
        }]

        return self._call_api(messages, max_tokens=1500)

    def diagnose_service_failure(
        self,
        service_name: str,
        status_output: str,
        recent_logs: str
    ) -> AIResponse:
        """
        Diagnose why a service failed.

        Args:
            service_name: Name of the failed service
            status_output: Output of systemctl status
            recent_logs: Recent journalctl logs

        Returns:
            AIResponse with diagnosis and recovery steps
        """
        messages = [{
            "role": "user",
            "content": f"""Service {service_name} has failed. Diagnose and provide recovery steps.

SYSTEMCTL STATUS:
```
{status_output}
```

RECENT LOGS:
```
{recent_logs[-6000:]}
```

Provide:
1. DIAGNOSIS: What went wrong
2. RECOVERY: Exact commands to fix and restart
3. PREVENTION: How to prevent this in future"""
        }]

        return self._call_api(messages, max_tokens=2000)

    def analyze_trading_issue(
        self,
        issue_description: str,
        relevant_code: Optional[str] = None,
        trade_logs: Optional[str] = None
    ) -> AIResponse:
        """
        Analyze a trading system issue.

        Args:
            issue_description: Description of the problem
            relevant_code: Optional code snippet
            trade_logs: Optional trade execution logs

        Returns:
            AIResponse with analysis
        """
        context = f"ISSUE: {issue_description}\n\n"

        if relevant_code:
            context += f"CODE:\n```python\n{relevant_code[:4000]}\n```\n\n"

        if trade_logs:
            context += f"TRADE LOGS:\n```\n{trade_logs[-4000:]}\n```\n\n"

        messages = [{
            "role": "user",
            "content": f"""{context}
Analyze this trading system issue. Provide:
1. ROOT CAUSE
2. IMPACT on trading
3. FIX with code changes if needed
4. VALIDATION steps to confirm fix"""
        }]

        return self._call_api(messages, max_tokens=2000)

    def get_health_summary(self, system_info: Dict[str, Any]) -> AIResponse:
        """
        Generate a health summary from system information.

        Args:
            system_info: Dict with CPU, memory, disk, service status, etc.

        Returns:
            AIResponse with health assessment
        """
        messages = [{
            "role": "user",
            "content": f"""Generate a health summary for this trading system:

SYSTEM INFO:
```json
{json.dumps(system_info, indent=2)}
```

Provide:
1. OVERALL HEALTH: Good/Warning/Critical
2. ISSUES: List any problems found
3. RECOMMENDATIONS: Actions to improve health"""
        }]

        return self._call_api(messages, max_tokens=1000, temperature=0.2)

    def suggest_command(self, task_description: str) -> AIResponse:
        """
        Suggest shell commands for a given task.

        Args:
            task_description: What needs to be done

        Returns:
            AIResponse with suggested commands
        """
        wsb_path = os.getenv("WSB_SNAKE_PATH", "/home/ubuntu/wsb-snake")
        messages = [{
            "role": "user",
            "content": f"""Suggest the exact shell commands to: {task_description}

Context:
- Working directory: {wsb_path}
- Python venv: {wsb_path}/venv
- Services: wsb-snake.service, wsb-dashboard.service

Provide only the commands, one per line, ready to execute."""
        }]

        return self._call_api(messages, max_tokens=500, temperature=0.1)


# Singleton instance
_advisor: Optional[AIAdvisor] = None


def get_ai_advisor() -> AIAdvisor:
    """Get or create the AI advisor singleton."""
    global _advisor
    if _advisor is None:
        _advisor = AIAdvisor()
    return _advisor
