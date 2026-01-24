"""
Engine Orchestrator

Coordinates all 6 engines to run the complete 0DTE intelligence pipeline.

Pipeline:
1. Ignition Detector -> Early momentum signals
2. Pressure Engine -> Options flow analysis
3. Surge Hunter -> Power hour setups
4. Probability Generator -> Fuse all signals
5. Learning Memory -> Track and learn from outcomes
6. Paper Trader -> Simulate trades

Output: Telegram alerts for high-conviction setups
"""

from datetime import datetime
from typing import Dict, List, Any, Optional

from wsb_snake.utils.logger import log
from wsb_snake.utils.session_regime import (
    get_session_info, is_market_open, is_power_hour,
    should_scan_for_signals, get_0dte_urgency
)
from wsb_snake.db.database import save_signal, get_daily_stats
from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert

# Import engines
from wsb_snake.engines.ignition_detector import run_ignition_scan
from wsb_snake.engines.pressure_engine import run_pressure_scan
from wsb_snake.engines.surge_hunter import run_surge_hunt
from wsb_snake.engines.probability_generator import generate_probabilities
from wsb_snake.engines.learning_memory import learning_memory
from wsb_snake.engines.paper_trader import paper_trader


class SnakeOrchestrator:
    """
    Orchestrates the complete 0DTE signal pipeline.
    """
    
    # Alert thresholds
    A_PLUS_THRESHOLD = 85
    A_THRESHOLD = 70
    B_THRESHOLD = 50
    
    def __init__(self):
        self.last_scan_time: Optional[datetime] = None
        self.alerts_sent_today: Dict[str, datetime] = {}
        self.alert_cooldown_minutes = 30  # Don't spam same ticker
    
    def run_full_pipeline(self) -> Dict:
        """
        Run the complete signal pipeline.
        
        Returns:
            Dict with pipeline results
        """
        start_time = datetime.utcnow()
        session_info = get_session_info()
        
        log.info("=" * 50)
        log.info(f"🐍 WSB Snake Pipeline Starting")
        log.info(f"   Session: {session_info['session']}")
        log.info(f"   Market Open: {session_info['is_open']}")
        log.info(f"   Power Hour: {session_info['is_power_hour']}")
        log.info("=" * 50)
        
        results = {
            "timestamp": start_time.isoformat(),
            "session": session_info,
            "ignition_signals": [],
            "pressure_signals": [],
            "surge_signals": [],
            "probabilities": [],
            "alerts_sent": 0,
            "paper_trades": 0,
            "errors": [],
        }
        
        try:
            # Stage 1: Run all detection engines in sequence
            log.info("Stage 1: Running detection engines...")
            
            # Engine 1: Ignition Detector
            log.info("  → Engine 1: Ignition Detector")
            results["ignition_signals"] = run_ignition_scan()
            
            # Engine 2: Pressure Engine
            log.info("  → Engine 2: Pressure Engine")
            results["pressure_signals"] = run_pressure_scan()
            
            # Engine 3: Surge Hunter (more active during power hour)
            if is_power_hour() or session_info.get("minutes_to_close", 999) < 120:
                log.info("  → Engine 3: Surge Hunter (power hour active)")
                results["surge_signals"] = run_surge_hunt()
            else:
                log.info("  → Engine 3: Surge Hunter (skipped - not power hour)")
            
            # Stage 2: Fuse signals
            log.info("Stage 2: Fusing signals...")
            results["probabilities"] = generate_probabilities(
                results["ignition_signals"],
                results["pressure_signals"],
                results["surge_signals"],
            )
            
            # Stage 3: Process high-conviction signals
            log.info("Stage 3: Processing alerts and trades...")
            for prob in results["probabilities"]:
                ticker = prob.get("ticker")
                score = prob.get("combined_score", 0)
                
                if not ticker:
                    continue
                    
                # Determine tier
                if score >= self.A_PLUS_THRESHOLD:
                    tier = "A+"
                elif score >= self.A_THRESHOLD:
                    tier = "A"
                elif score >= self.B_THRESHOLD:
                    tier = "B"
                else:
                    tier = "C"
                
                # Save signal to database
                signal_data = {
                    "ticker": ticker,
                    "timestamp": datetime.utcnow().isoformat(),
                    "setup_type": prob.get("action", "UNKNOWN"),
                    "score": score,
                    "tier": tier,
                    "price": prob.get("entry_price"),
                    "session_type": session_info.get("session"),
                    "minutes_to_close": session_info.get("minutes_to_close"),
                    "options_pressure_score": prob.get("pressure_score"),
                    "features": {
                        "ignition_score": prob.get("ignition_score"),
                        "pressure_score": prob.get("pressure_score"),
                        "surge_score": prob.get("surge_score"),
                        "direction": prob.get("direction"),
                    },
                    "evidence": prob.get("bull_thesis", []) + prob.get("bear_thesis", []),
                }
                signal_id = save_signal(signal_data)
                prob["signal_id"] = signal_id
                
                # Send alerts for A+ and A tier
                if tier in ["A+", "A"]:
                    if self._should_send_alert(ticker):
                        self._send_alert(prob, tier, session_info)
                        results["alerts_sent"] += 1
                        
                        # Paper trade high-conviction signals
                        position = paper_trader.evaluate_signal(prob)
                        if position:
                            results["paper_trades"] += 1
            
            # Stage 4: Update paper positions
            log.info("Stage 4: Managing paper positions...")
            # Get current prices for position management
            current_prices = self._get_current_prices(results["probabilities"])
            
            filled = paper_trader.fill_pending_orders(current_prices)
            closed = paper_trader.check_exits(current_prices)
            
            log.info(f"   Filled: {len(filled)} | Closed: {len(closed)}")
            
            # Summary
            duration = (datetime.utcnow() - start_time).total_seconds()
            log.info("=" * 50)
            log.info(f"🐍 Pipeline Complete in {duration:.1f}s")
            log.info(f"   Ignition signals: {len(results['ignition_signals'])}")
            log.info(f"   Pressure signals: {len(results['pressure_signals'])}")
            log.info(f"   Surge signals: {len(results['surge_signals'])}")
            log.info(f"   Probabilities: {len(results['probabilities'])}")
            log.info(f"   Alerts sent: {results['alerts_sent']}")
            log.info(f"   Paper trades: {results['paper_trades']}")
            log.info("=" * 50)
            
            self.last_scan_time = datetime.utcnow()
            
        except Exception as e:
            log.error(f"Pipeline error: {e}")
            results["errors"].append(str(e))
        
        return results
    
    def _should_send_alert(self, ticker: str) -> bool:
        """Check if we should send an alert for this ticker."""
        if ticker not in self.alerts_sent_today:
            return True
        
        last_alert = self.alerts_sent_today[ticker]
        minutes_since = (datetime.utcnow() - last_alert).total_seconds() / 60
        
        return minutes_since >= self.alert_cooldown_minutes
    
    def _send_alert(self, prob: Dict, tier: str, session_info: Dict) -> None:
        """Send Telegram alert for high-conviction signal."""
        ticker = prob.get("ticker")
        score = prob.get("combined_score", 0)
        action = prob.get("action", "WATCH")
        direction = prob.get("direction", "neutral")
        
        # Format message
        emoji = "🔥" if tier == "A+" else "⚡"
        dir_emoji = "📈" if direction == "long" else "📉" if direction == "short" else "➡️"
        
        message = f"""{emoji} **WSB SNAKE ALERT — ${ticker}**
Score: {score:.0f}/100 | Tier: {tier}

{dir_emoji} **Action: {action}**
Direction: {direction.upper()}

📊 **Component Scores**
• Ignition: {prob.get('ignition_score', 0):.0f}
• Pressure: {prob.get('pressure_score', 0):.0f}
• Surge: {prob.get('surge_score', 0):.0f}

💡 **Thesis**"""
        
        # Add bull thesis
        for thesis in prob.get("bull_thesis", [])[:3]:
            message += f"\n• {thesis}"
        
        # Add bear thesis if any
        bear = prob.get("bear_thesis", [])
        if bear:
            message += "\n⚠️ **Risks**"
            for thesis in bear[:2]:
                message += f"\n• {thesis}"
        
        # Trade levels
        entry = prob.get("entry_price", 0)
        stop = prob.get("stop_loss", 0)
        t1 = prob.get("target_1", 0)
        
        if entry > 0:
            message += f"""

🎯 **Levels**
Entry: ${entry:.2f}
Stop: ${stop:.2f}
Target 1: ${t1:.2f}
R:R = {prob.get('risk_reward_ratio', 0):.1f}"""
        
        # Timing
        urgency = prob.get("time_sensitivity", "medium")
        mins_to_close = session_info.get("minutes_to_close", 0)
        
        message += f"""

⏰ **Timing**
Urgency: {urgency.upper()}
Minutes to close: {mins_to_close:.0f}"""
        
        # Send alert
        try:
            send_telegram_alert(message)
            self.alerts_sent_today[ticker] = datetime.utcnow()
            log.info(f"Alert sent for {ticker}")
        except Exception as e:
            log.error(f"Failed to send alert: {e}")
    
    def _get_current_prices(self, probabilities: List[Dict]) -> Dict[str, float]:
        """Extract current prices from probability outputs."""
        result: Dict[str, float] = {}
        for p in probabilities:
            ticker = p.get("ticker")
            price = p.get("entry_price", 0)
            if ticker and isinstance(ticker, str) and price:
                result[ticker] = float(price)
        return result
    
    def send_daily_report(self) -> None:
        """Send end-of-day report via Telegram."""
        report = paper_trader.get_daily_report()
        message = paper_trader.format_daily_report(report)
        
        # Add learning stats
        learning_summary = learning_memory.get_learning_summary()
        
        message += f"""

🧠 **Learning Stats**
Training signals: {learning_summary.get('total_training_signals', 0)}
Avg win rate: {learning_summary.get('average_win_rate', 0)*100:.0f}%
Best feature: {learning_summary.get('best_feature', 'N/A')} ({learning_summary.get('best_weight', 1.0):.2f})
Worst feature: {learning_summary.get('worst_feature', 'N/A')} ({learning_summary.get('worst_weight', 1.0):.2f})"""
        
        try:
            send_telegram_alert(message)
            log.info("Daily report sent")
        except Exception as e:
            log.error(f"Failed to send daily report: {e}")


# Global instance
orchestrator = SnakeOrchestrator()


def run_pipeline() -> Dict:
    """Run the full pipeline. Convenience function for scheduler."""
    return orchestrator.run_full_pipeline()


def send_daily_summary() -> None:
    """Send daily summary. Called at end of trading day."""
    orchestrator.send_daily_report()
