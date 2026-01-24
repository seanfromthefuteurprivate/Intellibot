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
from wsb_snake.engines.state_machine import state_machine, update_state, should_strike, get_venom_report
from wsb_snake.engines.probability_engine import probability_engine, get_chop_score, should_block, get_target_levels
from wsb_snake.engines.family_classifier import (
    family_classifier, classify_setup, get_leaderboard,
    record_family_signal, SetupFamily, FamilyLifecycle
)
from wsb_snake.engines.inception_detector import (
    inception_detector, detect_inception, InceptionState, InstabilityState
)
from wsb_snake.collectors.polygon_options import polygon_options
from wsb_snake.config import ZERO_DTE_UNIVERSE


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
            "family_classifications": [],
            "family_leaderboard": [],
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
            
            # Stage 2: Fuse signals with chop filter
            log.info("Stage 2: Fusing signals with chop filter...")
            results["probabilities"] = generate_probabilities(
                results["ignition_signals"],
                results["pressure_signals"],
                results["surge_signals"],
            )
            
            # Apply chop kill filter
            filtered_probs = []
            for prob in results["probabilities"]:
                ticker = prob.get("ticker")
                if ticker:
                    blocked, reason = should_block(ticker)
                    if blocked:
                        log.info(f"🚫 Chop filter blocked {ticker}: {reason}")
                    else:
                        filtered_probs.append(prob)
            results["probabilities"] = filtered_probs
            
            # Stage 2.5: Update state machine for ALL tickers with signals
            log.info("Stage 2.5: Updating state machine...")
            
            # Collect all tickers that have any signals
            all_tickers_with_signals = set()
            for sig in results["ignition_signals"]:
                if sig.get("ticker"):
                    all_tickers_with_signals.add(sig.get("ticker"))
            for sig in results["pressure_signals"]:
                if sig.get("ticker"):
                    all_tickers_with_signals.add(sig.get("ticker"))
            for sig in results["surge_signals"]:
                if sig.get("ticker"):
                    all_tickers_with_signals.add(sig.get("ticker"))
            for prob in results["probabilities"]:
                if prob.get("ticker"):
                    all_tickers_with_signals.add(prob.get("ticker"))
            
            # Update state machine for ALL tickers with signals (enables LURK→COILED→RATTLE)
            for ticker in all_tickers_with_signals:
                # Find matching signals
                ignition = next((s for s in results["ignition_signals"] if s.get("ticker") == ticker), None)
                pressure = next((s for s in results["pressure_signals"] if s.get("ticker") == ticker), None)
                surge = next((s for s in results["surge_signals"] if s.get("ticker") == ticker), None)
                prob = next((p for p in results["probabilities"] if p.get("ticker") == ticker), None)
                
                # Get current price - from prob, or fallback to levels/snapshot
                current_price = 0.0
                if prob and prob.get("entry_price"):
                    current_price = float(prob.get("entry_price", 0) or 0)
                
                if current_price == 0:
                    # Fallback: get price from target levels (which queries Polygon)
                    levels = get_target_levels(ticker)
                    current_price = float(levels.get("current_price", 0) or 0)
                
                # Calculate P(hit target by close) using Probability Engine for ALL tickers
                p_hit = 0.0
                entry_quality = "poor"
                
                if current_price > 0:
                    # Get target level (day high for long, day low for short)
                    direction = "long"
                    if pressure:
                        direction = pressure.get("direction", "long")
                    
                    levels = get_target_levels(ticker)
                    if direction == "long":
                        target = levels.get("day_high", current_price * 1.01)
                    else:
                        target = levels.get("day_low", current_price * 0.99)
                    
                    # Calculate probability - this works even without fused probability
                    prob_estimate = probability_engine.calculate_probability(ticker, target, current_price)
                    p_hit = prob_estimate.p_hit_by_close
                    entry_quality = prob_estimate.entry_quality
                
                # Build probability output for state machine
                # Inject p_hit into existing prob, or create new one
                if prob:
                    prob["p_hit_by_close"] = p_hit
                    prob["probability_win"] = p_hit  # State machine uses this field
                    prob["entry_quality"] = entry_quality
                    prob_output = prob
                else:
                    # Create synthetic probability output for tickers without fused prob
                    # This allows STRIKE to occur based on probability engine alone
                    ignition_score = ignition.get("score", 0) if ignition else 0
                    pressure_score = pressure.get("score", 0) if pressure else 0
                    surge_score = surge.get("score", 0) if surge else 0
                    
                    # Combined score from components
                    combined_score = (ignition_score * 0.4 + pressure_score * 0.35 + surge_score * 0.25)
                    
                    prob_output = {
                        "ticker": ticker,
                        "probability_win": p_hit,
                        "p_hit_by_close": p_hit,
                        "combined_score": combined_score,
                        "entry_quality": entry_quality,
                        "entry_price": current_price,
                    }
                
                # Update state
                state = update_state(
                    ticker=ticker,
                    ignition_signal=ignition,
                    pressure_signal=pressure,
                    surge_signal=surge,
                    probability_output=prob_output,
                    current_price=current_price,
                )
                
                log.debug(f"State: {ticker} -> {state.get('state')} | P(hit)={p_hit:.2%}")
            
            # Stage 2.6: Classify setups into families
            log.info("Stage 2.6: Classifying setups into families...")
            
            # Get market regime for family classification
            from wsb_snake.engines.pressure_engine import get_market_regime
            regime_data = get_market_regime()
            current_regime = regime_data.get("regime", "neutral")
            
            # Get family leaderboard first
            results["family_leaderboard"] = get_leaderboard(current_regime)
            alive_families = [f for f in results["family_leaderboard"] if f.get("alive")]
            log.info(f"   Family Leaderboard: {len(alive_families)} alive of {len(results['family_leaderboard'])}")
            
            # Classify each ticker into families
            for ticker in all_tickers_with_signals:
                ignition = next((s for s in results["ignition_signals"] if s.get("ticker") == ticker), None)
                pressure = next((s for s in results["pressure_signals"] if s.get("ticker") == ticker), None)
                surge = next((s for s in results["surge_signals"] if s.get("ticker") == ticker), None)
                
                families = classify_setup(
                    ticker=ticker,
                    regime=current_regime,
                    ignition_signal=ignition,
                    pressure_signal=pressure,
                    surge_signal=surge,
                )
                
                for fam in families:
                    fam["ticker"] = ticker
                    results["family_classifications"].append(fam)
                    
                    if fam.get("lifecycle") in ["alive", "peaked"]:
                        log.info(f"   🎯 {ticker}: {fam.get('family')} is {fam.get('lifecycle')} (viability={fam.get('viability_score'):.2f})")
            
            # Stage 2.7: Inception Detection (Convex Instability)
            log.info("Stage 2.7: Running Inception Detection...")
            results["inception_states"] = []
            
            for ticker in all_tickers_with_signals:
                try:
                    ignition = next((s for s in results["ignition_signals"] if s.get("ticker") == ticker), None)
                    pressure = next((s for s in results["pressure_signals"] if s.get("ticker") == ticker), None)
                    prob = next((p for p in results["probabilities"] if p.get("ticker") == ticker), None)
                    
                    current_price = prob.get("entry_price", 0) if prob else 0
                    
                    if current_price <= 0:
                        continue
                    
                    price_bars = ignition.get("price_bars", []) if ignition else []
                    current_volume = ignition.get("volume", 0) if ignition else 0
                    
                    ticker_changes = {}
                    for t in ["SPY", "QQQ", "IWM", "VIX"]:
                        sig = next((s for s in results["ignition_signals"] if s.get("ticker") == t), None)
                        if sig:
                            ticker_changes[t] = sig.get("change_pct", 0)
                    
                    options_data = None
                    try:
                        options_data = polygon_options.get_full_options_analysis(ticker, current_price)
                    except Exception as e:
                        log.debug(f"Options analysis unavailable for {ticker}: {e}")
                    
                    inception_state = detect_inception(
                        ticker=ticker,
                        price_bars=price_bars,
                        current_price=current_price,
                        current_volume=current_volume,
                        ticker_changes=ticker_changes,
                        options_data=options_data,
                        related_tickers=ticker_changes,
                    )
                    
                    results["inception_states"].append(inception_state.to_dict())
                    
                    if inception_state.inception_detected:
                        log.warning(f"🌀 INCEPTION: {ticker} | Index={inception_state.instability_index:.2f} | Confidence={inception_state.inception_confidence:.2%}")
                        
                        if self._should_send_alert(ticker):
                            self._send_inception_alert(ticker, inception_state, options_data, session_info)
                            results["alerts_sent"] += 1
                            
                except Exception as e:
                    log.warning(f"Inception detection failed for {ticker}: {e}")
            
            inception_count = len([s for s in results["inception_states"] if s.get("inception_detected")])
            log.info(f"   Inception detections: {inception_count}")
            
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
                
                # Get family classification for this ticker
                ticker_families = [f for f in results["family_classifications"] if f.get("ticker") == ticker]
                top_family = ticker_families[0] if ticker_families else None
                
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
                        "family": top_family.get("family") if top_family else None,
                        "family_viability": top_family.get("viability_score") if top_family else 0,
                    },
                    "evidence": prob.get("bull_thesis", []) + prob.get("bear_thesis", []),
                }
                signal_id = save_signal(signal_data)
                prob["signal_id"] = signal_id
                
                # Send alerts for A+ and A tier - but only if state machine says STRIKE
                # AND family is alive (viable)
                if tier in ["A+", "A"]:
                    # Check if state machine has reached STRIKE state
                    family_alive = top_family and top_family.get("lifecycle") in ["alive", "peaked"]
                    
                    if should_strike(ticker) and family_alive:
                        family_name = top_family.get("family", "unknown") if top_family else "unknown"
                        log.info(f"🐍 STRIKE STATE: {ticker} ({family_name}) - sending alert!")
                        
                        if self._should_send_alert(ticker):
                            # Add family info to probability for alert
                            prob["setup_family"] = family_name
                            prob["family_viability"] = top_family.get("viability_score", 0) if top_family else 0
                            
                            self._send_alert(prob, tier, session_info)
                            results["alerts_sent"] += 1
                            
                            # Record family signal for learning
                            record_family_signal(ticker, family_name)
                            
                            # Paper trade high-conviction signals
                            position = paper_trader.evaluate_signal(prob)
                            if position:
                                results["paper_trades"] += 1
                    elif should_strike(ticker):
                        # State machine ready but no viable family
                        log.info(f"👁️ WATCH: {ticker} score {score:.0f} - no viable family")
                    else:
                        # State machine not ready - log as watch
                        log.info(f"👁️ WATCH: {ticker} score {score:.0f} - awaiting state escalation")
            
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
            log.info(f"   Alive families: {len([f for f in results['family_classifications'] if f.get('lifecycle') in ['alive', 'peaked']])}")
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
    
    def _send_inception_alert(
        self,
        ticker: str,
        inception_state: InceptionState,
        options_data: Optional[Dict],
        session_info: Dict
    ) -> None:
        """Send Telegram alert for inception (convex instability) detection."""
        message = f"""🌀 **INCEPTION DETECTED — ${ticker}**
Instability Index: {inception_state.instability_index:.2f}
Confidence: {inception_state.inception_confidence:.0%}
State: {inception_state.instability_state.value.upper()}

🔬 **Sensor Readings**
• Event Horizon: {inception_state.event_horizon.sensitivity_score:.2f}
• Correlation Fracture: {inception_state.correlation_fracture.fracture_score:.2f}
• Liquidity Fragility: {inception_state.liquidity_elasticity.fragility_score:.2f}
• Temporal Distortion: {inception_state.temporal_anomaly.time_distortion_score:.2f}
• Attention Surge: {inception_state.attention_surge.attention_acceleration:.2f}

⚠️ **Signals**"""
        
        for signal in inception_state.signals[:5]:
            message += f"\n• {signal}"
        
        if options_data and options_data.get("has_data"):
            gex = options_data.get("gex", {})
            max_pain = options_data.get("max_pain", {})
            
            message += f"""

📊 **Options Structure**
• GEX Regime: {gex.get('gex_regime', 'unknown').upper()}
• Max Pain: ${max_pain.get('max_pain_strike', 0):.0f} ({max_pain.get('distance_pct', 0):.1f}% away)"""
            
            walls = options_data.get("volume_walls", {})
            if walls.get("nearest_resistance"):
                message += f"\n• Resistance Wall: ${walls['nearest_resistance']:.0f}"
            if walls.get("nearest_support"):
                message += f"\n• Support Wall: ${walls['nearest_support']:.0f}"
        
        mins_to_close = session_info.get("minutes_to_close", 0)
        
        message += f"""

⏰ **Timing**
Minutes to close: {mins_to_close:.0f}

💡 **Interpretation**
This is a phase transition detection — small moves may create outsized effects. 
The market is entering a state of convex instability."""
        
        try:
            send_telegram_alert(message)
            self.alerts_sent_today[ticker] = datetime.utcnow()
            log.info(f"Inception alert sent for {ticker}")
        except Exception as e:
            log.error(f"Failed to send inception alert: {e}")
    
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
        """Send end-of-day report via Telegram (VENOM state)."""
        # Get venom report from state machine
        venom = get_venom_report()
        
        report = paper_trader.get_daily_report()
        message = paper_trader.format_daily_report(report)
        
        # Add state machine stats
        message += f"""

🐍 **State Machine Stats**
Total strikes: {venom.get('total_strikes', 0)}
Outcomes tracked: {venom.get('total_outcomes', 0)}
Win rate: {venom.get('win_rate', 0)*100:.0f}%
Avg win: {venom.get('avg_win_pct', 0):.1f}%
Avg loss: {venom.get('avg_loss_pct', 0):.1f}%"""
        
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
