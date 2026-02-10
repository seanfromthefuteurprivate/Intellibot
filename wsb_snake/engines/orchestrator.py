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
from wsb_snake.engines.chart_brain import get_chart_brain
from wsb_snake.collectors.polygon_options import polygon_options
from wsb_snake.collectors.reddit_collector import get_wsb_trending, get_wsb_catalysts
from wsb_snake.collectors.finnhub_collector import finnhub_collector
from wsb_snake.collectors.sec_edgar_collector import sec_edgar_collector
from wsb_snake.collectors.finviz_collector import finviz_collector
from wsb_snake.collectors.finra_darkpool import finra_darkpool
from wsb_snake.collectors.options_flow_scraper import options_flow_scraper
from wsb_snake.collectors.level2_simulator import level2_simulator
from wsb_snake.collectors.congressional_trading import congressional_trading
from wsb_snake.collectors.fred_collector import fred_collector
from wsb_snake.collectors.vix_structure import vix_structure
from wsb_snake.collectors.earnings_calendar import earnings_calendar
from wsb_snake.collectors.alpha_vantage_collector import alpha_vantage
from wsb_snake.collectors.benzinga_news import benzinga_news
from wsb_snake.engines.strategy_classifier import strategy_classifier, StrategyType
from wsb_snake.engines.multi_day_scanner import multi_day_scanner
from wsb_snake.engines.zero_dte_volatility import zero_dte_volatility
from wsb_snake.engines.earnings_otm_engine import earnings_otm_engine
from wsb_snake.training.historical_trainer import historical_trainer
from wsb_snake.learning.pattern_memory import pattern_memory
from wsb_snake.learning.time_learning import time_learning
from wsb_snake.learning.event_outcomes import event_outcome_db
from wsb_snake.learning.stalking_mode import stalking_mode, StalkState
from wsb_snake.learning.session_learnings import session_learnings, battle_plan
from wsb_snake.engines.spy_scalper import spy_scalper
from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.utils.cpl_gate import check as cpl_check, block_trade as cpl_block
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
            # Stage 0: Scan Reddit WSB for trending tickers and catalysts
            log.info("Stage 0: Scanning Reddit r/wallstreetbets...")
            try:
                wsb_trending = get_wsb_trending()
                wsb_catalysts = get_wsb_catalysts()
                results["wsb_trending"] = wsb_trending
                results["wsb_catalysts"] = wsb_catalysts
                
                if wsb_trending:
                    trending_str = ", ".join([f"{t['ticker']}({t['mentions']})" for t in wsb_trending[:5]])
                    log.info(f"   WSB Trending: {trending_str}")
                if wsb_catalysts:
                    log.info(f"   WSB Catalysts: {len(wsb_catalysts)} detected")
            except Exception as e:
                log.warning(f"   WSB scan error: {e}")
                results["wsb_trending"] = []
                results["wsb_catalysts"] = []
            
            # Stage 0.5: Collect alternative data (Finnhub, SEC EDGAR, Finviz)
            log.info("Stage 0.5: Collecting alternative data sources...")
            results["alt_data"] = {}
            
            try:
                # Sample a few high-priority tickers for alt data
                priority_tickers = ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"]
                
                for ticker in priority_tickers[:3]:
                    alt = {}
                    
                    # Finnhub: News + Social + Insider sentiment
                    try:
                        finnhub_data = finnhub_collector.get_all_sentiment(ticker)
                        alt["finnhub"] = {
                            "sentiment": finnhub_data.get("combined_score", 0),
                            "direction": finnhub_data.get("direction", "neutral"),
                            "news_buzz": finnhub_data.get("news", {}).get("buzz", 0),
                            "social_mentions": finnhub_data.get("social", {}).get("mentions", 0),
                            "insider_buying": finnhub_data.get("insider", {}).get("buying", False),
                        }
                    except Exception as e:
                        log.debug(f"   Finnhub error for {ticker}: {e}")
                        alt["finnhub"] = {"sentiment": 0, "direction": "neutral"}
                    
                    # Benzinga: Premium financial news
                    try:
                        benzinga_data = benzinga_news.get_news(tickers=[ticker], hours_back=4)
                        # Simple sentiment: count positive vs negative words in headlines
                        sentiment_score = 0
                        for article in benzinga_data:
                            title = article.get("title", "").lower()
                            if any(w in title for w in ["surge", "jump", "rally", "beat", "upgrade"]):
                                sentiment_score += 1
                            elif any(w in title for w in ["drop", "fall", "miss", "downgrade", "warning"]):
                                sentiment_score -= 1
                        alt["benzinga"] = {
                            "articles": len(benzinga_data),
                            "sentiment": sentiment_score,
                            "headlines": [a.get("title", "")[:50] for a in benzinga_data[:3]],
                        }
                        if benzinga_data:
                            log.info(f"   Benzinga: {len(benzinga_data)} articles for {ticker}")
                    except Exception as e:
                        log.debug(f"   Benzinga error for {ticker}: {e}")
                        alt["benzinga"] = {"articles": 0, "sentiment": 0}
                    
                    # SEC EDGAR: Insider activity
                    try:
                        edgar_data = sec_edgar_collector.get_insider_activity(ticker)
                        alt["sec_edgar"] = {
                            "signal": edgar_data.get("signal", "low_activity"),
                            "filings_this_week": edgar_data.get("filings_this_week", 0),
                        }
                    except Exception as e:
                        log.debug(f"   SEC EDGAR error for {ticker}: {e}")
                        alt["sec_edgar"] = {"signal": "unknown"}
                    
                    # Finviz: Unusual volume
                    try:
                        finviz_data = finviz_collector.get_unusual_volume(ticker)
                        alt["finviz"] = {
                            "unusual_volume": finviz_data.get("unusual", False),
                            "rel_volume": finviz_data.get("rel_volume", 1.0),
                            "signal": finviz_data.get("signal", "normal"),
                        }
                    except Exception as e:
                        log.debug(f"   Finviz error for {ticker}: {e}")
                        alt["finviz"] = {"unusual_volume": False, "rel_volume": 1.0}
                    
                    results["alt_data"][ticker] = alt
                
                # Log summary
                volume_alerts = [t for t, d in results["alt_data"].items() if d.get("finviz", {}).get("unusual_volume")]
                sentiment_alerts = [t for t, d in results["alt_data"].items() if d.get("finnhub", {}).get("direction") != "neutral"]
                
                if volume_alerts:
                    log.info(f"   Unusual volume: {', '.join(volume_alerts)}")
                if sentiment_alerts:
                    log.info(f"   Sentiment signals: {', '.join(sentiment_alerts)}")
                    
            except Exception as e:
                log.warning(f"   Alt data collection error: {e}")
            
            # Stage 0.6: Advanced data (Dark Pool, Options Flow, Level 2 Sim)
            log.info("Stage 0.6: Collecting advanced market structure data...")
            results["advanced_data"] = {}
            
            try:
                for ticker in ["SPY", "QQQ", "TSLA"][:2]:
                    adv = {}
                    
                    # FINRA Dark Pool data (free)
                    try:
                        dark_pool = finra_darkpool.get_dark_pool_ratio(ticker)
                        adv["dark_pool"] = {
                            "ats_volume": dark_pool.get("ats_volume", 0),
                            "signal": dark_pool.get("signal", "unknown"),
                            "top_ats": dark_pool.get("top_ats", [])[:2],
                        }
                    except Exception as e:
                        log.debug(f"Dark pool error for {ticker}: {e}")
                        adv["dark_pool"] = {"signal": "unknown"}
                    
                    # Unusual Options Flow (scraped/simulated)
                    try:
                        flow_signals = options_flow_scraper.get_smart_money_signals([ticker])
                        flow = flow_signals.get(ticker, {})
                        adv["options_flow"] = {
                            "signal": flow.get("signal", "neutral"),
                            "confidence": flow.get("confidence", 0),
                            "put_call_ratio": flow.get("put_call_ratio", 1.0),
                        }
                    except Exception as e:
                        log.debug(f"Options flow error for {ticker}: {e}")
                        adv["options_flow"] = {"signal": "neutral"}
                    
                    # Simulated Level 2 from options data
                    try:
                        l2_book = level2_simulator.get_synthetic_order_book(ticker)
                        adv["level2"] = {
                            "bias": l2_book.get("bias", "neutral"),
                            "imbalance": l2_book.get("imbalance", 0),
                            "key_support": l2_book.get("key_support", 0),
                            "key_resistance": l2_book.get("key_resistance", 0),
                        }
                    except Exception as e:
                        log.debug(f"Level 2 sim error for {ticker}: {e}")
                        adv["level2"] = {"bias": "neutral"}
                    
                    results["advanced_data"][ticker] = adv
                
                # Log summary
                bullish_flow = [t for t, d in results["advanced_data"].items() 
                               if d.get("options_flow", {}).get("signal") == "bullish"]
                bearish_flow = [t for t, d in results["advanced_data"].items() 
                               if d.get("options_flow", {}).get("signal") == "bearish"]
                
                if bullish_flow:
                    log.info(f"   Bullish flow: {', '.join(bullish_flow)}")
                if bearish_flow:
                    log.info(f"   Bearish flow: {', '.join(bearish_flow)}")
                    
            except Exception as e:
                log.warning(f"   Advanced data collection error: {e}")
            
            # Stage 0.7: Collect expanded data sources (Phase 11)
            log.info("Stage 0.7: Collecting expanded data sources...")
            results["expanded_data"] = {
                "congressional": {},
                "macro_regime": {},
                "vix_structure": {},
                "earnings": {},
                "sentiment": {},
            }
            
            try:
                # Congressional Trading (100% free)
                try:
                    hot_congress = congressional_trading.get_hot_tickers(days_back=14)
                    results["expanded_data"]["congressional"]["hot_tickers"] = hot_congress[:10]
                    
                    for ticker in ZERO_DTE_UNIVERSE[:5]:
                        congress_signal = congressional_trading.get_trades_for_ticker(ticker)
                        if congress_signal.get("total_trades", 0) > 0:
                            results["expanded_data"]["congressional"][ticker] = congress_signal
                    
                    log.info(f"   Congressional: {len(hot_congress)} hot tickers")
                except Exception as e:
                    log.debug(f"Congressional trading error: {e}")
                
                # FRED Macro Regime
                try:
                    macro_regime = fred_collector.get_macro_regime()
                    results["expanded_data"]["macro_regime"] = macro_regime
                    log.info(f"   Macro: {macro_regime.get('overall_regime', 'unknown')} | VIX: {macro_regime.get('metrics', {}).get('vix', 0):.1f}")
                except Exception as e:
                    log.debug(f"FRED error: {e}")
                
                # VIX Term Structure
                try:
                    vix_signal = vix_structure.get_trading_signal()
                    results["expanded_data"]["vix_structure"] = vix_signal
                    log.info(f"   VIX: {vix_signal.get('vix', 0):.1f} | Structure: {vix_signal.get('structure', 'unknown')}")
                except Exception as e:
                    log.debug(f"VIX structure error: {e}")
                
                # Earnings Calendar
                try:
                    this_week_earnings = earnings_calendar.get_this_week_earnings(ZERO_DTE_UNIVERSE)
                    results["expanded_data"]["earnings"]["this_week"] = this_week_earnings
                    
                    if this_week_earnings:
                        earnings_str = ", ".join([f"{e['symbol']}({e['days_until']}d)" for e in this_week_earnings[:5]])
                        log.info(f"   Earnings This Week: {earnings_str}")
                except Exception as e:
                    log.debug(f"Earnings calendar error: {e}")
                
                # Alpha Vantage Sentiment (rate limited)
                try:
                    market_sentiment = alpha_vantage.get_market_sentiment()
                    results["expanded_data"]["sentiment"]["market"] = market_sentiment
                    log.info(f"   Market Sentiment: {market_sentiment.get('market_sentiment', 'unknown')}")
                except Exception as e:
                    log.debug(f"Alpha Vantage error: {e}")
                    
            except Exception as e:
                log.warning(f"   Expanded data error: {e}")
            
            # Stage 0.8: Run multi-day scanner (every 4 hours)
            log.info("Stage 0.8: Multi-day opportunity scan...")
            try:
                # Only run full scan periodically to save API calls
                if multi_day_scanner.last_scan is None or \
                   (datetime.now() - multi_day_scanner.last_scan).total_seconds() > 14400:
                    multi_day_results = multi_day_scanner.run_full_scan()
                    results["multi_day_setups"] = multi_day_results
                    log.info(f"   Multi-day setups: {multi_day_results.get('total_setups', 0)}")
                else:
                    results["multi_day_setups"] = {
                        "total_setups": len(multi_day_scanner.active_setups),
                        "cached": True,
                    }
                    log.info(f"   Multi-day: Using cached ({len(multi_day_scanner.active_setups)} setups)")
            except Exception as e:
                log.warning(f"   Multi-day scanner error: {e}")
            
            # Stage 0.85: Learning Module Updates
            log.info("Stage 0.85: Advanced learning and stalking...")
            try:
                # Get time-of-day recommendation
                time_rec = time_learning.get_recommendation()
                results["learning"] = {
                    "time_quality": time_rec.quality_score,
                    "time_recommendation": time_rec.recommendation,
                    "time_session": time_rec.session
                }
                
                if time_rec.quality_score >= 60:
                    log.info(f"   Time Learning: {time_rec.session.upper()} quality={time_rec.quality_score:.0f} ({time_rec.recommendation})")
                elif time_rec.quality_score < 40:
                    log.info(f"   Time Learning: LOW quality period ({time_rec.quality_score:.0f})")
                
                # Check stalked setups
                # Gather current prices for stalking check
                stalk_prices = {}
                for ticker in ZERO_DTE_UNIVERSE:
                    levels = get_target_levels(ticker)
                    if levels.get("current_price"):
                        stalk_prices[ticker] = float(levels["current_price"])
                
                stalk_alerts = stalking_mode.check_all_setups(stalk_prices)
                
                # Also check for exit signals on filled positions
                exit_alerts = stalking_mode.check_exits(stalk_prices)
                stalk_alerts.extend(exit_alerts)
                
                results["learning"]["stalk_alerts"] = len(stalk_alerts)
                
                # Send alerts for triggered stalks (entry and exit signals)
                for alert in stalk_alerts:
                    if alert.urgency >= 4:
                        # Use formatted message for actionable trade signals
                        if alert.alert_type in ["entry", "exit"]:
                            alert_msg = alert.format_telegram_message()
                        else:
                            # Null-safe formatting for non-entry/exit alerts
                            entry = alert.entry_price or 0
                            target = alert.target_price or 0
                            stop = alert.stop_loss or 0
                            trade_type = alert.trade_type or "STOCK"
                            direction = (alert.direction or "long").upper()
                            
                            alert_msg = f"""
*[STALKED SETUP {alert.alert_type.upper()}]*

{alert.symbol} - {alert.message}

Entry: ${entry:.2f} | Target: ${target:.2f} | Stop: ${stop:.2f}
Type: {trade_type} ({direction})

_Urgency: {alert.urgency}/5_
"""
                        send_telegram_alert(alert_msg)
                        log.info(f"   Stalk alert: {alert.symbol} - {alert.alert_type}")
                
                # Add new setups to stalk from multi-day scanner with trade details
                for setup in multi_day_scanner.active_setups[:5]:
                    if setup.confidence >= 60:
                        # Check if not already stalking
                        existing = [s for s in stalking_mode.get_active_setups() 
                                    if s.symbol == setup.symbol and s.setup_type == setup.setup_type]
                        if not existing:
                            # Get current price for trade level calculation
                            current = stalk_prices.get(setup.symbol, 0)
                            if current > 0:
                                # Calculate trade levels based on expected move
                                expected_pct = getattr(setup, 'expected_move_pct', None) or getattr(setup, 'expected_move', 3.0)
                                direction = "long" if setup.direction == "bullish" else "short"
                                
                                if direction == "long":
                                    entry_price = current
                                    target_price = current * (1 + expected_pct / 100)
                                    stop_loss = current * (1 - expected_pct / 200)  # 2:1 R:R
                                    trade_type = "CALLS"
                                else:
                                    entry_price = current
                                    target_price = current * (1 - expected_pct / 100)
                                    stop_loss = current * (1 + expected_pct / 200)
                                    trade_type = "PUTS"
                                
                                stalking_mode.add_setup(
                                    symbol=setup.symbol,
                                    setup_type=setup.setup_type,
                                    trigger_price=entry_price,
                                    trigger_condition="above" if direction == "long" else "below",
                                    catalyst=setup.catalyst,
                                    expected_move=expected_pct,
                                    expires_hours=72,
                                    entry_price=entry_price,
                                    target_price=target_price,
                                    stop_loss=stop_loss,
                                    direction=direction,
                                    trade_type=trade_type
                                )
                                log.info(f"   Stalking: {setup.symbol} {trade_type} Entry ${entry_price:.2f} Target ${target_price:.2f}")
                
                log.info(f"   Active stalks: {len(stalking_mode.get_active_setups())}")
                
            except Exception as e:
                log.warning(f"   Learning module error: {e}")
            
            # Stage 0.9: Phase 1 Strategy Engines (0DTE Vol + Earnings OTM)
            log.info("Stage 0.9: Phase 1 strategies (0DTE Vol + Earnings OTM)...")
            results["phase1_setups"] = {
                "zero_dte_volatility": [],
                "earnings_otm": [],
            }
            
            try:
                # 0DTE Volatility Engine - check for vol expansion plays
                vix_data = results.get("expanded_data", {}).get("vix_structure", {})

                # Check economic calendar for today's macro events (NFP, FOMC, CPI)
                macro_event = None
                try:
                    econ_events = fred_collector.get_economic_calendar(weeks=1)
                    today = datetime.now().strftime("%Y-%m-%d")
                    for event in econ_events:
                        if event.get("date") == today and event.get("impact") == "high":
                            macro_event = event.get("name", event.get("event_type", "MACRO_EVENT"))
                            log.info(f"🔥 HIGH-IMPACT MACRO EVENT TODAY: {macro_event}")
                            break
                except Exception as e:
                    log.debug(f"Economic calendar check failed: {e}")

                market_data = {
                    "vix": vix_data.get("vix", 15.0),
                    "vix_structure": vix_data.get("structure", "contango"),
                    "macro_event": macro_event,
                    "SPY_iv": 0.18,
                    "SPY_iv_percentile": 35,
                    "QQQ_iv": 0.22,
                    "QQQ_iv_percentile": 40,
                    "IWM_iv": 0.25,
                    "IWM_iv_percentile": 45,
                }
                
                vol_setups = zero_dte_volatility.scan_volatility_opportunities(market_data)
                results["phase1_setups"]["zero_dte_volatility"] = [
                    {"symbol": s.symbol, "structure": s.structure, "edge": s.vol_edge, "confidence": s.confidence}
                    for s in vol_setups
                ]
                if vol_setups:
                    log.info(f"   0DTE Volatility: {len(vol_setups)} setups found")
                
                # Earnings OTM Engine - check for lotto plays
                earnings_data = results.get("expanded_data", {}).get("earnings", {}).get("this_week", [])
                if earnings_data:
                    earnings_calendar_formatted = []
                    for e in earnings_data:
                        earnings_calendar_formatted.append({
                            "symbol": e.get("symbol"),
                            "days_until": e.get("days_until", 0),
                            "date": e.get("date", ""),
                            "expected_move": e.get("expected_move", 5.0),
                            "iv_rank": e.get("iv_rank", 50),
                            "price": e.get("price", 100),
                        })
                    
                    earnings_setups = earnings_otm_engine.scan_earnings_opportunities(
                        earnings_calendar_formatted, 
                        market_data
                    )
                    results["phase1_setups"]["earnings_otm"] = [
                        {"symbol": s.symbol, "direction": s.direction, "strike": s.strike_type, 
                         "dte": s.dte, "confidence": s.confidence}
                        for s in earnings_setups
                    ]
                    if earnings_setups:
                        log.info(f"   Earnings OTM: {len(earnings_setups)} lotto plays found")
                
                # Send Telegram alerts for high-confidence Phase 1 setups
                for vol_setup in zero_dte_volatility.active_setups:
                    if vol_setup.confidence >= 60:
                        alert_msg = zero_dte_volatility.format_telegram_alert(vol_setup)
                        send_telegram_alert(alert_msg)
                        log.info(f"   Sent 0DTE Vol alert: {vol_setup.symbol}")
                
                for earnings_setup in earnings_otm_engine.active_setups:
                    if earnings_setup.confidence >= 55:
                        alert_msg = earnings_otm_engine.format_telegram_alert(earnings_setup)
                        send_telegram_alert(alert_msg)
                        log.info(f"   Sent Earnings OTM alert: {earnings_setup.symbol}")
                        
            except Exception as e:
                log.warning(f"   Phase 1 engines error: {e}")
            
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
                    
                    current_price = 0.0
                    if prob and prob.get("entry_price"):
                        current_price = float(prob.get("entry_price", 0))
                    elif pressure and pressure.get("price"):
                        current_price = float(pressure.get("price", 0))
                    elif ignition and ignition.get("price"):
                        current_price = float(ignition.get("price", 0))
                    
                    if current_price <= 0:
                        levels = get_target_levels(ticker)
                        current_price = float(levels.get("current_price", 0) or 0)
                    
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
            
            # Stage 2.8: AI Chart Brain Validation
            log.info("Stage 2.8: Running AI Chart Analysis validation...")
            results["ai_validations"] = []
            
            try:
                chart_brain = get_chart_brain()
                
                for prob in results["probabilities"]:
                    ticker = prob.get("ticker")
                    if not ticker:
                        continue
                    
                    # Determine signal direction
                    direction = "NEUTRAL"
                    ignition = next((s for s in results["ignition_signals"] if s.get("ticker") == ticker), None)
                    if ignition:
                        if ignition.get("direction") == "bullish" or ignition.get("bias", "").lower() == "bullish":
                            direction = "LONG"
                        elif ignition.get("direction") == "bearish" or ignition.get("bias", "").lower() == "bearish":
                            direction = "SHORT"
                    
                    # Validate against AI analysis
                    validation = chart_brain.validate_signal(
                        ticker=ticker,
                        signal_direction=direction,
                        signal_score=prob.get("combined_score", 0)
                    )
                    
                    validation["ticker"] = ticker
                    validation["original_score"] = prob.get("combined_score", 0)
                    results["ai_validations"].append(validation)
                    
                    # Update probability score with AI adjustment
                    if validation.get("adjusted_score") != prob.get("combined_score"):
                        prob["ai_adjusted_score"] = validation["adjusted_score"]
                        prob["ai_agrees"] = validation.get("ai_agrees", True)
                        prob["ai_confidence"] = validation.get("ai_confidence", 0.5)
                        
                        if validation.get("ai_agrees") and validation.get("ai_confidence", 0) > 0.7:
                            log.info(f"   🧠 AI confirms {ticker}: {direction} (conf={validation.get('ai_confidence', 0):.0%})")
                        elif not validation.get("ai_agrees") and validation.get("ai_confidence", 0) > 0.6:
                            log.warning(f"   ⚠️ AI disagrees with {ticker}: AI says {validation.get('ai_direction')}")
                            
            except Exception as e:
                log.warning(f"AI chart validation error: {e}")
            
            ai_confirmed = len([v for v in results["ai_validations"] if v.get("ai_agrees") and v.get("ai_confidence", 0) > 0.7])
            log.info(f"   AI confirmations: {ai_confirmed}")
            
            # Stage 2.9: Apply alternative data boosts (Finnhub, Finviz, SEC EDGAR)
            log.info("Stage 2.9: Applying alternative data signal boosts...")
            alt_boosts_applied = 0
            
            for prob in results["probabilities"]:
                ticker = prob.get("ticker")
                if not ticker:
                    continue
                
                current_score = prob.get("ai_adjusted_score", prob.get("combined_score", 0))
                total_boost = 0
                boost_reasons = []
                
                # Get alt data for this ticker
                alt = results.get("alt_data", {}).get(ticker, {})
                
                # Finviz unusual volume boost (most impactful)
                finviz = alt.get("finviz", {})
                if finviz.get("unusual_volume"):
                    rel_vol = finviz.get("rel_volume", 1.0)
                    if rel_vol >= 3.0:
                        boost = 15
                        boost_reasons.append(f"Extreme volume ({rel_vol:.1f}x)")
                    elif rel_vol >= 2.0:
                        boost = 10
                        boost_reasons.append(f"High volume ({rel_vol:.1f}x)")
                    else:
                        boost = 5
                        boost_reasons.append(f"Elevated volume ({rel_vol:.1f}x)")
                    total_boost += boost
                
                # Finnhub sentiment alignment boost
                finnhub = alt.get("finnhub", {})
                signal_direction = "bullish"  # Default
                ignition = next((s for s in results["ignition_signals"] if s.get("ticker") == ticker), None)
                if ignition:
                    signal_direction = ignition.get("direction", "bullish")
                
                finnhub_dir = finnhub.get("direction", "neutral")
                if finnhub_dir != "neutral":
                    if finnhub_dir == signal_direction:
                        boost = 8 if abs(finnhub.get("sentiment", 0)) > 0.2 else 4
                        boost_reasons.append(f"Finnhub sentiment aligned ({finnhub_dir})")
                        total_boost += boost
                    else:
                        # Sentiment disagrees - slight penalty
                        penalty = -5 if abs(finnhub.get("sentiment", 0)) > 0.2 else -2
                        boost_reasons.append(f"Finnhub sentiment contra ({finnhub_dir})")
                        total_boost += penalty
                
                # Finnhub insider buying boost
                if finnhub.get("insider_buying"):
                    total_boost += 5
                    boost_reasons.append("Insider buying detected")
                
                # SEC EDGAR high activity boost
                edgar = alt.get("sec_edgar", {})
                if edgar.get("signal") == "high_activity":
                    total_boost += 5
                    boost_reasons.append("High insider filing activity")
                
                # Apply boost if any
                if total_boost != 0:
                    new_score = max(0, min(100, current_score + total_boost))
                    prob["alt_data_boost"] = total_boost
                    prob["alt_boost_reasons"] = boost_reasons
                    prob["final_score"] = new_score
                    alt_boosts_applied += 1
                    
                    if total_boost > 0:
                        log.info(f"   📊 {ticker}: +{total_boost} pts from alt data ({', '.join(boost_reasons)})")
                    else:
                        log.debug(f"   📊 {ticker}: {total_boost} pts from alt data ({', '.join(boost_reasons)})")
                else:
                    prob["final_score"] = current_score
            
            log.info(f"   Alt data boosts applied: {alt_boosts_applied}")
            
            # Stage 2.95: Pattern Memory matching
            log.info("Stage 2.95: Pattern memory matching...")
            pattern_matches_found = 0
            
            for prob in results["probabilities"]:
                ticker = prob.get("ticker")
                if not ticker:
                    continue
                
                try:
                    # Get price bars from ignition signal (already computed)
                    ignition = next((s for s in results["ignition_signals"] if s.get("ticker") == ticker), None)
                    price_bars = ignition.get("price_bars", []) if ignition else []
                    
                    # Only attempt pattern matching if we have sufficient bar data
                    if len(price_bars) >= 5:
                        # Get context from ignition and pressure
                        rsi = ignition.get("rsi", 50) if ignition else 50
                        vwap_pos = "above" if ignition and ignition.get("above_vwap", False) else "below"
                        
                        matches = pattern_memory.find_matching_patterns(
                            symbol=ticker,
                            bars=price_bars,
                            rsi=rsi,
                            vwap_position=vwap_pos
                        )
                        
                        if matches:
                            best_match = matches[0]
                            prob["pattern_match"] = {
                                "pattern_type": best_match.pattern_type,
                                "similarity": best_match.similarity_score,
                                "historical_win_rate": best_match.historical_win_rate,
                                "direction": best_match.direction
                            }
                            
                            # Only boost if pattern has sufficient historical samples and confidence
                            if best_match.confidence >= 70 and best_match.historical_win_rate >= 0.6 and best_match.similarity_score >= 60:
                                current_score = prob.get("final_score", prob.get("combined_score", 0))
                                boost = min(15, int(best_match.confidence * 0.15))
                                prob["final_score"] = current_score + boost
                                pattern_matches_found += 1
                                log.info(f"   Pattern match: {ticker} {best_match.pattern_type} (sim={best_match.similarity_score:.0f}%, WR={best_match.historical_win_rate:.0%})")
                            
                except Exception as e:
                    log.debug(f"Pattern matching error for {ticker}: {e}")
            
            log.info(f"   Pattern matches: {pattern_matches_found}")
            
            # Stage 3: Process high-conviction signals
            log.info("Stage 3: Processing alerts and trades...")
            for prob in results["probabilities"]:
                ticker = prob.get("ticker")
                # Use final_score (includes AI + alt data boosts), fallback to combined_score
                score = prob.get("final_score", prob.get("ai_adjusted_score", prob.get("combined_score", 0)))
                
                if not ticker:
                    continue
                    
                # Determine tier based on final score
                if score >= self.A_PLUS_THRESHOLD:
                    tier = "A+"
                elif score >= self.A_THRESHOLD:
                    tier = "A"
                elif score >= self.B_THRESHOLD:
                    tier = "B"
                else:
                    tier = "C"
                
                # Add price bars for learning (from ignition signal)
                ignition = next((s for s in results["ignition_signals"] if s.get("ticker") == ticker), None)
                if ignition:
                    prob["price_bars"] = ignition.get("price_bars", [])
                    if not prob.get("strategy"):
                        prob["strategy"] = prob.get("action", "UNKNOWN")
                
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
                            
                            # EXECUTE REAL ALPACA PAPER TRADE - CRITICAL CONNECTION
                            entry_price = prob.get("entry_price") or prob.get("price", 0)
                            direction_raw = prob.get("direction", "long")
                            direction = "long" if direction_raw in ["bullish", "long", "calls"] else "short"
                            
                            # Calculate targets based on direction
                            if direction == "long":
                                target = entry_price * 1.004  # 0.4% target
                                stop = entry_price * 0.997   # 0.3% stop
                            else:
                                target = entry_price * 0.996
                                stop = entry_price * 1.003
                            
                            confidence = score  # Use probability score as confidence
                            
                            try:
                                if entry_price > 0:
                                    # CPL GATE - Check alignment before execution
                                    cpl_ok, cpl_reason = cpl_check(ticker, direction)
                                    if not cpl_ok:
                                        cpl_block(ticker, direction, cpl_reason)
                                    else:
                                        log.info(f"✅ CPL GATE PASSED: {ticker} - {cpl_reason}")
                                        # Only execute if CPL passed
                                        alpaca_position = alpaca_executor.execute_scalp_entry(
                                            underlying=ticker,
                                            direction=direction,
                                            entry_price=entry_price,
                                            target_price=target,
                                            stop_loss=stop,
                                            confidence=confidence,
                                            pattern=f"orchestrator_{family_name}"
                                        )
                                        if alpaca_position:
                                            log.info(f"🚀 ALPACA EXECUTED: {ticker} {direction} via orchestrator")
                                            results["paper_trades"] += 1
                                        else:
                                            log.warning(f"Alpaca trade not placed for {ticker}")
                            except Exception as e:
                                log.error(f"Alpaca execution error for {ticker}: {e}")
                            
                            # Legacy paper trader (keep for compatibility)
                            position = paper_trader.evaluate_signal(prob)
                            if position:
                                log.debug(f"Legacy paper trader also opened {ticker}")
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
            
            # Stage 4.5: MANDATORY 0DTE EOD CLOSE
            # All 0DTE positions MUST close before market close (3:55 PM ET)
            if alpaca_executor.should_close_for_eod():
                log.info("⏰ Stage 4.5: MANDATORY EOD CLOSE - All 0DTE positions must close!")
                eod_closed = alpaca_executor.close_all_0dte_positions()
                log.info(f"   EOD positions closed: {eod_closed}")
            
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
        ticker = prob.get("ticker", "")
        if not ticker:
            return
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
            
            # EXECUTE TRADE ON ALPACA - Signal-execution connection!
            if score >= 70 and entry > 0 and stop > 0:
                try:
                    # CPL GATE - Check alignment before execution
                    cpl_ok, cpl_reason = cpl_check(ticker, direction)
                    if not cpl_ok:
                        cpl_block(ticker, direction, cpl_reason)
                    else:
                        log.info(f"✅ CPL GATE PASSED: {ticker} - {cpl_reason}")
                        # Only execute if CPL passed
                        position = alpaca_executor.execute_scalp_entry(
                            underlying=ticker,
                            direction=direction,
                            entry_price=entry,
                            stop_loss=stop,
                            target_price=t1 if t1 > 0 else entry * 1.15,
                            confidence=score,
                            pattern="orchestrator_signal"
                        )
                        if position:
                            symbol = getattr(position, 'option_symbol', ticker) if hasattr(position, 'option_symbol') else ticker
                            log.info(f"🚀 TRADE EXECUTED for {ticker}: {symbol}")
                            send_telegram_alert(f"🚀 **TRADE EXECUTED** {ticker} {direction.upper()} @ ${entry:.2f}")
                        else:
                            log.warning(f"Trade not executed for {ticker} - check Alpaca logs")
                except Exception as trade_err:
                    log.error(f"Failed to execute trade for {ticker}: {trade_err}")
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
• Options Pressure: {inception_state.options_pressure:.2f}

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


    def run_historical_training(self, weeks: int = 6) -> Dict[str, Any]:
        """
        Run historical training on startup.
        
        Downloads historical data, analyzes past events,
        and updates Learning Memory weights.
        
        Args:
            weeks: Number of weeks of history to analyze
            
        Returns:
            Training summary
        """
        log.info("=" * 50)
        log.info("RUNNING STARTUP TRAINING")
        log.info("=" * 50)
        
        try:
            summary = historical_trainer.run_full_training(weeks=weeks)
            
            if summary.get("upcoming_events", 0) > 0:
                log.info(f"Upcoming events loaded: {summary['upcoming_events']}")
                
                for event in historical_trainer.upcoming_events[:5]:
                    log.info(f"  {event.date}: {event.event_type.upper()} - {event.symbol or 'Market'}")
            
            learning_summary = learning_memory.get_learning_summary()
            log.info(f"Learning Memory updated:")
            log.info(f"  Best feature: {learning_summary.get('best_feature', 'N/A')}")
            log.info(f"  Weights adjusted: {summary.get('weight_updates', 0)}")
            
            training_msg = f"""
*[TRAINING COMPLETE]*

Analyzed {summary.get('weeks_analyzed', 0)} weeks of history
━━━━━━━━━━━━━━━━━━

*Events Analyzed:* {summary.get('total_events', 0)}
*Earnings Events:* {summary.get('earnings_events', 0)}
*Macro Events:* {summary.get('macro_events', 0)}
*Weights Updated:* {summary.get('weight_updates', 0)}

*Upcoming Events:* {summary.get('upcoming_events', 0)}

_Engine calibrated from historical data_
"""
            send_telegram_alert(training_msg)
            
            return summary
            
        except Exception as e:
            log.error(f"Training error: {e}")
            return {"error": str(e)}
    
    def get_upcoming_events_summary(self) -> str:
        """Get formatted summary of upcoming events."""
        events = historical_trainer.upcoming_events[:10]
        
        if not events:
            return "No upcoming events loaded"
        
        lines = ["*[UPCOMING EVENTS]*", ""]
        
        for event in events:
            symbol_str = event.symbol if event.symbol else "Market"
            lines.append(f"*{event.date}* - {event.event_type.upper()}: {symbol_str}")
            lines.append(f"  Expected Move: {event.expected_move:.1f}%")
        
        return "\n".join(lines)


# Global instance
orchestrator = SnakeOrchestrator()


def run_pipeline() -> Dict:
    """Run the full pipeline. Convenience function for scheduler."""
    return orchestrator.run_full_pipeline()


def run_startup_training(weeks: int = 6) -> Dict:
    """Run historical training on startup."""
    return orchestrator.run_historical_training(weeks=weeks)


def send_daily_summary() -> None:
    """Send daily summary. Called at end of trading day."""
    orchestrator.send_daily_report()
