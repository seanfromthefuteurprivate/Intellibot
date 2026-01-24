"""
Multi-Day Opportunity Scanner

Scans for setups beyond 0DTE including:
- Earnings plays (7-10 DTE)
- Post-earnings continuation (3-10 DTE)
- News/catalyst windows (7-21 DTE)
- Macro reaction trades (3-10 DTE)

Runs less frequently than 0DTE scanner (every 4 hours vs every 30 seconds).
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from wsb_snake.utils.logger import log
from wsb_snake.config import ZERO_DTE_UNIVERSE


@dataclass
class MultiDaySetup:
    """A multi-day trading opportunity"""
    symbol: str
    setup_type: str
    direction: str
    dte_target: int
    entry_reason: str
    confidence: float
    risk_reward: str
    key_levels: Dict
    catalyst: str
    timestamp: str


class MultiDayScanner:
    """
    Scans for multi-day options opportunities.
    
    Unlike 0DTE which requires minute-by-minute monitoring,
    these setups can be identified and planned in advance.
    """
    
    EXPANDED_UNIVERSE = ZERO_DTE_UNIVERSE + [
        "NFLX", "CRM", "COST", "JPM", "V", "MA", 
        "UNH", "PFE", "MRNA", "XOM", "CVX"
    ]
    
    def __init__(self):
        self.active_setups: List[MultiDaySetup] = []
        self.last_scan = None
        log.info("Multi-Day Scanner initialized")
    
    def scan_earnings_plays(self) -> List[MultiDaySetup]:
        """
        Scan for earnings-related setups.
        
        Types:
        - Pre-earnings directional (7-10 DTE)
        - Pre-earnings volatility (straddles)
        - Post-earnings continuation
        """
        setups = []
        
        try:
            from wsb_snake.collectors.earnings_calendar import earnings_calendar
            from wsb_snake.collectors.polygon_options import polygon_options
            
            for symbol in self.EXPANDED_UNIVERSE:
                try:
                    signal = earnings_calendar.get_earnings_signal(symbol)
                    
                    if not signal.get("has_earnings"):
                        continue
                    
                    days_until = signal.get("days_until")
                    if days_until is None:
                        continue
                    
                    if 5 <= days_until <= 14:
                        pattern_bias = signal.get("pattern_bias", "neutral")
                        
                        if pattern_bias in ["bullish", "slight_bullish"]:
                            direction = "CALLS"
                            confidence = 65 if pattern_bias == "bullish" else 55
                        elif pattern_bias in ["bearish"]:
                            direction = "PUTS"
                            confidence = 60
                        else:
                            direction = "STRADDLE"
                            confidence = 50
                        
                        setups.append(MultiDaySetup(
                            symbol=symbol,
                            setup_type="EARNINGS_DIRECTIONAL" if direction != "STRADDLE" else "EARNINGS_VOLATILITY",
                            direction=direction,
                            dte_target=min(days_until + 3, 14),
                            entry_reason=f"Earnings on {signal.get('earnings_date')}",
                            confidence=confidence,
                            risk_reward="1:3 to 1:5",
                            key_levels={"earnings_date": signal.get("earnings_date")},
                            catalyst="EARNINGS",
                            timestamp=datetime.now().isoformat(),
                        ))
                    
                    elif days_until < 0 and days_until >= -5:
                        setups.append(MultiDaySetup(
                            symbol=symbol,
                            setup_type="POST_EARNINGS_CONTINUATION",
                            direction="TREND",
                            dte_target=7,
                            entry_reason=f"Post-earnings drift potential",
                            confidence=55,
                            risk_reward="1:2",
                            key_levels={},
                            catalyst="POST_EARNINGS",
                            timestamp=datetime.now().isoformat(),
                        ))
                        
                except Exception as e:
                    log.debug(f"Error scanning {symbol} for earnings: {e}")
                    continue
            
        except Exception as e:
            log.warning(f"Earnings scan error: {e}")
        
        return setups
    
    def scan_macro_setups(self) -> List[MultiDaySetup]:
        """
        Scan for macro reaction opportunities.
        
        Focus on SPY/QQQ/IWM after major macro events.
        """
        setups = []
        
        try:
            from wsb_snake.collectors.fred_collector import fred_collector
            from wsb_snake.collectors.vix_structure import vix_structure
            
            macro = fred_collector.get_macro_regime()
            vix_signal = vix_structure.get_trading_signal()
            
            for symbol in ["SPY", "QQQ", "IWM"]:
                options_bias = macro.get("options_bias", "neutral")
                vix_regime = vix_signal.get("structure", "normal")
                
                if options_bias == "calls_favored" and vix_regime != "backwardation":
                    setups.append(MultiDaySetup(
                        symbol=symbol,
                        setup_type="MACRO_REACTION",
                        direction="CALLS",
                        dte_target=5,
                        entry_reason=f"Macro regime: {macro.get('overall_regime')}",
                        confidence=55,
                        risk_reward="1:2",
                        key_levels={
                            "fed_funds": macro.get("metrics", {}).get("fed_funds"),
                            "vix": vix_signal.get("vix"),
                        },
                        catalyst="MACRO_ENVIRONMENT",
                        timestamp=datetime.now().isoformat(),
                    ))
                
                elif options_bias == "puts_favored" or vix_regime == "backwardation":
                    setups.append(MultiDaySetup(
                        symbol=symbol,
                        setup_type="MACRO_REACTION",
                        direction="PUTS",
                        dte_target=5,
                        entry_reason=f"Risk-off regime, VIX: {vix_regime}",
                        confidence=60,
                        risk_reward="1:2",
                        key_levels={
                            "vix": vix_signal.get("vix"),
                            "fear_level": vix_signal.get("fear_level"),
                        },
                        catalyst="RISK_OFF",
                        timestamp=datetime.now().isoformat(),
                    ))
                    
        except Exception as e:
            log.warning(f"Macro scan error: {e}")
        
        return setups
    
    def scan_congressional_signals(self) -> List[MultiDaySetup]:
        """
        Scan for congressional trading signals.
        
        Politicians often trade ahead of legislation.
        """
        setups = []
        
        try:
            from wsb_snake.collectors.congressional_trading import congressional_trading
            
            hot_tickers = congressional_trading.get_hot_tickers(days_back=14)
            
            for ticker_data in hot_tickers[:10]:
                ticker = ticker_data.get("ticker", "")
                if ticker not in self.EXPANDED_UNIVERSE:
                    continue
                
                net_sentiment = ticker_data.get("net_sentiment", 0)
                total_trades = ticker_data.get("total_trades", 0)
                
                if total_trades >= 3 and abs(net_sentiment) >= 2:
                    direction = "CALLS" if net_sentiment > 0 else "PUTS"
                    
                    setups.append(MultiDaySetup(
                        symbol=ticker,
                        setup_type="CONGRESSIONAL_SIGNAL",
                        direction=direction,
                        dte_target=14,
                        entry_reason=f"Congress: {ticker_data.get('buys')} buys, {ticker_data.get('sells')} sells",
                        confidence=60,
                        risk_reward="1:3",
                        key_levels={
                            "buys": ticker_data.get("buys"),
                            "sells": ticker_data.get("sells"),
                        },
                        catalyst="CONGRESSIONAL_TRADING",
                        timestamp=datetime.now().isoformat(),
                    ))
                    
        except Exception as e:
            log.warning(f"Congressional scan error: {e}")
        
        return setups
    
    def run_full_scan(self) -> Dict[str, Any]:
        """
        Run complete multi-day opportunity scan.
        
        Returns:
            Dict with all identified setups
        """
        log.info("Running multi-day opportunity scan...")
        
        all_setups = []
        
        earnings_setups = self.scan_earnings_plays()
        all_setups.extend(earnings_setups)
        log.info(f"  Earnings setups: {len(earnings_setups)}")
        
        macro_setups = self.scan_macro_setups()
        all_setups.extend(macro_setups)
        log.info(f"  Macro setups: {len(macro_setups)}")
        
        congress_setups = self.scan_congressional_signals()
        all_setups.extend(congress_setups)
        log.info(f"  Congressional setups: {len(congress_setups)}")
        
        self.active_setups = all_setups
        self.last_scan = datetime.now()
        
        by_type = {}
        for setup in all_setups:
            if setup.setup_type not in by_type:
                by_type[setup.setup_type] = []
            by_type[setup.setup_type].append({
                "symbol": setup.symbol,
                "direction": setup.direction,
                "dte": setup.dte_target,
                "confidence": setup.confidence,
                "catalyst": setup.catalyst,
            })
        
        return {
            "total_setups": len(all_setups),
            "by_type": by_type,
            "top_setups": sorted(all_setups, key=lambda x: x.confidence, reverse=True)[:5],
            "scan_time": self.last_scan.isoformat(),
        }
    
    def get_telegram_summary(self) -> str:
        """Generate Telegram-ready summary of multi-day setups"""
        if not self.active_setups:
            return ""
        
        top_setups = sorted(self.active_setups, key=lambda x: x.confidence, reverse=True)[:5]
        
        lines = ["📊 *MULTI-DAY OPPORTUNITIES*\n"]
        
        for setup in top_setups:
            emoji = "📈" if setup.direction == "CALLS" else "📉" if setup.direction == "PUTS" else "↔️"
            lines.append(
                f"{emoji} *{setup.symbol}* | {setup.setup_type}\n"
                f"   Direction: {setup.direction} | DTE: {setup.dte_target}\n"
                f"   Confidence: {setup.confidence}% | R:R {setup.risk_reward}\n"
                f"   Catalyst: {setup.catalyst}\n"
            )
        
        return "\n".join(lines)


multi_day_scanner = MultiDayScanner()
