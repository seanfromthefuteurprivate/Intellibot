"""
Engine 6: Paper Shadow Trader

Simulates trades based on high-conviction signals without real money.
Tracks P&L, win rate, and generates daily reports.

Features:
- Paper execution of A+ and A tier signals
- Position management (entry, stop, targets)
- Daily P&L tracking
- Performance reports via Telegram
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from wsb_snake.db.database import get_connection, save_paper_trade
from wsb_snake.utils.logger import log
from wsb_snake.engines.learning_memory import learning_memory


@dataclass
class PaperPosition:
    """An active paper position."""
    trade_id: int
    ticker: str
    direction: str  # "long" or "short"
    entry_price: float
    stop_price: float
    target_1_price: float
    target_2_price: float
    position_size: int
    status: str  # "pending", "open", "closed"
    
    # Tracking
    entry_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    r_multiple: float = 0.0


class PaperTrader:
    """
    Engine 6: Paper Shadow Trader
    
    Simulates trades on high-conviction signals.
    """
    
    # Configuration
    PAPER_ACCOUNT_SIZE = 100_000  # $100k paper account
    MAX_POSITION_SIZE = 5000      # Max $5k per position
    DEFAULT_SHARES = 100          # Default share size
    MIN_SCORE_TO_TRADE = 70       # Only trade A tier and above
    
    def __init__(self):
        self.active_positions: Dict[str, PaperPosition] = {}
        self._load_open_positions()
    
    def _load_open_positions(self):
        """Load any open positions from database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_trades
            WHERE status = 'open'
        """)
        
        for row in cursor.fetchall():
            pos = PaperPosition(
                trade_id=row["id"],
                ticker=row["ticker"],
                direction=row["direction"],
                entry_price=row["fill_price"] or row["entry_price"],
                stop_price=row["stop_price"],
                target_1_price=row["target_1_price"],
                target_2_price=row["target_2_price"],
                position_size=row["position_size"],
                status="open",
            )
            self.active_positions[row["ticker"]] = pos
        
        conn.close()
        log.info(f"Loaded {len(self.active_positions)} open paper positions")
    
    def evaluate_signal(self, probability_output: Dict) -> Optional[PaperPosition]:
        """
        Evaluate a probability output and potentially take a paper trade.
        
        Args:
            probability_output: Output from probability generator
            
        Returns:
            PaperPosition if trade taken, None otherwise
        """
        ticker = probability_output.get("ticker")
        score = probability_output.get("combined_score", 0)
        action = probability_output.get("action", "AVOID")
        
        # Only trade high-conviction signals
        if score < self.MIN_SCORE_TO_TRADE:
            return None
        
        if action not in ["STRONG_LONG", "LONG", "STRONG_SHORT", "SHORT"]:
            return None
        
        # Don't double up on same ticker
        if ticker in self.active_positions:
            log.debug(f"Already have position in {ticker}")
            return None
        
        # Determine direction
        direction = "long" if "LONG" in action else "short"
        
        # Get trade levels
        entry_price = probability_output.get("entry_price", 0)
        stop_price = probability_output.get("stop_loss", 0)
        target_1 = probability_output.get("target_1", 0)
        target_2 = probability_output.get("target_2", 0)
        
        if entry_price <= 0 or stop_price <= 0:
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_pct=probability_output.get("max_position_pct", 2.0),
        )
        
        # Create paper trade in database
        trade_data = {
            "signal_id": probability_output.get("signal_id", 0),
            "ticker": ticker,
            "direction": direction,
            "entry_trigger": f"Score {score:.0f} - {action}",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_1_price": target_1,
            "target_2_price": target_2,
            "position_size": position_size,
        }
        
        trade_id = save_paper_trade(trade_data)
        
        # Create position object
        position = PaperPosition(
            trade_id=trade_id,
            ticker=ticker,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_1_price=target_1,
            target_2_price=target_2,
            position_size=position_size,
            status="pending",
        )
        
        self.active_positions[ticker] = position
        
        log.info(f"Paper trade created: {direction.upper()} {ticker} @ ${entry_price:.2f}")
        
        return position
    
    def fill_pending_orders(self, current_prices: Dict[str, float]) -> List[PaperPosition]:
        """
        Fill any pending orders that have reached entry price.
        
        Args:
            current_prices: Dict of ticker -> current price
            
        Returns:
            List of newly filled positions
        """
        filled = []
        
        for ticker, position in list(self.active_positions.items()):
            if position.status != "pending":
                continue
            
            current_price = current_prices.get(ticker)
            if not current_price:
                continue
            
            # Check if entry triggered
            # For simplicity, fill at market if within 0.5% of entry
            entry_diff = abs(current_price - position.entry_price) / position.entry_price
            
            if entry_diff <= 0.005:
                position.status = "open"
                position.entry_time = datetime.utcnow()
                
                # Update database
                self._update_trade_status(
                    position.trade_id,
                    status="open",
                    fill_price=current_price,
                    fill_time=position.entry_time,
                )
                
                filled.append(position)
                log.info(f"Paper order filled: {ticker} @ ${current_price:.2f}")
        
        return filled
    
    def check_exits(self, current_prices: Dict[str, float]) -> List[PaperPosition]:
        """
        Check open positions for stop/target hits.
        
        Args:
            current_prices: Dict of ticker -> current price
            
        Returns:
            List of closed positions
        """
        closed = []
        
        for ticker, position in list(self.active_positions.items()):
            if position.status != "open":
                continue
            
            current_price = current_prices.get(ticker)
            if not current_price:
                continue
            
            exit_reason = None
            exit_price = None
            
            if position.direction == "long":
                # Check stop
                if current_price <= position.stop_price:
                    exit_reason = "stop_hit"
                    exit_price = position.stop_price
                # Check target 1
                elif current_price >= position.target_1_price:
                    exit_reason = "target_1_hit"
                    exit_price = position.target_1_price
            else:  # short
                # Check stop
                if current_price >= position.stop_price:
                    exit_reason = "stop_hit"
                    exit_price = position.stop_price
                # Check target 1
                elif current_price <= position.target_1_price:
                    exit_reason = "target_1_hit"
                    exit_price = position.target_1_price
            
            if exit_reason:
                position = self._close_position(position, exit_price, exit_reason)
                closed.append(position)
        
        return closed
    
    def _close_position(
        self,
        position: PaperPosition,
        exit_price: float,
        exit_reason: str,
    ) -> PaperPosition:
        """Close a position and calculate P&L."""
        position.exit_price = exit_price
        position.exit_time = datetime.utcnow()
        position.exit_reason = exit_reason
        position.status = "closed"
        
        # Calculate P&L
        if position.direction == "long":
            position.pnl = (exit_price - position.entry_price) * position.position_size
        else:
            position.pnl = (position.entry_price - exit_price) * position.position_size
        
        # Calculate R-multiple
        risk = abs(position.entry_price - position.stop_price)
        if risk > 0:
            actual_move = exit_price - position.entry_price if position.direction == "long" else position.entry_price - exit_price
            position.r_multiple = actual_move / risk
        
        # Update database
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE paper_trades
            SET status = 'closed', exit_price = ?, exit_time = ?,
                exit_reason = ?, pnl = ?, r_multiple = ?
            WHERE id = ?
        """, (
            exit_price,
            position.exit_time.isoformat(),
            exit_reason,
            position.pnl,
            position.r_multiple,
            position.trade_id,
        ))
        
        conn.commit()
        conn.close()
        
        # Remove from active positions
        if position.ticker in self.active_positions:
            del self.active_positions[position.ticker]
        
        # Record outcome for learning
        outcome_type = "win" if position.pnl > 0 else "loss" if position.pnl < 0 else "scratch"
        
        # Get signal_id from trade
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT signal_id FROM paper_trades WHERE id = ?", (position.trade_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row["signal_id"]:
            learning_memory.record_outcome(
                signal_id=row["signal_id"],
                entry_price=position.entry_price,
                exit_price=exit_price,
                max_price=exit_price if position.pnl > 0 else position.entry_price,
                min_price=position.entry_price if position.pnl > 0 else exit_price,
                outcome_type=outcome_type,
            )
        
        log.info(
            f"Paper trade closed: {position.ticker} | "
            f"P&L: ${position.pnl:.2f} | R: {position.r_multiple:.2f}"
        )
        
        return position
    
    def _calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        max_position_pct: float,
    ) -> int:
        """Calculate position size based on risk."""
        # Risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        # Max risk amount (% of account)
        max_risk = self.PAPER_ACCOUNT_SIZE * (max_position_pct / 100) * 0.1  # 10% of position as risk
        
        # Calculate shares
        if risk_per_share > 0:
            shares = int(max_risk / risk_per_share)
        else:
            shares = self.DEFAULT_SHARES
        
        # Cap by max position size
        max_shares = int(self.MAX_POSITION_SIZE / entry_price)
        shares = min(shares, max_shares, self.DEFAULT_SHARES * 5)
        
        return max(1, shares)
    
    def _update_trade_status(
        self,
        trade_id: int,
        status: str,
        fill_price: float = None,
        fill_time: datetime = None,
    ):
        """Update trade status in database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        if fill_price and fill_time:
            cursor.execute("""
                UPDATE paper_trades
                SET status = ?, fill_price = ?, fill_time = ?
                WHERE id = ?
            """, (status, fill_price, fill_time.isoformat(), trade_id))
        else:
            cursor.execute("""
                UPDATE paper_trades SET status = ? WHERE id = ?
            """, (status, trade_id))
        
        conn.commit()
        conn.close()
    
    def get_daily_report(self, report_date: str = None) -> Dict:
        """
        Generate daily performance report.
        
        Args:
            report_date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Dict with performance metrics
        """
        if report_date is None:
            report_date = date.today().strftime("%Y-%m-%d")
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get today's closed trades
        cursor.execute("""
            SELECT * FROM paper_trades
            WHERE date(exit_time) = ? AND status = 'closed'
        """, (report_date,))
        
        trades = cursor.fetchall()
        conn.close()
        
        if not trades:
            return {
                "date": report_date,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "scratches": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_r": 0,
                "best_trade": None,
                "worst_trade": None,
            }
        
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = sum(1 for t in trades if t["pnl"] < 0)
        scratches = len(trades) - wins - losses
        
        total_pnl = sum(t["pnl"] for t in trades)
        r_multiples = [t["r_multiple"] for t in trades if t["r_multiple"]]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        
        best_trade = max(trades, key=lambda t: t["pnl"])
        worst_trade = min(trades, key=lambda t: t["pnl"])
        
        return {
            "date": report_date,
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "win_rate": wins / len(trades) if trades else 0,
            "total_pnl": total_pnl,
            "avg_r": avg_r,
            "best_trade": {
                "ticker": best_trade["ticker"],
                "pnl": best_trade["pnl"],
                "r": best_trade["r_multiple"],
            },
            "worst_trade": {
                "ticker": worst_trade["ticker"],
                "pnl": worst_trade["pnl"],
                "r": worst_trade["r_multiple"],
            },
            "trades": [
                {
                    "ticker": t["ticker"],
                    "direction": t["direction"],
                    "entry": t["fill_price"],
                    "exit": t["exit_price"],
                    "pnl": t["pnl"],
                    "r": t["r_multiple"],
                    "reason": t["exit_reason"],
                }
                for t in trades
            ],
        }
    
    def format_daily_report(self, report: Dict) -> str:
        """Format daily report for Telegram."""
        if report["total_trades"] == 0:
            return f"📊 **WSB Snake Daily Report - {report['date']}**\n\nNo paper trades today."
        
        emoji = "🟢" if report["total_pnl"] > 0 else "🔴" if report["total_pnl"] < 0 else "⚪"
        
        msg = f"""📊 **WSB Snake Daily Report**
Date: {report['date']}

{emoji} **Summary**
Trades: {report['total_trades']}
Wins: {report['wins']} | Losses: {report['losses']} | Scratches: {report['scratches']}
Win Rate: {report['win_rate']*100:.0f}%

💰 **P&L**
Total: ${report['total_pnl']:+,.2f}
Avg R: {report['avg_r']:+.2f}R

🏆 **Best Trade**
{report['best_trade']['ticker']}: ${report['best_trade']['pnl']:+,.2f} ({report['best_trade']['r']:+.2f}R)

💀 **Worst Trade**
{report['worst_trade']['ticker']}: ${report['worst_trade']['pnl']:+,.2f} ({report['worst_trade']['r']:+.2f}R)

📝 **All Trades**"""
        
        for t in report["trades"][:5]:
            emoji = "✅" if t["pnl"] > 0 else "❌" if t["pnl"] < 0 else "➖"
            msg += f"\n{emoji} {t['ticker']}: ${t['pnl']:+,.2f}"
        
        return msg


# Global instance
paper_trader = PaperTrader()
