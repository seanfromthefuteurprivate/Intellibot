"""
VENOM DATA INGESTION PIPELINE
═══════════════════════════════════════════════════════════════════

Automatically ingests trade data from:
1. Alpaca API (order history)
2. Google Drive (screenshots)
3. SQLite database (existing trades)

Feeds data into:
- Semantic Memory (TradeOutcome objects)
- Trade Graph (TradeNode objects)

Runs continuously to capture all new trades.
"""

import os
import hashlib
import requests
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

log = get_logger(__name__)


@dataclass
class ParsedTrade:
    """A parsed trade from Alpaca or other source."""
    ticker: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl_dollars: float
    pnl_percent: float
    contracts: int
    pattern: str
    option_type: str  # "CALL" or "PUT"
    strike: float
    expiry: str
    source: str  # "alpaca", "screenshot", "database"


class AlpacaTradeIngestor:
    """
    Ingests trade history from Alpaca API and feeds into learning systems.
    """

    def __init__(self):
        self.api_key = os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self._processed_orders = set()  # Track processed order IDs
        self._load_processed()

    def _load_processed(self):
        """Load already-processed order IDs from database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT order_id FROM ingested_orders")
            rows = cursor.fetchall()
            self._processed_orders = set(row[0] for row in rows)
            conn.close()
            log.info(f"DATA_INGESTOR: Loaded {len(self._processed_orders)} processed order IDs")
        except Exception as e:
            log.debug(f"Could not load processed orders: {e}")

    def _mark_processed(self, order_id: str):
        """Mark an order as processed."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO ingested_orders (order_id, ingested_at)
                VALUES (?, ?)
            """, (order_id, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            self._processed_orders.add(order_id)
        except Exception as e:
            log.debug(f"Could not mark order processed: {e}")

    def fetch_all_orders(self, days_back: int = 90) -> List[Dict]:
        """Fetch all filled orders from Alpaca."""
        if not self.api_key or not self.secret_key:
            log.warning("DATA_INGESTOR: Alpaca credentials not configured")
            return []

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
        }

        all_orders = []
        after_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%dT00:00:00Z')

        url = f"{self.base_url}/v2/orders"
        params = {
            'status': 'filled',
            'limit': 500,
            'after': after_date,
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                all_orders = resp.json()
                log.info(f"DATA_INGESTOR: Fetched {len(all_orders)} filled orders from Alpaca")
            else:
                log.warning(f"DATA_INGESTOR: Alpaca API error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            log.error(f"DATA_INGESTOR: Failed to fetch Alpaca orders: {e}")

        return all_orders

    def parse_option_symbol(self, symbol: str) -> Tuple[str, str, float, str]:
        """
        Parse OCC option symbol like SPY260226C00690000.
        Returns (underlying, expiry, strike, option_type)
        """
        try:
            # Format: UNDERLYING + YYMMDD + C/P + STRIKE(8 digits)
            # Find where digits start for expiry
            i = 0
            while i < len(symbol) and not symbol[i].isdigit():
                i += 1

            underlying = symbol[:i]
            rest = symbol[i:]

            # Expiry is 6 digits
            expiry = rest[:6]
            option_type = "CALL" if rest[6] == "C" else "PUT"
            strike = float(rest[7:]) / 1000  # Strike is in 1/1000ths

            expiry_formatted = f"20{expiry[:2]}-{expiry[2:4]}-{expiry[4:6]}"

            return underlying, expiry_formatted, strike, option_type
        except Exception as e:
            log.debug(f"Could not parse option symbol {symbol}: {e}")
            return symbol, "", 0.0, "UNKNOWN"

    def match_trades(self, orders: List[Dict]) -> List[ParsedTrade]:
        """
        Match buy/sell orders to create complete trades with P&L.
        """
        # Group orders by symbol
        by_symbol = {}
        for order in orders:
            symbol = order.get("symbol", "")
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(order)

        trades = []

        for symbol, symbol_orders in by_symbol.items():
            # Sort by time
            symbol_orders.sort(key=lambda x: x.get("filled_at", ""))

            # Match buys to sells
            pending_buys = []
            for order in symbol_orders:
                side = order.get("side", "")
                qty = int(order.get("filled_qty", 0))
                price = float(order.get("filled_avg_price", 0))
                filled_at = order.get("filled_at", "")
                order_id = order.get("id", "")

                if side == "buy":
                    pending_buys.append({
                        "qty": qty,
                        "price": price,
                        "time": filled_at,
                        "order_id": order_id,
                    })
                elif side == "sell" and pending_buys:
                    # Match with oldest buy
                    buy = pending_buys.pop(0)

                    # Skip if already processed
                    if order_id in self._processed_orders or buy["order_id"] in self._processed_orders:
                        continue

                    # Calculate P&L
                    entry_price = buy["price"]
                    exit_price = price
                    contracts = min(buy["qty"], qty)
                    pnl_dollars = (exit_price - entry_price) * contracts * 100
                    pnl_percent = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                    # Parse option symbol
                    underlying, expiry, strike, option_type = self.parse_option_symbol(symbol)

                    # Determine direction
                    direction = "LONG" if option_type == "CALL" else "SHORT"

                    # Parse times
                    try:
                        entry_time = datetime.fromisoformat(buy["time"].replace("Z", "+00:00"))
                        exit_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
                    except:
                        entry_time = datetime.now(timezone.utc)
                        exit_time = datetime.now(timezone.utc)

                    trade = ParsedTrade(
                        ticker=underlying,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=entry_time,
                        exit_time=exit_time,
                        pnl_dollars=pnl_dollars,
                        pnl_percent=pnl_percent,
                        contracts=contracts,
                        pattern=f"ALPACA_{option_type}_{underlying}",
                        option_type=option_type,
                        strike=strike,
                        expiry=expiry,
                        source="alpaca",
                    )
                    trades.append(trade)

                    # Mark as processed
                    self._mark_processed(order_id)
                    self._mark_processed(buy["order_id"])

        log.info(f"DATA_INGESTOR: Matched {len(trades)} complete trades")
        return trades


class ScreenshotIngestor:
    """
    Ingests trade data from Google Drive screenshots via the screenshot learning system.
    Bridges learned_trades table into Semantic Memory and Trade Graph.
    """

    def __init__(self):
        self._processed_screenshot_ids = set()
        self._load_processed()

    def _load_processed(self):
        """Load already-processed screenshot IDs."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT screenshot_id FROM ingested_screenshots")
            rows = cursor.fetchall()
            self._processed_screenshot_ids = set(row[0] for row in rows)
            conn.close()
            log.info(f"SCREENSHOT_INGESTOR: Loaded {len(self._processed_screenshot_ids)} processed screenshot IDs")
        except Exception as e:
            log.debug(f"Could not load processed screenshots: {e}")

    def _mark_processed(self, screenshot_id: int):
        """Mark a screenshot as processed."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO ingested_screenshots (screenshot_id, ingested_at)
                VALUES (?, ?)
            """, (screenshot_id, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            self._processed_screenshot_ids.add(screenshot_id)
        except Exception as e:
            log.debug(f"Could not mark screenshot processed: {e}")

    def trigger_screenshot_sync(self) -> int:
        """
        Trigger Google Drive sync and return count of new screenshots processed.
        """
        try:
            from wsb_snake.collectors.screenshot_system import screenshot_system

            # Process any new screenshots from Google Drive
            results = screenshot_system.process_new_screenshots()
            new_count = len([r for r in results if r.get("status") == "success"])
            log.info(f"SCREENSHOT_INGESTOR: Synced {new_count} new screenshots from Google Drive")
            return new_count
        except Exception as e:
            log.warning(f"SCREENSHOT_INGESTOR: Google Drive sync failed (may not be configured): {e}")
            return 0

    def fetch_learned_trades(self) -> List[ParsedTrade]:
        """
        Fetch all learned trades from the screenshot system that haven't been ingested yet.
        """
        trades = []

        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get all learned trades that haven't been ingested to semantic/graph
            cursor.execute("""
                SELECT
                    id, screenshot_id, ticker, trade_type, strike, expiry,
                    entry_price, exit_price, contracts, shares,
                    capital_deployed, profit_loss, profit_loss_pct,
                    platform, trade_date, entry_time, exit_time,
                    holding_period_minutes, detected_pattern, setup_description
                FROM learned_trades
                WHERE id NOT IN (SELECT screenshot_id FROM ingested_screenshots)
            """)

            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                try:
                    # Parse trade type to direction and option_type
                    trade_type = row["trade_type"] or "CALLS"
                    option_type = "CALL" if "CALL" in trade_type.upper() else "PUT"
                    direction = "LONG" if option_type == "CALL" else "SHORT"

                    # Parse times
                    entry_time_str = row["entry_time"] or row["trade_date"] or datetime.now().isoformat()
                    exit_time_str = row["exit_time"] or entry_time_str

                    try:
                        entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    except:
                        entry_time = datetime.now(timezone.utc)

                    try:
                        exit_time = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
                    except:
                        exit_time = entry_time + timedelta(minutes=row["holding_period_minutes"] or 30)

                    # Calculate P&L if not provided
                    pnl_dollars = row["profit_loss"] or 0.0
                    pnl_percent = row["profit_loss_pct"] or 0.0

                    # Build pattern name
                    pattern = row["detected_pattern"] or f"SCREENSHOT_{option_type}_{row['ticker']}"

                    trade = ParsedTrade(
                        ticker=row["ticker"] or "SPY",
                        direction=direction,
                        entry_price=row["entry_price"] or 1.0,
                        exit_price=row["exit_price"] or row["entry_price"] or 1.0,
                        entry_time=entry_time,
                        exit_time=exit_time,
                        pnl_dollars=pnl_dollars,
                        pnl_percent=pnl_percent,
                        contracts=row["contracts"] or row["shares"] or 1,
                        pattern=pattern,
                        option_type=option_type,
                        strike=row["strike"] or 0.0,
                        expiry=row["expiry"] or "",
                        source="screenshot",
                    )
                    trades.append(trade)

                    # Mark as processed (using the learned_trade id, not screenshot_id)
                    self._mark_processed(row["id"])

                except Exception as e:
                    log.debug(f"Could not parse learned trade {row['id']}: {e}")

            log.info(f"SCREENSHOT_INGESTOR: Found {len(trades)} unprocessed learned trades")

        except Exception as e:
            log.warning(f"SCREENSHOT_INGESTOR: Could not fetch learned trades: {e}")

        return trades


class DataIngestor:
    """
    Master data ingestion pipeline.
    Feeds all trade data into Semantic Memory and Trade Graph.
    """

    def __init__(self):
        self.alpaca_ingestor = AlpacaTradeIngestor()
        self.screenshot_ingestor = ScreenshotIngestor()
        self._running = False
        self._thread = None
        self._ensure_tables()

    def _ensure_tables(self):
        """Create ingestion tracking tables."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingested_orders (
                    order_id TEXT PRIMARY KEY,
                    ingested_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingested_screenshots (
                    screenshot_id INTEGER PRIMARY KEY,
                    ingested_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    source TEXT,
                    trades_ingested INTEGER,
                    semantic_records INTEGER,
                    graph_records INTEGER
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug(f"Table creation skipped: {e}")

    def ingest_to_semantic_memory(self, trade: ParsedTrade) -> bool:
        """Ingest a trade into Semantic Memory."""
        try:
            from wsb_snake.learning.semantic_memory import (
                get_semantic_memory,
                TradeOutcome,
                TradeConditions,
            )

            semantic = get_semantic_memory()

            # Build conditions
            conditions = TradeConditions(
                ticker=trade.ticker,
                direction=trade.direction,
                entry_price=trade.entry_price,
                rsi=50.0,  # Default - we don't have this from Alpaca
                adx=25.0,
                atr=1.0,
                macd_signal="neutral",
                volume_ratio=1.0,
                regime="unknown",
                vix=20.0,
                gex_regime="unknown",
                hydra_direction="NEUTRAL",
                confluence_score=0.7,
                stop_distance_pct=10.0,
                target_distance_pct=15.0,
            )

            # Build outcome
            trade_id = hashlib.md5(
                f"{trade.ticker}_{trade.entry_time}_{trade.exit_time}".encode()
            ).hexdigest()[:12]

            duration = int((trade.exit_time - trade.entry_time).total_seconds() / 60)

            outcome = TradeOutcome(
                trade_id=trade_id,
                conditions=conditions,
                entry_reasoning=f"Ingested from {trade.source}: {trade.pattern}",
                pnl_dollars=trade.pnl_dollars,
                pnl_percent=trade.pnl_percent,
                duration_minutes=duration,
                max_adverse_excursion_pct=min(0, trade.pnl_percent),
                max_favorable_excursion_pct=max(0, trade.pnl_percent),
                exit_reason="TARGET" if trade.pnl_dollars > 0 else "STOP",
                exit_price=trade.exit_price,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                lessons_learned=f"{'WIN' if trade.pnl_dollars > 0 else 'LOSS'} {trade.pnl_percent:+.1f}%",
            )

            semantic.record_trade(outcome)
            return True

        except Exception as e:
            log.debug(f"Semantic memory ingestion failed: {e}")
            return False

    def ingest_to_trade_graph(self, trade: ParsedTrade) -> bool:
        """Ingest a trade into Trade Graph."""
        try:
            from wsb_snake.learning.trade_graph import get_trade_graph

            trade_graph = get_trade_graph()

            # Build conditions for the graph
            conditions = {
                "ticker": trade.ticker,
                "pattern": trade.pattern,
                "direction": trade.direction.lower(),
                "option_type": trade.option_type,
                "strike": trade.strike,
                "expiry": trade.expiry,
                "regime": "unknown",
                "gex_regime": "unknown",
                "hydra_direction": "NEUTRAL",
                "flow_bias": "NEUTRAL",
                "volume_ratio": 1.0,
            }

            trade_graph.record_trade(
                ticker=trade.ticker,
                direction=trade.direction.lower(),
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                pnl_dollars=trade.pnl_dollars,
                pattern=trade.pattern,
                entry_reasoning=f"Ingested from {trade.source}",
                exit_reasoning="TARGET" if trade.pnl_dollars > 0 else "STOP",
                conditions=conditions,
                metadata={
                    "source": trade.source,
                    "option_type": trade.option_type,
                    "strike": trade.strike,
                    "contracts": trade.contracts,
                },
            )
            return True

        except Exception as e:
            log.debug(f"Trade graph ingestion failed: {e}")
            return False

    def run_full_ingestion(self) -> Dict:
        """
        Run complete ingestion from all sources.
        Returns stats on what was ingested.
        """
        stats = {
            "alpaca_trades": 0,
            "screenshot_trades": 0,
            "screenshot_synced": 0,
            "semantic_records": 0,
            "graph_records": 0,
            "high_conviction_tagged": 0,
        }

        log.info("=" * 60)
        log.info("DATA_INGESTOR: Running full ingestion pipeline")
        log.info("=" * 60)

        # 1. Fetch Alpaca trades
        orders = self.alpaca_ingestor.fetch_all_orders(days_back=90)
        alpaca_trades = self.alpaca_ingestor.match_trades(orders)
        stats["alpaca_trades"] = len(alpaca_trades)

        # 2. Sync Google Drive screenshots and fetch learned trades
        stats["screenshot_synced"] = self.screenshot_ingestor.trigger_screenshot_sync()
        screenshot_trades = self.screenshot_ingestor.fetch_learned_trades()
        stats["screenshot_trades"] = len(screenshot_trades)

        # 3. Combine all trades
        all_trades = alpaca_trades + screenshot_trades
        log.info(f"DATA_INGESTOR: Total trades to process: {len(all_trades)} (Alpaca: {len(alpaca_trades)}, Screenshots: {len(screenshot_trades)})")

        # 4. Ingest each trade
        for trade in all_trades:
            if self.ingest_to_semantic_memory(trade):
                stats["semantic_records"] += 1

            if self.ingest_to_trade_graph(trade):
                stats["graph_records"] += 1

            # Tag high conviction trades (100%+ gains)
            if trade.pnl_percent >= 100:
                stats["high_conviction_tagged"] += 1
                log.info(
                    f"HIGH CONVICTION: {trade.ticker} {trade.option_type} "
                    f"+{trade.pnl_percent:.0f}% | Pattern: {trade.pattern} | Source: {trade.source}"
                )

        # 5. Record stats
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ingestion_stats (timestamp, source, trades_ingested, semantic_records, graph_records)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                "full_ingestion",
                stats["alpaca_trades"] + stats["screenshot_trades"],
                stats["semantic_records"],
                stats["graph_records"],
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug(f"Stats recording failed: {e}")

        log.info(f"DATA_INGESTOR: Complete - {stats}")
        return stats

    def start_continuous_ingestion(self, interval_minutes: int = 15):
        """Start continuous background ingestion."""
        if self._running:
            log.warning("DATA_INGESTOR: Already running")
            return

        self._running = True

        def _ingest_loop():
            while self._running:
                try:
                    self.run_full_ingestion()
                except Exception as e:
                    log.error(f"DATA_INGESTOR: Ingestion error - {e}")

                # Wait for next interval
                time.sleep(interval_minutes * 60)

        self._thread = threading.Thread(target=_ingest_loop, daemon=True)
        self._thread.start()
        log.info(f"DATA_INGESTOR: Started continuous ingestion (every {interval_minutes} min)")

    def stop_continuous_ingestion(self):
        """Stop continuous ingestion."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("DATA_INGESTOR: Stopped")


# Singleton instance
_data_ingestor: Optional[DataIngestor] = None


def get_data_ingestor() -> DataIngestor:
    """Get singleton DataIngestor instance."""
    global _data_ingestor
    if _data_ingestor is None:
        _data_ingestor = DataIngestor()
    return _data_ingestor


def run_initial_ingestion() -> Dict:
    """Run initial data ingestion - call at startup."""
    ingestor = get_data_ingestor()
    return ingestor.run_full_ingestion()


def start_auto_ingestion(interval_minutes: int = 15):
    """Start automatic continuous ingestion."""
    ingestor = get_data_ingestor()
    ingestor.start_continuous_ingestion(interval_minutes)
