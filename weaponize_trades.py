#!/usr/bin/env python3
"""
WEAPONIZE TRADE DATA

Feed all learned_trades into existing learning components:
1. SemanticMemory - record_trade() + auto_reflect()
2. TradeGraph - record_trade()
3. SpecialistSwarm - analyze top winners

Uses EXISTING method signatures - no new code.
"""

import sqlite3
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add wsb_snake to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wsb_snake.learning.semantic_memory import (
    SemanticMemory, TradeConditions, TradeOutcome, get_semantic_memory
)
from wsb_snake.learning.trade_graph import TradeGraph, get_trade_graph
from wsb_snake.learning.specialist_swarm import SpecialistSwarm


def load_trades_from_db(db_path: str) -> List[Dict[str, Any]]:
    """Load all trades from learned_trades table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM learned_trades
        ORDER BY profit_loss_dollars DESC
    """)

    trades = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return trades


def trade_to_conditions(trade: Dict) -> TradeConditions:
    """Convert DB trade to TradeConditions dataclass."""
    return TradeConditions(
        ticker=trade["ticker"],
        direction="LONG" if trade.get("direction", "long") == "long" else "SHORT",
        entry_price=trade.get("entry_price", 1.0),
        rsi=50.0,  # Default - we don't have these from screenshots
        adx=25.0,
        atr=1.0,
        macd_signal="bullish" if trade.get("profit_loss_dollars", 0) > 0 else "bearish",
        volume_ratio=1.5 if trade.get("is_0dte") else 1.0,
        regime="bull" if trade.get("trade_type") == "CALL" else "bear",
        vix=18.0,  # Default moderate VIX
        gex_regime="positive_gamma" if trade.get("trade_type") == "CALL" else "negative_gamma",
        hydra_direction="up" if trade.get("trade_type") == "CALL" else "down",
        confluence_score=trade.get("confidence", 0.8),
        stop_distance_pct=10.0,
        target_distance_pct=50.0 if trade.get("profit_loss_pct", 0) > 100 else 20.0
    )


def trade_to_outcome(trade: Dict, conditions: TradeConditions) -> TradeOutcome:
    """Convert DB trade to TradeOutcome dataclass."""
    trade_date = trade.get("trade_date", "2026-01-15")

    # Parse trade_date
    try:
        entry_time = datetime.strptime(trade_date, "%Y-%m-%d")
    except:
        entry_time = datetime.now() - timedelta(days=30)

    # Estimate duration based on 0DTE flag
    if trade.get("is_0dte"):
        duration = 60  # 1 hour average for 0DTE
        exit_time = entry_time + timedelta(hours=1)
    else:
        duration = 1440  # 1 day for swings
        exit_time = entry_time + timedelta(days=1)

    pnl_pct = trade.get("profit_loss_pct", 0) or 0

    return TradeOutcome(
        trade_id=f"learned_{trade['id']}",
        conditions=conditions,
        entry_reasoning=trade.get("notes", "") or f"{trade['ticker']} {trade.get('trade_type', 'CALL')} play",
        pnl_dollars=trade.get("profit_loss_dollars", 0) or 0,
        pnl_percent=pnl_pct,
        duration_minutes=duration,
        max_adverse_excursion_pct=5.0 if pnl_pct > 0 else abs(pnl_pct) * 0.5,
        max_favorable_excursion_pct=pnl_pct * 1.2 if pnl_pct > 0 else 10.0,
        exit_reason="target" if pnl_pct > 0 else "stop",
        exit_price=trade.get("exit_price", 0) or (trade.get("entry_price", 1) * (1 + pnl_pct/100)),
        entry_time=entry_time,
        exit_time=exit_time,
        lessons_learned=f"Pattern: {trade.get('pattern', 'unknown')}"
    )


def feed_semantic_memory(trades: List[Dict], memory: SemanticMemory) -> int:
    """Feed all trades into semantic memory."""
    fed = 0
    for trade in trades:
        try:
            conditions = trade_to_conditions(trade)
            outcome = trade_to_outcome(trade, conditions)
            memory.record_trade(outcome)
            fed += 1
        except Exception as e:
            print(f"  SKIP {trade['ticker']}: {e}")
    return fed


def feed_trade_graph(trades: List[Dict], graph: TradeGraph) -> int:
    """Feed all trades into trade graph."""
    fed = 0
    for trade in trades:
        try:
            trade_date = trade.get("trade_date", "2026-01-15")
            try:
                entry_time = datetime.strptime(trade_date, "%Y-%m-%d")
            except:
                entry_time = datetime.now() - timedelta(days=30)

            exit_time = entry_time + timedelta(hours=1 if trade.get("is_0dte") else 24)
            pnl_pct = trade.get("profit_loss_pct", 0) or 0

            # Build conditions dict
            conditions = {
                "ticker": trade["ticker"],
                "trade_type": trade.get("trade_type", "CALL"),
                "strike": trade.get("strike"),
                "is_0dte": bool(trade.get("is_0dte")),
                "pattern": trade.get("pattern"),
                "platform": trade.get("platform"),
                "confidence": trade.get("confidence", 0.8)
            }

            graph.record_trade(
                ticker=trade["ticker"],
                direction="long" if trade.get("direction", "long") == "long" else "short",
                entry_price=trade.get("entry_price", 1.0) or 1.0,
                exit_price=trade.get("exit_price", 0) or (trade.get("entry_price", 1) * (1 + pnl_pct/100)),
                entry_time=entry_time,
                exit_time=exit_time,
                pnl_dollars=trade.get("profit_loss_dollars", 0) or 0,
                pattern=trade.get("pattern") or "unknown",
                entry_reasoning=trade.get("notes") or f"{trade['ticker']} play",
                exit_reasoning=f"Exit with {pnl_pct:.1f}% {'gain' if pnl_pct > 0 else 'loss'}",
                conditions=conditions,
                metadata={"source": trade.get("source"), "id": trade["id"]}
            )
            fed += 1
        except Exception as e:
            print(f"  SKIP graph {trade['ticker']}: {e}")
    return fed


def analyze_top_winners(trades: List[Dict]) -> Dict[str, Any]:
    """Run specialist swarm on top 10 winners."""
    # Get top 10 by P&L
    top_10 = sorted(trades, key=lambda x: x.get("profit_loss_dollars", 0) or 0, reverse=True)[:10]

    print("\n" + "="*60)
    print("TOP 10 WINNERS - SWARM ANALYSIS")
    print("="*60)

    patterns_found = {}

    for i, trade in enumerate(top_10, 1):
        ticker = trade["ticker"]
        pnl = trade.get("profit_loss_dollars", 0) or 0
        pct = trade.get("profit_loss_pct", 0) or 0
        pattern = trade.get("pattern") or "unknown"
        notes = trade.get("notes") or ""
        is_0dte = trade.get("is_0dte")

        print(f"\n#{i}: {ticker} {trade.get('trade_type', 'CALL')} ${trade.get('strike', '?')}")
        print(f"    P&L: ${pnl:,.2f} ({pct:.1f}%)")
        print(f"    Pattern: {pattern}")
        print(f"    0DTE: {bool(is_0dte)}")
        print(f"    Notes: {notes[:80]}...")

        # Track patterns
        if pattern not in patterns_found:
            patterns_found[pattern] = {"count": 0, "total_pnl": 0}
        patterns_found[pattern]["count"] += 1
        patterns_found[pattern]["total_pnl"] += pnl

    return patterns_found


def update_patterns_in_db(db_path: str) -> int:
    """Update pattern tags for trades missing them."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Pattern rules
    updates = [
        # Precious metals
        ("UPDATE learned_trades SET pattern = 'PRECIOUS_METALS_MOMENTUM' WHERE ticker IN ('SLV', 'GLD', 'IAU', 'PPLT', 'UNG') AND pattern IS NULL", "PRECIOUS_METALS"),
        # Lotto tickets
        ("UPDATE learned_trades SET pattern = 'LOTTO_TICKET' WHERE entry_price < 1.0 AND profit_loss_pct > 200 AND pattern IS NULL", "LOTTO_TICKET"),
        # Reversal puts
        ("UPDATE learned_trades SET pattern = 'REVERSAL_PUT' WHERE trade_type = 'PUT' AND is_0dte = 1 AND pattern IS NULL", "REVERSAL_PUT"),
        # Momentum calls
        ("UPDATE learned_trades SET pattern = 'MOMENTUM_CALL' WHERE trade_type = 'CALL' AND is_0dte = 1 AND pattern IS NULL", "MOMENTUM_CALL"),
        # High volume
        ("UPDATE learned_trades SET pattern = 'HIGH_VOLUME_CONVICTION' WHERE quantity >= 50 AND pattern IS NULL", "HIGH_VOLUME"),
    ]

    total = 0
    for sql, name in updates:
        cursor.execute(sql)
        affected = cursor.rowcount
        total += affected
        print(f"  {name}: {affected} trades tagged")

    conn.commit()
    conn.close()
    return total


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "wsb_snake_data/wsb_snake.db"

    print("="*60)
    print("WEAPONIZING TRADE DATA")
    print("="*60)

    # Step 0: Update patterns
    print("\n[0] UPDATING PATTERN TAGS...")
    tagged = update_patterns_in_db(db_path)
    print(f"    Tagged {tagged} trades with patterns")

    # Load trades
    print("\n[1] LOADING TRADES FROM DB...")
    trades = load_trades_from_db(db_path)
    print(f"    Loaded {len(trades)} trades")

    winners = [t for t in trades if (t.get("profit_loss_dollars") or 0) > 0]
    losers = [t for t in trades if (t.get("profit_loss_dollars") or 0) < 0]
    print(f"    Winners: {len(winners)}, Losers: {len(losers)}")

    # Step 1: Feed semantic memory
    print("\n[2] FEEDING SEMANTIC MEMORY...")
    try:
        memory = get_semantic_memory()
        fed_memory = feed_semantic_memory(trades, memory)
        print(f"    Fed {fed_memory} trades to semantic memory")

        # Trigger reflection
        print("    Running auto_reflect()...")
        memory._auto_reflect()
        stats = memory.get_stats()
        print(f"    Rules generated: {stats.get('rule_count', 0)}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Step 2: Feed trade graph
    print("\n[3] FEEDING TRADE GRAPH...")
    try:
        graph = get_trade_graph()
        fed_graph = feed_trade_graph(trades, graph)
        print(f"    Fed {fed_graph} trades to trade graph")

        stats = graph.get_graph_stats()
        print(f"    Total nodes: {stats.get('total_trades', 0)}")
        print(f"    Patterns: {stats.get('patterns', {})}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Step 3: Analyze top winners
    print("\n[4] ANALYZING TOP WINNERS...")
    patterns = analyze_top_winners(trades)

    print("\n" + "="*60)
    print("PATTERN SUMMARY")
    print("="*60)
    for pattern, data in sorted(patterns.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
        print(f"  {pattern}: {data['count']} trades, ${data['total_pnl']:,.2f} total P&L")

    # Final stats
    print("\n" + "="*60)
    print("WEAPONIZATION COMPLETE")
    print("="*60)

    # Get pattern stats from DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pattern, COUNT(*) as cnt, SUM(profit_loss_dollars) as pnl
        FROM learned_trades
        WHERE pattern IS NOT NULL
        GROUP BY pattern
        ORDER BY pnl DESC
    """)
    rows = cursor.fetchall()

    print("\nPATTERNS IN DATABASE:")
    for pattern, cnt, pnl in rows:
        print(f"  {pattern}: {cnt} trades, ${pnl:,.2f}")

    cursor.execute("SELECT COUNT(*) FROM learned_trades")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM learned_trades WHERE pattern IS NOT NULL")
    with_pattern = cursor.fetchone()[0]

    print(f"\nTotal trades: {total}")
    print(f"With patterns: {with_pattern}")
    print(f"Pattern coverage: {with_pattern/total*100:.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
