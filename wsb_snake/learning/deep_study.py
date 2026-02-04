"""
Deep Study Module - Off-Market Hours Trade Analysis

When markets are closed, this module:
1. Processes pending screenshots from Google Drive
2. Researches WHY each trade worked (news, catalysts, patterns)
3. Cross-references with market data
4. Builds comprehensive trade understanding
5. Distills actionable formulas for replication

Runs automatically during:
- Pre-market (before 9:30 AM ET)
- After-hours (after 4:00 PM ET)
- Weekends
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pytz

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection
from wsb_snake.collectors.screenshot_system import screenshot_system

logger = get_logger(__name__)

# Study prompt for GPT-4o - deep analysis
DEEP_ANALYSIS_PROMPT = """You are analyzing a successful trade to understand WHY it worked.

TRADE DATA:
{trade_data}

MARKET CONTEXT:
{market_context}

NEWS/CATALYSTS FOUND:
{news_context}

Analyze this trade deeply and provide:

1. **PRIMARY CATALYST**: What was the main driver? (earnings, news, technical breakout, sector rotation, etc.)

2. **ENTRY TIMING**: Why was the entry time optimal? What setup preceded it?

3. **PATTERN CLASSIFICATION**: What chart/price pattern was this?
   - VWAP bounce, breakout, reversal, squeeze, momentum continuation, gap fill, etc.

4. **RISK/REWARD ANALYSIS**: How was the trade structured? Was it aggressive or conservative?

5. **REPLICATION CONDITIONS**: What conditions would need to be present to replicate this trade?
   - Time of day
   - Market regime (trending, ranging, volatile)
   - Volume conditions
   - Technical setup
   - Catalyst type

6. **CONFIDENCE SCORE**: 1-10, how replicable is this setup?

7. **ACTIONABLE FORMULA**: Write a simple IF-THEN rule for replication.
   Example: "IF SPY breaks above VWAP with 2x volume in power hour AND VIX < 20 THEN buy ATM calls, target +15%, stop -8%"

Return as JSON:
{
    "primary_catalyst": "...",
    "catalyst_type": "earnings|news|technical|sector|momentum|other",
    "entry_timing_analysis": "...",
    "pattern_type": "breakout|reversal|squeeze|momentum|vwap_play|gap_fill|other",
    "risk_reward_analysis": "...",
    "replication_conditions": {
        "time_windows": ["power_hour", "open"],
        "market_regime": "trending|ranging|volatile",
        "volume_requirement": "above_average|2x_average|3x_average",
        "technical_setup": "...",
        "catalyst_required": true|false
    },
    "confidence_score": 8,
    "actionable_formula": "IF ... THEN ...",
    "key_lessons": ["lesson1", "lesson2"],
    "warnings": ["warning1"]
}
"""


class DeepStudyEngine:
    """
    Off-market hours deep analysis engine.

    Studies trade screenshots to understand:
    - What catalyst drove the move
    - What pattern/setup preceded entry
    - What market conditions were present
    - How to replicate the success
    """

    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.polygon_api_key = os.environ.get("POLYGON_API_KEY", "")
        self.finnhub_api_key = os.environ.get("FINNHUB_API_KEY", "")

        self._et = pytz.timezone('US/Eastern')
        logger.info("DeepStudyEngine initialized")

    def is_study_time(self) -> bool:
        """Check if it's off-market hours (good time for deep study)."""
        now_et = datetime.now(self._et)
        hour = now_et.hour
        weekday = now_et.weekday()

        # Weekends
        if weekday >= 5:
            return True

        # Before market open (before 9:30 AM)
        if hour < 9 or (hour == 9 and now_et.minute < 30):
            return True

        # After market close (after 4:00 PM)
        if hour >= 16:
            return True

        return False

    def get_pending_studies(self, limit: int = 10) -> List[Dict]:
        """Get learned trades that haven't been deeply studied yet."""
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT lt.*, s.extracted_data
            FROM learned_trades lt
            LEFT JOIN screenshots s ON lt.screenshot_id = s.id
            WHERE lt.market_conditions IS NULL
              AND lt.profit_loss > 0
              AND lt.ticker IS NOT NULL
            ORDER BY lt.profit_loss_pct DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def fetch_market_context(self, ticker: str, trade_date: str) -> Dict:
        """Fetch market data context for the trade date."""
        context = {
            "ticker": ticker,
            "trade_date": trade_date,
            "price_data": None,
            "volume_data": None,
            "sector_performance": None,
            "vix_level": None,
            "spy_performance": None
        }

        if not self.polygon_api_key:
            return context

        try:
            # Parse date
            if trade_date:
                date_obj = datetime.strptime(trade_date, "%Y-%m-%d")
            else:
                date_obj = datetime.now() - timedelta(days=1)

            date_str = date_obj.strftime("%Y-%m-%d")

            # Fetch ticker data
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date_str}/{date_str}"
            resp = requests.get(url, params={"apiKey": self.polygon_api_key}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    bar = data["results"][0]
                    context["price_data"] = {
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "volume": bar.get("v"),
                        "vwap": bar.get("vw")
                    }
                    context["volume_data"] = bar.get("v")

            # Fetch SPY for market context
            spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{date_str}/{date_str}"
            resp = requests.get(spy_url, params={"apiKey": self.polygon_api_key}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    bar = data["results"][0]
                    spy_change = ((bar.get("c", 0) - bar.get("o", 1)) / bar.get("o", 1)) * 100
                    context["spy_performance"] = f"{spy_change:+.2f}%"

        except Exception as e:
            logger.warning(f"Failed to fetch market context: {e}")

        return context

    def fetch_news_context(self, ticker: str, trade_date: str) -> List[Dict]:
        """Fetch news around the trade date."""
        news = []

        if not self.finnhub_api_key:
            return news

        try:
            if trade_date:
                date_obj = datetime.strptime(trade_date, "%Y-%m-%d")
            else:
                date_obj = datetime.now() - timedelta(days=1)

            from_date = (date_obj - timedelta(days=2)).strftime("%Y-%m-%d")
            to_date = date_obj.strftime("%Y-%m-%d")

            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "token": self.finnhub_api_key
            }

            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                articles = resp.json()[:5]  # Top 5 articles
                for a in articles:
                    news.append({
                        "headline": a.get("headline"),
                        "summary": a.get("summary", "")[:200],
                        "source": a.get("source"),
                        "datetime": a.get("datetime")
                    })

        except Exception as e:
            logger.warning(f"Failed to fetch news context: {e}")

        return news

    def perform_deep_analysis(self, trade: Dict) -> Optional[Dict]:
        """
        Perform deep analysis on a trade using GPT-4o.

        Returns analyzed insights or None on failure.
        """
        if not self.openai_api_key:
            logger.warning("OpenAI API key not configured for deep study")
            return None

        ticker = trade.get("ticker")
        trade_date = trade.get("trade_date")

        # Gather context
        market_context = self.fetch_market_context(ticker, trade_date)
        news_context = self.fetch_news_context(ticker, trade_date)

        # Build trade data summary
        trade_data = {
            "ticker": ticker,
            "trade_type": trade.get("trade_type"),
            "strike": trade.get("strike"),
            "expiry": trade.get("expiry"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "profit_loss": trade.get("profit_loss"),
            "profit_loss_pct": trade.get("profit_loss_pct"),
            "entry_time": trade.get("entry_time"),
            "exit_time": trade.get("exit_time"),
            "platform": trade.get("platform"),
            "detected_pattern": trade.get("detected_pattern"),
            "setup_description": trade.get("setup_description")
        }

        # Format prompt
        prompt = DEEP_ANALYSIS_PROMPT.format(
            trade_data=json.dumps(trade_data, indent=2),
            market_context=json.dumps(market_context, indent=2),
            news_context=json.dumps(news_context, indent=2)
        )

        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.3
            }

            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if resp.status_code != 200:
                logger.error(f"OpenAI API error: {resp.status_code}")
                return None

            content = resp.json()["choices"][0]["message"]["content"]

            # Parse JSON from response
            content = content.strip()
            if content.startswith("```"):
                import re
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            analysis = json.loads(content)
            return analysis

        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
            return None

    def save_analysis(self, trade_id: int, analysis: Dict):
        """Save deep analysis results to the database."""
        conn = get_connection()
        cursor = conn.cursor()

        # Update learned_trades with analysis
        cursor.execute("""
            UPDATE learned_trades
            SET detected_pattern = ?,
                setup_description = ?,
                market_conditions = ?,
                confidence_score = ?
            WHERE id = ?
        """, (
            analysis.get("pattern_type"),
            analysis.get("actionable_formula"),
            json.dumps({
                "catalyst": analysis.get("primary_catalyst"),
                "catalyst_type": analysis.get("catalyst_type"),
                "replication_conditions": analysis.get("replication_conditions"),
                "key_lessons": analysis.get("key_lessons"),
                "warnings": analysis.get("warnings")
            }),
            analysis.get("confidence_score", 5) / 10.0,
            trade_id
        ))

        # Update or create trade recipe with deeper insights
        ticker = None
        cursor.execute("SELECT ticker, trade_type FROM learned_trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        if row:
            ticker = row["ticker"]
            trade_type = row["trade_type"]

            replication = analysis.get("replication_conditions", {})
            time_windows = replication.get("time_windows", ["any"])
            time_window = time_windows[0] if time_windows else "any"

            recipe_name = f"{ticker}_{trade_type}_{analysis.get('pattern_type', 'unknown')}_{time_window}"

            # Check if recipe exists
            cursor.execute("SELECT id FROM trade_recipes WHERE name = ?", (recipe_name,))
            existing = cursor.fetchone()

            if existing:
                # Update with new insights
                cursor.execute("""
                    UPDATE trade_recipes
                    SET entry_conditions = ?,
                        exit_conditions = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    json.dumps({
                        "pattern": analysis.get("pattern_type"),
                        "catalyst_type": analysis.get("catalyst_type"),
                        "formula": analysis.get("actionable_formula"),
                        "conditions": replication
                    }),
                    json.dumps({
                        "lessons": analysis.get("key_lessons"),
                        "warnings": analysis.get("warnings")
                    }),
                    datetime.utcnow().isoformat(),
                    existing["id"]
                ))
            else:
                # Create new detailed recipe
                cursor.execute("""
                    INSERT INTO trade_recipes (
                        name, ticker_pattern, trade_type, time_window,
                        entry_conditions, exit_conditions,
                        source_trade_count, win_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, 1.0)
                """, (
                    recipe_name,
                    ticker,
                    trade_type,
                    time_window,
                    json.dumps({
                        "pattern": analysis.get("pattern_type"),
                        "catalyst_type": analysis.get("catalyst_type"),
                        "formula": analysis.get("actionable_formula"),
                        "conditions": replication
                    }),
                    json.dumps({
                        "lessons": analysis.get("key_lessons"),
                        "warnings": analysis.get("warnings")
                    })
                ))

        conn.commit()
        conn.close()

        logger.info(f"Saved deep analysis for trade #{trade_id}")

    def run_study_session(self, max_trades: int = 5) -> Dict:
        """
        Run a deep study session on pending trades.

        Returns summary of what was studied.
        """
        # First, process any new screenshots
        logger.info("Processing new screenshots...")
        screenshot_results = screenshot_system.process_new_screenshots()

        # Get trades needing deep study
        pending = self.get_pending_studies(limit=max_trades)
        logger.info(f"Found {len(pending)} trades needing deep study")

        studied = []
        failed = []

        for trade in pending:
            trade_id = trade["id"]
            ticker = trade.get("ticker", "UNKNOWN")

            logger.info(f"Deep studying trade #{trade_id}: {ticker}")

            analysis = self.perform_deep_analysis(trade)

            if analysis:
                self.save_analysis(trade_id, analysis)
                studied.append({
                    "trade_id": trade_id,
                    "ticker": ticker,
                    "pattern": analysis.get("pattern_type"),
                    "catalyst": analysis.get("catalyst_type"),
                    "confidence": analysis.get("confidence_score"),
                    "formula": analysis.get("actionable_formula")
                })
            else:
                failed.append(trade_id)

        return {
            "screenshots_processed": len(screenshot_results),
            "trades_studied": len(studied),
            "trades_failed": len(failed),
            "studied_details": studied
        }

    def get_study_summary(self) -> Dict:
        """Get summary of all deep studies performed."""
        conn = get_connection()
        cursor = conn.cursor()

        # Count studied vs unstudied
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN market_conditions IS NOT NULL THEN 1 ELSE 0 END) as studied,
                SUM(CASE WHEN market_conditions IS NULL THEN 1 ELSE 0 END) as pending
            FROM learned_trades
            WHERE profit_loss > 0
        """)
        row = cursor.fetchone()

        # Get pattern distribution
        cursor.execute("""
            SELECT detected_pattern, COUNT(*) as count
            FROM learned_trades
            WHERE detected_pattern IS NOT NULL
            GROUP BY detected_pattern
            ORDER BY count DESC
        """)
        patterns = [{"pattern": r["detected_pattern"], "count": r["count"]} for r in cursor.fetchall()]

        # Get recipes with formulas
        cursor.execute("""
            SELECT name, entry_conditions
            FROM trade_recipes
            WHERE entry_conditions IS NOT NULL
              AND entry_conditions LIKE '%formula%'
            ORDER BY win_rate DESC
            LIMIT 10
        """)
        recipes_with_formulas = []
        for r in cursor.fetchall():
            try:
                conditions = json.loads(r["entry_conditions"])
                recipes_with_formulas.append({
                    "name": r["name"],
                    "formula": conditions.get("formula", "N/A")
                })
            except:
                pass

        conn.close()

        return {
            "total_winning_trades": row["total"] or 0,
            "deeply_studied": row["studied"] or 0,
            "pending_study": row["pending"] or 0,
            "pattern_distribution": patterns,
            "actionable_formulas": recipes_with_formulas
        }


# Global instance
deep_study_engine = DeepStudyEngine()


def run_idle_study():
    """
    Entry point for idle-time study.
    Call this from main.py during off-market hours.
    """
    engine = deep_study_engine

    if not engine.is_study_time():
        logger.info("Market hours - skipping deep study")
        return None

    logger.info("Off-market hours - starting deep study session")
    results = engine.run_study_session(max_trades=5)

    logger.info(f"Study session complete: {results['trades_studied']} trades analyzed")
    return results


# CLI
def main():
    """CLI for deep study module."""
    import argparse

    parser = argparse.ArgumentParser(description="Deep Study Engine")
    parser.add_argument("command", choices=["study", "summary", "force"])
    parser.add_argument("--max", type=int, default=5, help="Max trades to study")

    args = parser.parse_args()

    if args.command == "study":
        if not deep_study_engine.is_study_time():
            print("Market hours active. Use 'force' to study anyway.")
            return
        results = deep_study_engine.run_study_session(args.max)
        print(f"\nStudy Session Complete")
        print(f"  Screenshots processed: {results['screenshots_processed']}")
        print(f"  Trades studied: {results['trades_studied']}")
        if results['studied_details']:
            print("\nStudied Trades:")
            for s in results['studied_details']:
                print(f"  #{s['trade_id']} {s['ticker']}: {s['pattern']} ({s['catalyst']})")
                print(f"    Formula: {s['formula'][:80]}..." if s.get('formula') else "")

    elif args.command == "force":
        results = deep_study_engine.run_study_session(args.max)
        print(f"\nForced Study Complete: {results['trades_studied']} trades analyzed")
        for s in results.get('studied_details', []):
            print(f"  #{s['trade_id']} {s['ticker']}: {s['formula'][:100]}..." if s.get('formula') else "")

    elif args.command == "summary":
        summary = deep_study_engine.get_study_summary()
        print("\n=== Deep Study Summary ===\n")
        print(f"Total winning trades: {summary['total_winning_trades']}")
        print(f"Deeply studied: {summary['deeply_studied']}")
        print(f"Pending study: {summary['pending_study']}")

        if summary['pattern_distribution']:
            print("\nPattern Distribution:")
            for p in summary['pattern_distribution']:
                print(f"  {p['pattern']}: {p['count']}")

        if summary['actionable_formulas']:
            print("\nActionable Formulas:")
            for f in summary['actionable_formulas']:
                print(f"\n  [{f['name']}]")
                print(f"  {f['formula']}")


if __name__ == "__main__":
    main()
