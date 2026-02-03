"""
Screenshot Learning System - Main Integration Module

Ties together:
- ScreenshotCollector (Google Drive watcher)
- TradeExtractor (GPT-4o vision extraction)
- TradeLearner (pattern learning & application)

Usage:
    from wsb_snake.collectors.screenshot_system import screenshot_system

    # Start background watching
    screenshot_system.start()

    # Or process manually
    screenshot_system.process_new_screenshots()

    # Get confidence boost for a trade
    boost, reasons = screenshot_system.get_trade_boost("SPY", "CALLS", 15)
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from wsb_snake.utils.logger import get_logger
from wsb_snake.collectors.screenshot_collector import ScreenshotCollector
from wsb_snake.collectors.trade_extractor import TradeExtractor
from wsb_snake.learning.trade_learner import TradeLearner

logger = get_logger(__name__)


class ScreenshotLearningSystem:
    """
    Main orchestrator for the screenshot learning system.

    Integrates:
    1. Google Drive watching for new screenshots
    2. GPT-4o vision for trade data extraction
    3. Pattern learning and recipe creation
    4. Confidence boosting for live trades
    """

    def __init__(self):
        self.folder_id = os.environ.get(
            "GOOGLE_DRIVE_FOLDER_ID",
            "1EbGgR2r_0jxDjQWvlN9yuxlrzUPWvLf4"
        )
        self.scan_interval = int(os.environ.get("SCREENSHOT_SCAN_INTERVAL", 300))

        self._collector: Optional[ScreenshotCollector] = None
        self._extractor: Optional[TradeExtractor] = None
        self._learner: Optional[TradeLearner] = None

        self._initialized = False
        logger.info("ScreenshotLearningSystem created")

    def _ensure_initialized(self):
        """Lazy initialization of components."""
        if self._initialized:
            return

        try:
            self._collector = ScreenshotCollector(
                folder_id=self.folder_id,
                scan_interval_seconds=self.scan_interval
            )
            self._extractor = TradeExtractor()
            self._learner = TradeLearner()
            self._initialized = True
            logger.info("ScreenshotLearningSystem initialized")
        except Exception as e:
            logger.error(f"Failed to initialize screenshot system: {e}")
            raise

    def start(self):
        """Start background screenshot watching."""
        self._ensure_initialized()
        self._collector.start_watching()
        logger.info("Screenshot learning system started (background mode)")

    def stop(self):
        """Stop background watching."""
        if self._collector:
            self._collector.stop_watching()
        logger.info("Screenshot learning system stopped")

    def process_new_screenshots(self) -> List[Dict]:
        """
        Manually process all new screenshots.
        Returns list of extracted trade data.
        """
        self._ensure_initialized()

        results = []
        new_files = self._collector.fetch_new_screenshots()

        logger.info(f"Processing {len(new_files)} new screenshots")

        for file in new_files:
            try:
                # Extract trade data
                extracted = self._extractor.extract_trade_data(
                    image_base64=file.content_base64,
                    filename=file.name,
                    mime_type=file.mime_type
                )

                if extracted:
                    # Save to database
                    screenshot_id = self._collector.mark_processed(
                        file,
                        status="processed",
                        extracted_data=extracted
                    )

                    # Save learned trade
                    self._extractor.save_learned_trade(screenshot_id, extracted)

                    results.append({
                        "filename": file.name,
                        "status": "success",
                        "data": extracted
                    })

                    logger.info(f"Processed: {file.name} -> {extracted.get('ticker')} {extracted.get('trade_type')}")
                else:
                    self._collector.mark_processed(
                        file,
                        status="failed",
                        error_message="Could not extract trade data"
                    )
                    results.append({
                        "filename": file.name,
                        "status": "failed",
                        "error": "Extraction failed"
                    })

            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                self._collector.mark_processed(
                    file,
                    status="error",
                    error_message=str(e)
                )
                results.append({
                    "filename": file.name,
                    "status": "error",
                    "error": str(e)
                })

        # Reload learner with new data
        if results:
            self._learner.reload_recipes()

        return results

    def get_trade_boost(
        self,
        ticker: str,
        trade_type: str,
        current_hour: int,
        pattern: Optional[str] = None
    ) -> Tuple[float, List[str]]:
        """
        Get confidence boost based on learned patterns.

        Args:
            ticker: Stock symbol (e.g., "SPY")
            trade_type: "CALLS" or "PUTS"
            current_hour: Current hour (0-23, ET)
            pattern: Detected pattern name if any

        Returns:
            Tuple of (adjustment, list_of_reasons)
            adjustment is a multiplier: 0.15 means +15% confidence
        """
        self._ensure_initialized()
        return self._learner.get_confidence_adjustment(
            ticker, trade_type, current_hour, pattern
        )

    def should_replicate(
        self,
        ticker: str,
        trade_type: str,
        current_hour: int
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if current conditions match a proven winning setup.

        Returns:
            Tuple of (should_trade, matching_recipe)
        """
        self._ensure_initialized()
        return self._learner.should_replicate_trade(ticker, trade_type, current_hour)

    def get_ticker_insights(self, ticker: str) -> Dict:
        """Get learned insights for a specific ticker."""
        self._ensure_initialized()
        return self._learner.get_ticker_insights(ticker)

    def get_stats(self) -> Dict:
        """Get comprehensive stats about the learning system."""
        self._ensure_initialized()

        stats = {
            "collector": self._collector.get_processing_stats(),
            "learner": self._learner.get_learned_trade_stats(),
            "folder_id": self.folder_id,
            "scan_interval": self.scan_interval,
        }

        return stats

    def list_recipes(self) -> List[Dict]:
        """Get all active trade recipes."""
        self._ensure_initialized()
        return self._learner._recipes


# Global singleton
screenshot_system = ScreenshotLearningSystem()


# CLI interface
def main():
    """Command-line interface for the screenshot system."""
    import argparse

    parser = argparse.ArgumentParser(description="Screenshot Learning System")
    parser.add_argument("command", choices=["process", "stats", "recipes", "insights", "watch"])
    parser.add_argument("--ticker", help="Ticker for insights command")
    parser.add_argument("--folder", help="Override Google Drive folder ID")

    args = parser.parse_args()

    if args.folder:
        os.environ["GOOGLE_DRIVE_FOLDER_ID"] = args.folder

    if args.command == "process":
        print("Processing new screenshots...")
        results = screenshot_system.process_new_screenshots()
        print(f"\nProcessed {len(results)} screenshots:")
        for r in results:
            status = r["status"]
            filename = r["filename"]
            if status == "success":
                data = r["data"]
                print(f"  [OK] {filename} -> {data.get('ticker')} {data.get('trade_type')} P&L: {data.get('profit_loss_pct', 'N/A')}%")
            else:
                print(f"  [FAIL] {filename}: {r.get('error', 'Unknown error')}")

    elif args.command == "stats":
        stats = screenshot_system.get_stats()
        print("\n=== Screenshot Learning System Stats ===\n")

        print("Collector:")
        for k, v in stats["collector"].items():
            print(f"  {k}: {v}")

        print("\nLearner:")
        learner = stats["learner"]
        print(f"  Total learned trades: {learner.get('total_learned_trades', 0)}")
        print(f"  Winners: {learner.get('winners', 0)}")
        print(f"  Losers: {learner.get('losers', 0)}")
        print(f"  Avg P&L %: {learner.get('avg_pnl_pct', 0):.1f}%")
        print(f"  Active recipes: {learner.get('active_recipes', 0)}")

        if learner.get("top_recipes"):
            print("\n  Top Recipes:")
            for r in learner["top_recipes"]:
                print(f"    - {r['name']}: WR {r['win_rate']:.0%}, Avg {r['avg_profit_pct']:.1f}%")

    elif args.command == "recipes":
        recipes = screenshot_system.list_recipes()
        print(f"\n=== Active Trade Recipes ({len(recipes)}) ===\n")
        for r in recipes:
            print(f"  {r['name']}")
            print(f"    Ticker: {r.get('ticker_pattern')} | Type: {r.get('trade_type')} | Time: {r.get('time_window')}")
            print(f"    Win Rate: {r.get('win_rate', 0):.0%} | Trades: {r.get('source_trade_count', 0)} | Avg P&L: {r.get('avg_profit_pct', 0):.1f}%")
            print()

    elif args.command == "insights":
        if not args.ticker:
            print("Error: --ticker required for insights command")
            return
        insights = screenshot_system.get_ticker_insights(args.ticker.upper())
        print(f"\n=== Insights for {args.ticker.upper()} ===\n")
        print(f"  Total trades: {insights.get('total_trades', 0)}")
        print(f"  Win rate: {insights.get('win_rate', 0):.0%}")
        print(f"  Avg P&L: {insights.get('avg_pnl_pct', 0):.1f}%")
        if insights.get("best_trade_type"):
            print(f"  Best trade type: {insights['best_trade_type']} ({insights.get('best_type_avg_pnl', 0):.1f}%)")
        if insights.get("best_entry_times"):
            print(f"  Best entry times: {', '.join(insights['best_entry_times'])}")
        if insights.get("recipes"):
            print(f"  Recipes: {len(insights['recipes'])}")

    elif args.command == "watch":
        print(f"Starting screenshot watcher (folder: {screenshot_system.folder_id})...")
        print("Press Ctrl+C to stop\n")
        screenshot_system.start()
        try:
            import time
            while True:
                time.sleep(60)
                stats = screenshot_system.get_stats()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed: {stats['collector']['processed']}, Pending: {stats['collector']['pending']}")
        except KeyboardInterrupt:
            screenshot_system.stop()
            print("\nStopped.")


if __name__ == "__main__":
    main()
