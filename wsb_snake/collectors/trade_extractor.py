"""
Trade Extractor - GPT-4o Vision Trade Data Extraction

Extracts structured trade data from trading platform screenshots.
Supports: Robinhood, Webull, TD Ameritrade, Schwab, Fidelity, and others.

Uses GPT-4o vision for accurate OCR and context understanding.
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

logger = get_logger(__name__)

# Extraction prompt for GPT-4o vision
TRADE_EXTRACTION_PROMPT = """Analyze this trading screenshot and extract all trade information.

Return a JSON object with these fields (use null for any field you cannot determine):

{
    "ticker": "SPY",                    // Stock/ETF symbol
    "trade_type": "CALLS",              // CALLS, PUTS, SHARES, or null
    "strike": 590.00,                   // Strike price for options, null for shares
    "expiry": "2026-02-03",             // Expiration date (YYYY-MM-DD format)
    "direction": "long",                // long or short

    "entry_price": 2.50,                // Price paid per share/contract
    "exit_price": 4.25,                 // Price sold at
    "quantity": 10,                     // Number of contracts or shares

    "capital_deployed": 2500.00,        // Total capital used
    "profit_loss": 1750.00,             // Profit or loss in dollars (negative for loss)
    "profit_loss_pct": 70.0,            // Percentage gain/loss

    "platform": "Robinhood",            // Trading platform detected
    "trade_date": "2026-02-03",         // Date of trade
    "entry_time": "09:35",              // Entry time if visible (HH:MM)
    "exit_time": "10:45",               // Exit time if visible

    "is_winner": true,                  // true if profitable
    "is_options": true,                 // true if options trade

    "visible_indicators": ["VWAP", "RSI"],  // Any visible chart indicators
    "chart_pattern": "breakout",            // Pattern if visible (breakout, reversal, etc.)
    "notes": "Power hour momentum play"     // Any other relevant observations
}

IMPORTANT:
- Extract EXACTLY what you see, don't guess
- For P&L, positive = profit, negative = loss
- If it's a P&L summary showing multiple trades, extract the totals
- Look for: fill prices, average cost, current value, gain/loss
- Identify the platform from UI elements (Robinhood green, Webull dark theme, etc.)

Return ONLY valid JSON, no markdown or explanation."""


class TradeExtractor:
    """
    Extracts trade data from screenshots using GPT-4o vision.
    """

    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

        # Rate limiting
        self._last_call_time = 0
        self._min_call_interval = 3  # seconds between calls

        logger.info("TradeExtractor initialized")

    def extract_trade_data(
        self,
        image_base64: str,
        filename: str = "screenshot.png",
        mime_type: str = "image/png",
    ) -> Optional[Dict[str, Any]]:
        """
        Extract trade data from a screenshot using GPT-4o vision.

        Args:
            image_base64: Base64-encoded image content
            filename: Original filename (for logging)
            mime_type: Image MIME type

        Returns:
            Extracted trade data dict, or None on failure
        """
        import time

        # Rate limiting
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)

        self._last_call_time = time.time()

        # Try GPT-4o first, fall back to Gemini
        result = self._extract_with_openai(image_base64, mime_type)

        if result is None and self.gemini_api_key:
            logger.info("Falling back to Gemini for extraction")
            result = self._extract_with_gemini(image_base64, mime_type)

        if result:
            # Post-process and validate
            result = self._validate_and_clean(result, filename)

        return result

    def _extract_with_openai(
        self,
        image_base64: str,
        mime_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract using OpenAI GPT-4o vision."""
        import requests

        if not self.openai_api_key:
            logger.warning("OpenAI API key not configured")
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }

            # Determine media type for data URL
            media_type = mime_type if mime_type else "image/png"

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": TRADE_EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1  # Low temp for accurate extraction
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return None

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            return self._parse_json_response(content)

        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return None

    def _extract_with_gemini(
        self,
        image_base64: str,
        mime_type: str
    ) -> Optional[Dict[str, Any]]:
        """Extract using Google Gemini vision."""
        import requests

        if not self.gemini_api_key:
            return None

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"

            payload = {
                "contents": [{
                    "parts": [
                        {"text": TRADE_EXTRACTION_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type or "image/png",
                                "data": image_base64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1000
                }
            }

            response = requests.post(url, json=payload, timeout=60)

            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None

            data = response.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]

            return self._parse_json_response(content)

        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            return None

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from AI response, handling markdown code blocks."""
        try:
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                # Remove opening ```json or ```
                content = re.sub(r"^```(?:json)?\s*", "", content)
                # Remove closing ```
                content = re.sub(r"\s*```$", "", content)

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw content: {content[:500]}")
            return None

    def _validate_and_clean(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Dict[str, Any]:
        """Validate and clean extracted data."""

        # Ensure required fields exist
        data.setdefault("ticker", None)
        data.setdefault("trade_type", None)
        data.setdefault("entry_price", None)
        data.setdefault("exit_price", None)
        data.setdefault("profit_loss", None)
        data.setdefault("profit_loss_pct", None)
        data.setdefault("platform", "unknown")
        data.setdefault("is_winner", None)
        data.setdefault("is_options", None)

        # Normalize ticker to uppercase
        if data.get("ticker"):
            data["ticker"] = data["ticker"].upper().strip()

        # Normalize trade_type
        if data.get("trade_type"):
            tt = data["trade_type"].upper()
            if tt in ["CALL", "CALLS", "C"]:
                data["trade_type"] = "CALLS"
            elif tt in ["PUT", "PUTS", "P"]:
                data["trade_type"] = "PUTS"
            elif tt in ["SHARE", "SHARES", "STOCK", "STOCKS"]:
                data["trade_type"] = "SHARES"

        # Determine is_winner if not set
        if data.get("is_winner") is None:
            if data.get("profit_loss") is not None:
                data["is_winner"] = data["profit_loss"] > 0
            elif data.get("profit_loss_pct") is not None:
                data["is_winner"] = data["profit_loss_pct"] > 0

        # Determine is_options if not set
        if data.get("is_options") is None:
            data["is_options"] = data.get("trade_type") in ["CALLS", "PUTS"]

        # Add metadata
        data["source_filename"] = filename
        data["extracted_at"] = datetime.utcnow().isoformat()

        return data

    def save_learned_trade(
        self,
        screenshot_id: int,
        extracted_data: Dict[str, Any]
    ) -> int:
        """
        Save extracted trade data to the learned_trades table.

        Returns:
            The learned_trade ID
        """
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO learned_trades (
                screenshot_id, ticker, trade_type, strike, expiry,
                entry_price, exit_price, contracts, shares,
                capital_deployed, profit_loss, profit_loss_pct,
                platform, trade_date, entry_time, exit_time,
                detected_pattern, setup_description, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            screenshot_id,
            extracted_data.get("ticker"),
            extracted_data.get("trade_type"),
            extracted_data.get("strike"),
            extracted_data.get("expiry"),
            extracted_data.get("entry_price"),
            extracted_data.get("exit_price"),
            extracted_data.get("quantity") if extracted_data.get("is_options") else None,
            extracted_data.get("quantity") if not extracted_data.get("is_options") else None,
            extracted_data.get("capital_deployed"),
            extracted_data.get("profit_loss"),
            extracted_data.get("profit_loss_pct"),
            extracted_data.get("platform"),
            extracted_data.get("trade_date"),
            extracted_data.get("entry_time"),
            extracted_data.get("exit_time"),
            extracted_data.get("chart_pattern"),
            extracted_data.get("notes"),
            0.8 if extracted_data.get("ticker") else 0.3,  # Confidence based on extraction quality
        ))

        learned_trade_id = cursor.lastrowid

        # Update screenshot with learned trade ID
        cursor.execute("""
            UPDATE screenshots
            SET learned_trade_id = ?
            WHERE id = ?
        """, (learned_trade_id, screenshot_id))

        conn.commit()
        conn.close()

        logger.info(f"Saved learned trade #{learned_trade_id}: {extracted_data.get('ticker')} {extracted_data.get('trade_type')}")

        # Trigger pattern learning
        self._learn_from_trade(learned_trade_id, extracted_data)

        return learned_trade_id

    def _learn_from_trade(
        self,
        learned_trade_id: int,
        data: Dict[str, Any]
    ):
        """
        Analyze the trade and create/update trade recipes.
        This is where the system learns patterns from winning trades.
        """
        if not data.get("is_winner"):
            logger.debug(f"Skipping pattern learning for losing trade #{learned_trade_id}")
            return

        # Only learn from clear winners (>10% gain)
        pnl_pct = data.get("profit_loss_pct", 0) or 0
        if pnl_pct < 10:
            logger.debug(f"Trade #{learned_trade_id} gain too small for pattern learning ({pnl_pct}%)")
            return

        ticker = data.get("ticker")
        trade_type = data.get("trade_type")
        pattern = data.get("chart_pattern")
        entry_time = data.get("entry_time")

        if not ticker:
            return

        # Determine time window
        time_window = self._classify_time_window(entry_time)

        # Create or update recipe
        recipe_name = f"{ticker}_{trade_type}_{pattern or 'momentum'}_{time_window}"

        conn = get_connection()
        cursor = conn.cursor()

        # Check if recipe exists
        cursor.execute("SELECT id, source_trade_count, total_profit, win_rate FROM trade_recipes WHERE name = ?", (recipe_name,))
        existing = cursor.fetchone()

        if existing:
            # Update existing recipe
            new_count = existing["source_trade_count"] + 1
            new_profit = existing["total_profit"] + (data.get("profit_loss") or 0)
            # Simplified win rate update (assumes this is a winner)
            new_win_rate = ((existing["win_rate"] * existing["source_trade_count"]) + 1) / new_count

            cursor.execute("""
                UPDATE trade_recipes
                SET source_trade_count = ?,
                    total_profit = ?,
                    win_rate = ?,
                    avg_profit_pct = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                new_count,
                new_profit,
                new_win_rate,
                new_profit / new_count if new_count > 0 else 0,
                datetime.utcnow().isoformat(),
                existing["id"]
            ))

            logger.info(f"Updated recipe '{recipe_name}' (count: {new_count}, WR: {new_win_rate:.1%})")
        else:
            # Create new recipe
            entry_conditions = {
                "pattern": pattern,
                "trade_type": trade_type,
                "notes": data.get("notes"),
            }

            cursor.execute("""
                INSERT INTO trade_recipes (
                    name, ticker_pattern, trade_type, time_window,
                    entry_conditions, source_trade_count, total_profit,
                    win_rate, avg_profit_pct
                ) VALUES (?, ?, ?, ?, ?, 1, ?, 1.0, ?)
            """, (
                recipe_name,
                ticker,
                trade_type,
                time_window,
                json.dumps(entry_conditions),
                data.get("profit_loss") or 0,
                data.get("profit_loss_pct") or 0,
            ))

            logger.info(f"Created new recipe: '{recipe_name}'")

        conn.commit()
        conn.close()

    def _classify_time_window(self, entry_time: Optional[str]) -> str:
        """Classify entry time into trading session."""
        if not entry_time:
            return "unknown"

        try:
            # Parse HH:MM format
            parts = entry_time.split(":")
            hour = int(parts[0])

            if hour < 10:
                return "open"
            elif hour < 12:
                return "morning"
            elif hour < 14:
                return "midday"
            elif hour < 15:
                return "afternoon"
            else:
                return "power_hour"
        except:
            return "unknown"

    def get_matching_recipes(
        self,
        ticker: str,
        trade_type: str,
        current_hour: int
    ) -> List[Dict]:
        """
        Find trade recipes that match current conditions.
        Used to boost confidence when entering trades.

        Returns:
            List of matching recipes with their stats
        """
        time_window = self._classify_time_window(f"{current_hour}:00")

        conn = get_connection()
        cursor = conn.cursor()

        # Find matching recipes
        cursor.execute("""
            SELECT *
            FROM trade_recipes
            WHERE is_active = 1
              AND (ticker_pattern = ? OR ticker_pattern = 'ANY')
              AND (trade_type = ? OR trade_type IS NULL)
              AND (time_window = ? OR time_window = 'any')
              AND source_trade_count >= 2
              AND win_rate >= 0.6
            ORDER BY win_rate DESC, source_trade_count DESC
            LIMIT 5
        """, (ticker, trade_type, time_window))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]


# Global instance
trade_extractor = TradeExtractor()
