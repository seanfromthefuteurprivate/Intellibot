"""
FRESH BLOOD — GO / NO-GO Evaluator

Run this Friday 4:30 PM ET to make the deployment decision.

GO conditions (need at least 2 of 3):
- At least 2 paper trades placed during the week
- Win rate >= 50% on paper trades
- No critical bugs or crashes remaining

NO-GO conditions (any one kills it):
- 0 paper trades all week (even after lowering thresholds Wednesday)
- Win rate < 30% on 3+ trades
- Alpaca options API unreliable or returning bad data
"""

import json
import glob
import os
from datetime import datetime
from typing import Dict, List
import logging

log = logging.getLogger(__name__)


def load_all_week_trades() -> List[Dict]:
    """Load all trades from this week's report files."""
    trades = []

    # Check for daily result files
    for filename in glob.glob("report_2026-03-*.txt"):
        # Extract trades from report... simplified for now
        pass

    # Also check the main trades file
    try:
        with open("fresh_blood_trades.json", "r") as f:
            trades.extend(json.load(f))
    except FileNotFoundError:
        pass

    # Check for any gap_fade_results files
    for filename in glob.glob("gap_fade_results*.json"):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    trades.extend(data)
        except:
            pass

    return trades


def check_api_health() -> Dict:
    """Check Alpaca options API health."""
    import requests
    from config import ALPACA_KEY, ALPACA_SECRET, ALPACA_DATA_URL

    try:
        resp = requests.get(
            f"{ALPACA_DATA_URL}/v1beta1/options/snapshots?symbols=SPY",
            headers={
                "APCA-API-KEY-ID": ALPACA_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET
            },
            params={"feed": "indicative", "limit": 1}
        )

        if resp.status_code == 200:
            return {"healthy": True, "message": "Options API responding"}
        else:
            return {"healthy": False, "message": f"API returned {resp.status_code}"}
    except Exception as e:
        return {"healthy": False, "message": str(e)}


def evaluate_go_nogo() -> Dict:
    """
    Evaluate GO/NO-GO criteria.

    Returns decision and reasoning.
    """
    log.info("=" * 60)
    log.info("GO / NO-GO EVALUATION")
    log.info("=" * 60)
    log.info(f"Date: {datetime.now()}")

    result = {
        "timestamp": datetime.now().isoformat(),
        "decision": None,
        "criteria_met": [],
        "criteria_failed": [],
        "details": {}
    }

    # Load week's trades
    trades = load_all_week_trades()
    result["details"]["total_trades"] = len(trades)

    winners = [t for t in trades if t.get("pnl_pct", 0) > 0]
    result["details"]["winners"] = len(winners)

    win_rate = len(winners) / len(trades) * 100 if trades else 0
    result["details"]["win_rate"] = win_rate

    log.info(f"\nTRADE SUMMARY:")
    log.info(f"  Total trades: {len(trades)}")
    log.info(f"  Winners: {len(winners)}")
    log.info(f"  Win rate: {win_rate:.0f}%")

    # Check API health
    api_health = check_api_health()
    result["details"]["api_health"] = api_health
    log.info(f"\nAPI HEALTH: {api_health['message']}")

    # === EVALUATE CRITERIA ===

    log.info(f"\n" + "=" * 60)
    log.info("CRITERIA EVALUATION")
    log.info("=" * 60)

    # GO Criteria (need 2 of 3)
    go_criteria = []

    # 1. At least 2 trades
    if len(trades) >= 2:
        go_criteria.append("2+ trades placed")
        result["criteria_met"].append("2+ trades placed")
        log.info("  ✅ 2+ trades placed")
    else:
        result["criteria_failed"].append("Less than 2 trades")
        log.info("  ❌ Less than 2 trades")

    # 2. Win rate >= 50%
    if len(trades) >= 1 and win_rate >= 50:
        go_criteria.append("Win rate >= 50%")
        result["criteria_met"].append("Win rate >= 50%")
        log.info(f"  ✅ Win rate {win_rate:.0f}% >= 50%")
    else:
        result["criteria_failed"].append("Win rate < 50%")
        log.info(f"  ❌ Win rate {win_rate:.0f}% < 50%")

    # 3. API healthy
    if api_health["healthy"]:
        go_criteria.append("API healthy")
        result["criteria_met"].append("API healthy")
        log.info("  ✅ API healthy")
    else:
        result["criteria_failed"].append("API unhealthy")
        log.info("  ❌ API unhealthy")

    # === NO-GO CONDITIONS (any one kills it) ===

    nogo_triggered = False

    # 1. Zero trades
    if len(trades) == 0:
        nogo_triggered = True
        result["criteria_failed"].append("FATAL: Zero trades all week")
        log.info("  🚫 FATAL: Zero trades all week")

    # 2. Win rate < 30% on 3+ trades
    if len(trades) >= 3 and win_rate < 30:
        nogo_triggered = True
        result["criteria_failed"].append(f"FATAL: Win rate {win_rate:.0f}% < 30% on 3+ trades")
        log.info(f"  🚫 FATAL: Win rate {win_rate:.0f}% < 30% on 3+ trades")

    # 3. API dead
    if not api_health["healthy"]:
        nogo_triggered = True
        result["criteria_failed"].append("FATAL: API unhealthy")
        log.info("  🚫 FATAL: API unhealthy")

    # === DECISION ===

    log.info(f"\n" + "=" * 60)
    log.info("DECISION")
    log.info("=" * 60)

    if nogo_triggered:
        result["decision"] = "NO-GO"
        log.info("\n🚫 DECISION: NO-GO")
        log.info("   At least one fatal condition triggered.")
        log.info("   Do NOT deploy to live. Fix issues or kill strategy.")
    elif len(go_criteria) >= 2:
        result["decision"] = "GO"
        log.info("\n✅ DECISION: GO")
        log.info(f"   {len(go_criteria)}/3 GO criteria met.")
        log.info("   Deploy to LIVE on Monday with $500 position size.")
    else:
        result["decision"] = "HOLD"
        log.info("\n⏸️ DECISION: HOLD")
        log.info(f"   Only {len(go_criteria)}/3 GO criteria met.")
        log.info("   Need more data. Continue paper trading.")

    # Save decision
    with open("go_nogo_decision.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"\nDecision saved to go_nogo_decision.json")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    evaluate_go_nogo()
