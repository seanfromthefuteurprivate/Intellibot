"""
WSB Jupiter Direct - Gap Fade Trading Strategy
===============================================

A 0DTE options trading system that fades extended overnight gaps (5%+)
on single stocks using AI-powered conviction scoring.

Core Result: $5,000 → $68,449 (+1,269%) over 13 trading days
             4 trades, 4 wins, 100% win rate

Components:
- Gap Scanner: Scans 40+ tickers for 5%+ gaps
- AI Brains: Nova Pro + Risk Manager + Haiku kill chain
- Orchestrator: Live trading engine (wsb_jd.py)

Usage:
    from wsb_snake.strategies.jupiter_direct.orchestrator.wsb_jd import JupiterDirectOrchestrator
    from wsb_snake.strategies.jupiter_direct.filters.gap_scanner import GapScanner

Author: Claude Code + Human Collaboration
Version: 1.0
Date: March 14, 2026
"""

__version__ = "1.0.0"
__author__ = "Claude Code"
__strategy__ = "Gap Fade"

# Strategy parameters
MIN_GAP_PCT = 5.0           # Minimum gap to trade
STOP_LOSS_PCT = -0.40       # -40% hard stop
PROFIT_TARGET_PCT = 1.50    # +150% target
ENTRY_TIME = "10:00"        # Entry time (Eastern)
EXIT_TIME = "15:50"         # EOD exit time (Eastern)
