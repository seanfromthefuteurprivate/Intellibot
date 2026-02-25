"""
WSB Snake Predator AI Stack v2.1

Optimized multi-layer AI intelligence pipeline for 0DTE options trading.
Integrated with HYDRA for market structure intelligence.

Layers:
- L0: Speed Filter (Nova Micro) - Kill obvious losers in <80ms
- L1: Vision (GPT-4o) - Chart pattern detection (15%)
- L2: Semantic (Titan V2) - News similarity matching (10%)
- LH: HYDRA Intelligence - GEX/Flow/DarkPool/Sequence (20%)
- L4: Adversarial (Claude Sonnet) - Thesis destroyer (15%)
- L5: Contrarian (DeepSeek) - Trap detection (10%)
- L6: Strategy DNA - Your historical wins (25%)
- L12: Synthesis (Claude Sonnet) - Final arbiter (5%)

Total: 100% weighted conviction score
"""

from .predator_stack_v2 import PredatorStackV2, PredatorVerdict
from .bedrock_client import BedrockClient

__all__ = ['PredatorStackV2', 'PredatorVerdict', 'BedrockClient']
