#!/usr/bin/env python3
"""
SWARM AUDIT: 12 Legendary Investor Personas Review the System

Each persona independently evaluates the trading system and provides:
1. Single deadliest weakness
2. Highest-ROI improvement
3. Recommended position sizing strategy
4. What to ADD
5. What to REMOVE
6. Confidence $5K → $50K in one week
"""
import os
import json
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# 12 Legendary Investor Personas
PERSONAS = {
    "druckenmiller": {
        "name": "Stanley Druckenmiller",
        "style": "Macro momentum trader. Follow trends with size. When you see it, bet big.",
        "philosophy": "It's not whether you're right or wrong, but how much money you make when you're right.",
    },
    "burry": {
        "name": "Michael Burry",
        "style": "Deep value contrarian. Find what everyone else misses. Fade extremes.",
        "philosophy": "I focus on limiting downside rather than chasing upside. The edge comes from seeing what others ignore.",
    },
    "ackman": {
        "name": "Bill Ackman",
        "style": "Activist with conviction. Concentrated positions with catalysts.",
        "philosophy": "Invest in what you understand and be willing to bet heavily when the odds favor you.",
    },
    "graham": {
        "name": "Benjamin Graham",
        "style": "Value investing father. Margin of safety above all. Never overpay.",
        "philosophy": "In the short run, the market is a voting machine. In the long run, it's a weighing machine.",
    },
    "soros": {
        "name": "George Soros",
        "style": "Reflexivity trader. Markets create their own reality. Exploit feedback loops.",
        "philosophy": "When I see a bubble forming, I rush to buy, adding fuel to the fire.",
    },
    "simons": {
        "name": "Jim Simons",
        "style": "Quantitative systematic. Statistical edge, high frequency, no emotion.",
        "philosophy": "We never override the models. Gut feelings are irrelevant.",
    },
    "dalio": {
        "name": "Ray Dalio",
        "style": "Risk parity. Diversify. Understand the machine. Principles-based decisions.",
        "philosophy": "Pain plus reflection equals progress. The economy is a machine.",
    },
    "lynch": {
        "name": "Peter Lynch",
        "style": "Growth at reasonable price. Know what you own. Ten-baggers in plain sight.",
        "philosophy": "Know what you own, and know why you own it.",
    },
    "tudor_jones": {
        "name": "Paul Tudor Jones",
        "style": "Technical macro. Price is truth. Cut losers fast, let winners run.",
        "philosophy": "The most important rule of trading is to play great defense, not offense.",
    },
    "icahn": {
        "name": "Carl Icahn",
        "style": "Aggressive activist. Force change. Create your own catalysts.",
        "philosophy": "In life and business, there are two cardinal sins: impatience and laziness.",
    },
    "buffett": {
        "name": "Warren Buffett",
        "style": "Patient value compounder. Wonderful companies at fair prices.",
        "philosophy": "Be fearful when others are greedy and greedy when others are fearful.",
    },
    "livermore": {
        "name": "Jesse Livermore",
        "style": "Pure tape reader. Pyramids into winners. Ride the trend to the end.",
        "philosophy": "It was never my thinking that made big money for me. It was always my sitting.",
    },
}

SYSTEM_DESCRIPTION = """You are reviewing a SPY 0DTE automated trading system. It has:
- GEX-based signal detection (gamma exposure flip points)
- HYDRA multi-asset confirmation (predator dashboard)
- 12-specialist debate engine (bull/bear adversarial)
- Trailing stop ladder up to 500% (from -15% to +500%)
- Momentum pyramiding (add 50% at +30% profit)
- VIX-adaptive stops (wider stops in high vol)
- Time-of-day aggression multipliers (1.8x Power Hour)
- Gamma squeeze detector (auto-trigger BERSERKER mode)
- Overnight gap detector (pre-market edge)
- Compounding risk governor with win-streak sizing
- Semantic memory that learns from past trades (embeddings)
- Trade graph pattern matching (GNN similarity)
- 79 historical Alpaca trades loaded for learning

The account starts at $5K with $10-15K margin. The goal is maximum compound growth trading SPY 0DTE options daily."""

def query_persona(persona_id: str, persona_info: Dict) -> Optional[Dict]:
    """Query a single persona for their audit."""
    prompt = f"""You are {persona_info['name']}.
Style: {persona_info['style']}
Philosophy: "{persona_info['philosophy']}"

{SYSTEM_DESCRIPTION}

From YOUR specific investment philosophy, answer these 6 questions. Be specific and actionable:

1. DEADLIEST WEAKNESS: What is the single deadliest weakness in this system?

2. HIGHEST-ROI IMPROVEMENT: What is the single highest-ROI improvement you would make?

3. POSITION SIZING: What position sizing strategy would YOU use? Be specific (e.g., % per trade, scaling rules).

4. ADD: What would you ADD that doesn't exist yet?

5. REMOVE: What would you REMOVE or disable?

6. CONFIDENCE: What's your confidence this system compounds $5K to $50K in one week? Give a percentage and explain why.

Return your response as JSON:
{{
    "persona": "{persona_id}",
    "weakness": "...",
    "improvement": "...",
    "position_sizing": "...",
    "add": "...",
    "remove": "...",
    "confidence_pct": 0-100,
    "confidence_reasoning": "..."
}}"""

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "max_tokens": 1000,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        txt = r.json()["choices"][0]["message"]["content"]
        txt = txt.replace("```json", "").replace("```", "").strip()
        return json.loads(txt)
    except Exception as e:
        print(f"Error querying {persona_id}: {e}")
        return None


def run_full_audit() -> Dict:
    """Run audit with all 12 personas."""
    print("=" * 60)
    print("SWARM AUDIT: 12 Legendary Investor Personas")
    print("=" * 60)

    results = []
    for persona_id, persona_info in PERSONAS.items():
        print(f"\n[{persona_id.upper()}] Querying {persona_info['name']}...")
        result = query_persona(persona_id, persona_info)
        if result:
            results.append(result)
            print(f"  → Confidence: {result.get('confidence_pct', 'N/A')}%")
            print(f"  → Weakness: {result.get('weakness', 'N/A')[:80]}...")
        else:
            print(f"  → FAILED")

    # Find consensus (3+ personas agree)
    print("\n" + "=" * 60)
    print("CONSENSUS ANALYSIS (3+ personas agree)")
    print("=" * 60)

    # Aggregate by category
    weaknesses = [r.get("weakness", "") for r in results]
    improvements = [r.get("improvement", "") for r in results]
    adds = [r.get("add", "") for r in results]
    removes = [r.get("remove", "") for r in results]
    confidences = [r.get("confidence_pct", 0) for r in results if r.get("confidence_pct")]

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    print(f"\nAVERAGE CONFIDENCE: {avg_confidence:.1f}%")
    print(f"CONFIDENCE RANGE: {min(confidences) if confidences else 0}% - {max(confidences) if confidences else 0}%")

    # Save results
    output = {
        "persona_results": results,
        "avg_confidence": avg_confidence,
        "min_confidence": min(confidences) if confidences else 0,
        "max_confidence": max(confidences) if confidences else 0,
    }

    with open("swarm_audit_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to swarm_audit_results.json")

    return output


def print_detailed_results(results: Dict):
    """Print detailed results from all personas."""
    print("\n" + "=" * 60)
    print("DETAILED PERSONA RESPONSES")
    print("=" * 60)

    for r in results.get("persona_results", []):
        persona = r.get("persona", "unknown").upper()
        print(f"\n{'='*40}")
        print(f"{persona}")
        print(f"{'='*40}")
        print(f"WEAKNESS: {r.get('weakness', 'N/A')}")
        print(f"IMPROVEMENT: {r.get('improvement', 'N/A')}")
        print(f"POSITION SIZING: {r.get('position_sizing', 'N/A')}")
        print(f"ADD: {r.get('add', 'N/A')}")
        print(f"REMOVE: {r.get('remove', 'N/A')}")
        print(f"CONFIDENCE: {r.get('confidence_pct', 'N/A')}% - {r.get('confidence_reasoning', 'N/A')}")


if __name__ == "__main__":
    results = run_full_audit()
    print_detailed_results(results)
