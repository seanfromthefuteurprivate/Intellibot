#!/usr/bin/env python3
"""
Audit .env: report which expected keys are set (non-empty).
Does NOT print values. Use: python script/audit_env.py [path_to_env]
Default: .env in project root.
"""
import os
import sys

# Keys the app uses (config + scattered getenv)
REQUIRED = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "POLYGON_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
]
OPTIONAL_BUT_USED = [
    "ALPACA_BASE_URL",
    "ALPACA_LIVE_TRADING",
    "FINNHUB_API_KEY",
    "BENZINGA_API_KEY",
    "GEMINI_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "GOOGLE_DRIVE_FOLDER_ID",
    "GOOGLE_SERVICE_ACCOUNT",
    "SCREENSHOT_SCAN_INTERVAL",
    "WSB_SNAKE_DATA_DIR",
    "WSB_SNAKE_DB_PATH",
]
OPTIONAL_TUNING = [
    "RISK_MAX_DAILY_LOSS",
    "RISK_MAX_CONCURRENT_POSITIONS",
    "RISK_MAX_DAILY_EXPOSURE",
    "SCALP_TARGET_PCT",
    "SCALP_STOP_PCT",
    "SCALP_MAX_HOLD_MINUTES",
    "FRED_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "GEMINI_ENABLED",
]
ALL_KEYS = REQUIRED + OPTIONAL_BUT_USED + OPTIONAL_TUNING


def load_env(path: str) -> dict:
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                out[k] = v
    return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, ".env")
    env = load_env(env_path)
    missing_required = []
    missing_optional_used = []
    for k in REQUIRED:
        if not env.get(k) or env.get(k).startswith("your_"):
            missing_required.append(k)
    for k in OPTIONAL_BUT_USED:
        if not env.get(k) or env.get(k).startswith("your_"):
            missing_optional_used.append(k)
    print("=== ENV AUDIT (values not shown) ===\n")
    print("REQUIRED (trading + alerts + AI):")
    for k in REQUIRED:
        status = "set" if env.get(k) and not (env.get(k) or "").startswith("your_") else "MISSING"
        print(f"  {k}: {status}")
    print("\nOPTIONAL (used by features):")
    for k in OPTIONAL_BUT_USED:
        status = "set" if env.get(k) and not (env.get(k) or "").startswith("your_") else "missing"
        print(f"  {k}: {status}")
    print("\nOPTIONAL (tuning):")
    for k in OPTIONAL_TUNING:
        status = "set" if env.get(k) else "not set"
        print(f"  {k}: {status}")
    print()
    if missing_required:
        print("FAIL: Missing required keys:", ", ".join(missing_required))
        sys.exit(1)
    print("OK: All required keys present.")
    if missing_optional_used:
        print("Note: Some optional keys missing (features may be limited).")
    sys.exit(0)


if __name__ == "__main__":
    main()
