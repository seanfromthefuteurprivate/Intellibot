import json
import os
from datetime import datetime
from typing import List, Optional
from wsb_snake.signals.signal_types import Signal
from wsb_snake.utils.logger import log

STORE_DIR = "signal_logs"

def ensure_store_dir():
    """Ensure the signal store directory exists."""
    if not os.path.exists(STORE_DIR):
        os.makedirs(STORE_DIR)

def save_signal(signal: Signal) -> str:
    """
    Save a signal to the JSON ledger.
    Returns the filepath of the saved signal.
    """
    ensure_store_dir()
    
    timestamp_str = signal.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{signal.ticker}_{timestamp_str}.json"
    filepath = os.path.join(STORE_DIR, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(signal.to_dict(), f, indent=2)
        log.debug(f"Saved signal to {filepath}")
    except Exception as e:
        log.error(f"Failed to save signal: {e}")
        
    return filepath


def save_signals_batch(signals: List[Signal]) -> str:
    """
    Save multiple signals to a single batch file.
    Returns the filepath.
    """
    ensure_store_dir()
    
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_{timestamp_str}.json"
    filepath = os.path.join(STORE_DIR, filename)
    
    try:
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'count': len(signals),
            'signals': [s.to_dict() for s in signals],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        log.info(f"Saved {len(signals)} signals to {filepath}")
    except Exception as e:
        log.error(f"Failed to save signal batch: {e}")
        
    return filepath


def load_recent_signals(limit: int = 50) -> List[dict]:
    """
    Load the most recent signals from the store.
    """
    ensure_store_dir()
    
    files = []
    for f in os.listdir(STORE_DIR):
        if f.endswith('.json'):
            filepath = os.path.join(STORE_DIR, f)
            files.append((filepath, os.path.getmtime(filepath)))
    
    # Sort by modification time, newest first
    files.sort(key=lambda x: x[1], reverse=True)
    
    signals = []
    for filepath, _ in files[:limit]:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Handle both single signals and batches
                if 'signals' in data:
                    signals.extend(data['signals'])
                else:
                    signals.append(data)
        except Exception as e:
            log.warning(f"Failed to load {filepath}: {e}")
            
    return signals[:limit]
