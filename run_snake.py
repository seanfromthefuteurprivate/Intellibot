#!/usr/bin/env python3
"""
WSB Snake - Run Script
Starts the market intelligence pipeline.
"""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wsb_snake.main import main

if __name__ == "__main__":
    main()
