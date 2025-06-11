#!/usr/bin/env python3
"""
Script to preprocess the downloaded dataset
Usage: python scripts/preprocess_data.py --visualize
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocess import main

if __name__ == "__main__":
    main()