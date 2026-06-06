#!/usr/bin/env python3
"""Embedding tool entry point"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embed import main

if __name__ == "__main__":
    main()
