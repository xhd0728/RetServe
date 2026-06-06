#!/usr/bin/env python3
"""Indexing tool entry point"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.index import main

if __name__ == "__main__":
    main()
