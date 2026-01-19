#!/usr/bin/env python3
"""Indexing tool entry point"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.index import main

if __name__ == "__main__":
    main()
