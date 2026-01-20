#!/usr/bin/env python3
"""Retrieval service entry point"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ret_serve import main, create_application
from src.config_loader import config_loader

# Create app at module level for uvicorn workers
try:
    # Load config
    settings = config_loader.load_service_settings("serve")
    # Create app instance
    app = create_application(settings)
except Exception as e:
    print(f"Failed to create app: {e}")
    import traceback
    traceback.print_exc()
    app = None

if __name__ == "__main__":
    main()
