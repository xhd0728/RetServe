#!/usr/bin/env python3
"""Retrieval service entry point"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_loader import config_loader
from src.ret_serve import create_application, main

# Expose a module-level app for uvicorn workers.
try:
    settings = config_loader.load_service_settings("serve")
    app = create_application(settings)
except Exception as e:
    print(f"Failed to create app: {e}")
    import traceback

    traceback.print_exc()
    app = None

if __name__ == "__main__":
    main()
