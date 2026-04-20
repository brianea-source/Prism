"""
tests/conftest.py
-----------------
Shared pytest configuration and fixtures for PRISM test suite.
"""

import sys
from pathlib import Path

# Ensure repo root is on the Python path for all tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
