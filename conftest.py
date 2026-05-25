# conftest.py – pytest configuration
import sys
import os

# Ensure src/ is on the import path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
