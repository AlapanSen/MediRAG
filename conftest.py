"""
conftest.py — project root
Ensures src/ is on the Python path so all test files can import from src.*
without needing PYTHONPATH to be set manually. (SRS Section 17)
"""
import sys
import os

# Add the src/ directory to path so `from modules.faithfulness import ...` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# Also add project root so `import src` works
sys.path.insert(0, os.path.dirname(__file__))
