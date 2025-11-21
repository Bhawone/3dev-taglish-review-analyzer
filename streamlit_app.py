"""
Streamlit Cloud entry point for Taglish Review Analyzer.

This file serves as the main entry point for Streamlit Cloud deployment.
It imports and executes the main app from app/ui/app.py.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and execute the main app
from app.ui.app import main

# Call main() to run the Streamlit app
main()

