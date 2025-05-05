"""
Run the Chicago Energy Performance Navigator dashboard.
"""
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the dashboard
from src.dashboard.main_dashboard import main

if __name__ == "__main__":
    # Run the dashboard
    main()