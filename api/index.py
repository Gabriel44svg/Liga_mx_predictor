# api/index.py - Entry point for Vercel
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

# This is the entry point for Vercel
handler = app
