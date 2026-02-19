#!/usr/bin/env python3
"""
run_pipeline.py
===============
Thin entry point for the Olist Seller Churn Analysis pipeline.

Usage:
    python scripts/run_pipeline.py

    # Or via Makefile:
    make run-pipeline

All configuration is read from .env (see .env.example).
All outputs are written to the paths defined in config/settings.py.
"""

import sys
import os

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import main

if __name__ == "__main__":
    main()
