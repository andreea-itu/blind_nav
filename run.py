"""
Blind Navigation Assistant — Entry Point
Run:  poetry run python run.py
"""
import runpy
runpy.run_module("blind_nav.detector", run_name="__main__")


# Run locally the model
# poetry run python run.py