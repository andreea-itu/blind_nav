"""
Blind Navigation Assistant — Browser Camera Mode
Receives frames from teststayontrails.html via WebSocket instead of local webcam.

Run:  poetry run python run_web.py
"""
import runpy
runpy.run_module("blind.web_receiver", run_name="__main__")


# Serve the tests/ folder on port 8443 with a self-signed cert
# python -m http.server 8080
# http://localhost:8080/teststayontrails.html
