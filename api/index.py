"""
Vercel Python runtime entry point. Vercel auto-detects an ASGI app named
`app` in api/*.py and wraps it as a serverless function -- no adapter code
needed, unlike some other platforms (e.g. AWS Lambda needs Mangum).

This just re-exports the real app from server/app.py so there's a single
source of truth for the routes -- don't duplicate endpoint logic here.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server.app import app  # noqa: F401  (re-exported for Vercel's detection)
