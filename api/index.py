"""Vercel Python entrypoint.

Vercel's Python runtime loads the top-level name `app` from a file in /api and
serves it as an ASGI application; `api/index.py` is served at `/api`, and the
rewrites in vercel.json send every `/api/*` path (plus `/analyze-video`) here.
The FastAPI routes already live under those paths, so the application is served
unchanged - this file adds an entrypoint, it does not add a second API.

The repository root is put on sys.path because a function is invoked from the
bundle root rather than from this directory, and `backend.app.main` imports
`ml`, `configuration` and `utilities` as top-level packages.

Deployment-specific behaviour (model paths, upload cap, CORS, whether background
jobs are available) is read from environment variables by
configuration/settings.py. Nothing is hardcoded here.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app.main import app  # noqa: E402,F401

__all__ = ["app"]
