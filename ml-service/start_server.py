#!/usr/bin/env python
"""Start the ML service FastAPI server."""
import os

import uvicorn


# Ensure we're in the ml-service directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
