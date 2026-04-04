#!/usr/bin/env python
"""Start the ML service FastAPI server."""
import uvicorn
import os

# Ensure we're in the ml-service directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")
