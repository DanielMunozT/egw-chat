#!/usr/bin/env python3
"""stop.py — Stop the Qdrant container.
Works on Linux, macOS, and Windows.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
CONTAINER_NAME = "qdrant-egw"


def load_env():
    """Load .env file into os.environ."""
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def main():
    load_env()

    force = "--force" in sys.argv or "-f" in sys.argv
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    # Check if QDRANT_URL points to a non-local instance
    host = urlparse(qdrant_url).hostname
    if host not in ("localhost", "127.0.0.1"):
        print(f"WARNING: QDRANT_URL points to a remote instance ({qdrant_url}).")
        print(f"This script only manages the local Docker container '{CONTAINER_NAME}'.")
        print("To stop a remote Qdrant, contact the administrator.")
        return

    # Check Docker
    docker = shutil.which("docker")
    if not docker:
        print(f"Qdrant container '{CONTAINER_NAME}' — Docker not found.")
        return

    # Check if container is running
    result = subprocess.run(
        [docker, "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    if CONTAINER_NAME not in result.stdout.strip().split("\n"):
        print(f"Qdrant container '{CONTAINER_NAME}' is not running.")
        return

    # Confirmation prompt
    if not force:
        print("WARNING: Other users may be connected to this Qdrant instance.")
        print(f"Container: {CONTAINER_NAME}")
        try:
            confirm = input("Stop Qdrant? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            return

    print("Stopping Qdrant...")
    subprocess.run([docker, "stop", CONTAINER_NAME], capture_output=True)
    subprocess.run([docker, "rm", CONTAINER_NAME], capture_output=True)
    print("Stopped. Data is preserved in .qdrant_storage/")


if __name__ == "__main__":
    main()
