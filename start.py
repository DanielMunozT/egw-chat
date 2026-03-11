#!/usr/bin/env python3
"""start.py — Connect to existing Qdrant or start a new container.
Works on Linux, macOS, and Windows.

Priority:
  1. Check if Qdrant is already reachable at QDRANT_URL
  2. Check if Docker container 'qdrant-egw' exists
  3. Start a new container as last resort
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONTAINER_NAME = "qdrant-egw"
QDRANT_IMAGE = "qdrant/qdrant:v1.16.2"
DEFAULT_PORT = 6333


def load_env():
    """Load .env file into os.environ."""
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def qdrant_ready(url: str) -> bool:
    """Check if Qdrant is reachable."""
    try:
        urllib.request.urlopen(f"{url}/readyz", timeout=5)
        return True
    except Exception:
        return False


def get_collections(url: str) -> list[str]:
    """Get collection names from Qdrant."""
    try:
        resp = urllib.request.urlopen(f"{url}/collections", timeout=5)
        data = json.loads(resp.read())
        return [c["name"] for c in data.get("result", {}).get("collections", [])]
    except Exception:
        return []


def report_collections(url: str):
    """Print available collections."""
    collections = get_collections(url)
    if collections:
        print("Available collections:")
        for name in collections:
            lang = name.replace("egw_corpus_", "") if name.startswith("egw_corpus_") else name
            print(f"  - {name} ({lang})")


def update_env(url: str):
    """Update QDRANT_URL in .env file."""
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        lines = env_file.read_text().splitlines()
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("QDRANT_URL="):
                lines[i] = f"QDRANT_URL={url}"
                found = True
                break
        if not found:
            lines.append(f"QDRANT_URL={url}")
        env_file.write_text("\n".join(lines) + "\n")
    else:
        env_file.write_text(f"QDRANT_URL={url}\n")


def docker_available() -> str | None:
    """Return docker path if available and daemon is running, else None."""
    docker = shutil.which("docker")
    if not docker:
        return None
    try:
        subprocess.run([docker, "info"], capture_output=True, check=True)
        return docker
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def docker_run(docker: str, *args, **kwargs) -> subprocess.CompletedProcess:
    """Run a docker command."""
    return subprocess.run([docker, *args], **kwargs)


def container_running(docker: str) -> bool:
    """Check if the container is currently running."""
    result = docker_run(docker, "ps", "--format", "{{.Names}}", capture_output=True, text=True)
    return CONTAINER_NAME in result.stdout.strip().split("\n")


def container_exists(docker: str) -> bool:
    """Check if the container exists (running or stopped)."""
    result = docker_run(docker, "ps", "-a", "--format", "{{.Names}}", capture_output=True, text=True)
    return CONTAINER_NAME in result.stdout.strip().split("\n")


def get_container_port(docker: str) -> str | None:
    """Get the host port mapped to container port 6333."""
    result = docker_run(docker, "port", CONTAINER_NAME, "6333/tcp", capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        # Output: "0.0.0.0:6333" or ":::6333"
        return result.stdout.strip().split(":")[-1]
    return None


def wait_for_qdrant(url: str, timeout: int = 30):
    """Wait for Qdrant to become ready."""
    for _ in range(timeout):
        if qdrant_ready(url):
            return True
        time.sleep(1)
    return False


def find_free_port(start: int = DEFAULT_PORT, end: int = 6500) -> int:
    """Find a free TCP port."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    print(f"ERROR: No free port in range {start}-{end}")
    sys.exit(1)


def restore_snapshots(qdrant_url: str):
    """Auto-restore snapshots if they exist and collections don't."""
    snapshot_dir = SCRIPT_DIR / "snapshots"
    if not snapshot_dir.is_dir():
        return

    existing = set(get_collections(qdrant_url))

    for snap in sorted(snapshot_dir.glob("*.snapshot")):
        collection_name = snap.stem  # e.g., egw_corpus_en.snapshot -> egw_corpus_en
        if collection_name in existing:
            print(f"Collection '{collection_name}' already exists, skipping restore.")
            continue

        print(f"Restoring snapshot: {snap.name} -> collection '{collection_name}'...")
        try:
            # Use curl for multipart upload (urllib doesn't handle this well)
            curl = shutil.which("curl")
            if curl:
                result = subprocess.run([
                    curl, "-sf", "-X", "POST",
                    f"{qdrant_url}/collections/{collection_name}/snapshots/upload",
                    "-H", "Content-Type: multipart/form-data",
                    "-F", f"snapshot=@{snap}",
                ], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  WARNING: Failed to restore {snap.name}: {result.stderr}")
                    continue
            else:
                # Pure Python multipart upload fallback
                _upload_snapshot_python(qdrant_url, collection_name, snap)
            print(f"  Restored '{collection_name}'")
        except Exception as e:
            print(f"  WARNING: Failed to restore {snap.name}: {e}")


def _upload_snapshot_python(qdrant_url: str, collection_name: str, snap_path: Path):
    """Upload a snapshot using pure Python (no curl dependency)."""
    import uuid
    boundary = uuid.uuid4().hex
    url = f"{qdrant_url}/collections/{collection_name}/snapshots/upload"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="snapshot"; filename="{snap_path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()

    file_data = snap_path.read_bytes()
    data = body + file_data + tail

    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    urllib.request.urlopen(req, timeout=600)


def print_done(qdrant_url: str):
    """Print final status."""
    report_collections(qdrant_url)
    print()
    print(f"export QDRANT_URL={qdrant_url}")
    print()
    print('To search: python scripts/search.py "your query"')


def main():
    os.chdir(SCRIPT_DIR)
    load_env()

    qdrant_url = os.environ.get("QDRANT_URL", f"http://localhost:{DEFAULT_PORT}")
    data_dir = os.environ.get("EGW_DATA_DIR", str(SCRIPT_DIR / ".qdrant_storage"))

    print("=== EGW Research — Starting Qdrant ===")

    # Phase 1: Already reachable?
    if qdrant_ready(qdrant_url):
        print(f"Qdrant already running at {qdrant_url}")
        print_done(qdrant_url)
        return

    # Phases 2 & 3 require Docker
    docker = docker_available()
    if not docker:
        docker_path = shutil.which("docker")
        if not docker_path:
            print(f"ERROR: Qdrant is not reachable at {qdrant_url} and Docker is not installed.")
            print("Either:")
            print("  - Set QDRANT_URL in .env to point to a running Qdrant instance, or")
            print("  - Install Docker: https://docs.docker.com/get-docker/")
            sys.exit(1)
        else:
            print(f"ERROR: Qdrant is not reachable at {qdrant_url} and Docker daemon is not running.")
            print("Start Docker and try again.")
            sys.exit(1)

    # Phase 2: Container already running?
    if container_running(docker):
        port = get_container_port(docker)
        qdrant_url = f"http://localhost:{port}"
        print(f"Container '{CONTAINER_NAME}' already running on port {port}")
        update_env(qdrant_url)
        print_done(qdrant_url)
        return

    # Container exists but stopped?
    if container_exists(docker):
        print(f"Restarting stopped container '{CONTAINER_NAME}'...")
        docker_run(docker, "start", CONTAINER_NAME, capture_output=True)
        port = get_container_port(docker)
        qdrant_url = f"http://localhost:{port}"
        if not wait_for_qdrant(qdrant_url):
            print("ERROR: Qdrant failed to start after 30s")
            docker_run(docker, "logs", CONTAINER_NAME, "--tail", "5")
            sys.exit(1)
        print(f"Qdrant ready on port {port}")
        update_env(qdrant_url)
        print_done(qdrant_url)
        return

    # Phase 3: Start new container
    docker_run(docker, "rm", "-f", CONTAINER_NAME, capture_output=True)

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    port = find_free_port()
    qdrant_url = f"http://localhost:{port}"

    print(f"Starting Qdrant on port {port}...")
    docker_run(
        docker, "run", "-d",
        "--name", CONTAINER_NAME,
        "--restart", "unless-stopped",
        "-p", f"{port}:6333",
        "-v", f"{data_dir}:/qdrant/storage",
        QDRANT_IMAGE,
        capture_output=True, check=True,
    )

    if not wait_for_qdrant(qdrant_url):
        print("ERROR: Qdrant failed to start after 30s")
        docker_run(docker, "logs", CONTAINER_NAME, "--tail", "5")
        sys.exit(1)
    print("Qdrant ready.")

    restore_snapshots(qdrant_url)
    update_env(qdrant_url)
    print_done(qdrant_url)


if __name__ == "__main__":
    main()
