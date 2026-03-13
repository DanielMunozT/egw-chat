#!/usr/bin/env python3
"""setup.py — One-time setup for EGW Research.
Works on Linux, macOS, and Windows.

Usage:
    python setup.py                     # Setup only (no language download)
    python setup.py --lang en           # Setup + download English data
    python setup.py --lang en,es        # Setup + download English + Spanish
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PACKAGE_URL = "https://munoz.tplinkdns.com/egw/packages/egw-research-{lang}.tar.gz"
AVAILABLE_LANGS = ["en", "es", "pt"]


def load_env():
    """Load .env file into os.environ."""
    env_file = SCRIPT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def check_python():
    """Verify Python 3.10+."""
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ is required.")
        print("Install from https://python.org or your package manager.")
        sys.exit(1)
    print(f"[OK] Python: {sys.version.split()[0]}")


def check_docker(qdrant_url: str):
    """Check Docker availability (conditional — not needed if Qdrant is reachable)."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{qdrant_url}/readyz", timeout=5)
        print(f"[OK] Qdrant already available at {qdrant_url} — Docker is not required")
        return
    except Exception:
        pass

    docker = shutil.which("docker")
    if not docker:
        print("WARNING: Docker is not installed.")
        print("  Docker is needed to run Qdrant locally.")
        print("  If you will connect to a shared Qdrant instance, set QDRANT_URL in .env and Docker is optional.")
        print("  Install from https://docs.docker.com/get-docker/")
        return

    try:
        subprocess.run([docker, "info"], capture_output=True, check=True)
        result = subprocess.run([docker, "--version"], capture_output=True, text=True)
        print(f"[OK] Docker: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: Docker is installed but the daemon is not running.")
        print("  Start Docker Desktop or run: sudo systemctl start docker")
        print("  If you will connect to a shared Qdrant instance, set QDRANT_URL in .env and Docker is optional.")


def setup_venv():
    """Create virtual environment if it doesn't exist."""
    venv_dir = SCRIPT_DIR / "venv"
    if not venv_dir.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    print("[OK] Virtual environment: venv/")
    return venv_dir


def get_pip(venv_dir: Path) -> str:
    """Get the pip executable from the venv."""
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "pip")
    return str(venv_dir / "bin" / "pip")


def get_python(venv_dir: Path) -> str:
    """Get the python executable from the venv."""
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "python")
    return str(venv_dir / "bin" / "python")


def install_deps(venv_dir: Path):
    """Install Python dependencies."""
    python = get_python(venv_dir)
    req = SCRIPT_DIR / "requirements.txt"
    print("Installing Python dependencies...")
    # Upgrade pip (best-effort — may fail on Windows if pip is locked)
    subprocess.run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
                   capture_output=True)
    subprocess.run([python, "-m", "pip", "install", "--quiet", "-r", str(req)], check=True)
    print("[OK] Dependencies installed")


def setup_env_file():
    """Create .env from template if it doesn't exist."""
    env_file = SCRIPT_DIR / ".env"
    example = SCRIPT_DIR / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copy2(example, env_file)
        print("[OK] Created .env from template")
    elif env_file.exists():
        print("[OK] .env file exists")


def download_embedding_model(venv_dir: Path):
    """Pre-download the embedding model."""
    python = get_python(venv_dir)
    print("Downloading embedding model (first run only, ~2GB)...")
    subprocess.run([
        python, "-c",
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('BAAI/bge-m3')",
    ], check=True, capture_output=True)
    print("[OK] Embedding model ready")


def download_language(lang: str):
    """Download and extract a language data package (snapshots + books)."""
    url = PACKAGE_URL.format(lang=lang)
    print(f"Downloading {lang} language package...")
    print(f"  URL: {url}")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Download with progress
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=300)
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)
        print()

        # Extract only snapshots/ and books/ from the tarball
        print(f"  Extracting data for '{lang}'...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Tarball has top-level dir (e.g., egw-research/snapshots/...)
                # Strip the first component and only extract snapshots/ and books/
                parts = member.name.split("/", 1)
                if len(parts) < 2:
                    continue
                rel_path = parts[1]
                if not (rel_path.startswith("snapshots/") or rel_path.startswith("books/")):
                    continue
                # Rewrite the member name to extract into SCRIPT_DIR
                member.name = rel_path
                tar.extract(member, path=str(SCRIPT_DIR))

        print(f"[OK] Language '{lang}' data installed")

    except urllib.error.HTTPError as e:
        print(f"ERROR: Failed to download {lang} package: HTTP {e.code}")
        if e.code == 404:
            print(f"  Package not found. Available languages: {', '.join(AVAILABLE_LANGS)}")
    except Exception as e:
        print(f"ERROR: Failed to download {lang} package: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="EGW Research — Setup")
    parser.add_argument("--lang", help=f"Language(s) to download, comma-separated ({', '.join(AVAILABLE_LANGS)})")
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)
    load_env()

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    print("=== EGW Research — Setup ===")
    print()

    check_python()
    check_docker(qdrant_url)
    venv_dir = setup_venv()
    install_deps(venv_dir)
    setup_env_file()
    download_embedding_model(venv_dir)

    # Download language packages if requested
    if args.lang:
        langs = [l.strip() for l in args.lang.split(",")]
        print()
        for lang in langs:
            if lang not in AVAILABLE_LANGS:
                print(f"WARNING: Unknown language '{lang}'. Available: {', '.join(AVAILABLE_LANGS)}")
                continue
            download_language(lang)

    print()
    print("=== Setup complete ===")
    print("Next steps:")

    import urllib.request as ur
    try:
        ur.urlopen(f"{qdrant_url}/readyz", timeout=5)
        print(f"  Qdrant is already running at {qdrant_url}")
        print('  Search: python scripts/search.py "your query"')
    except Exception:
        print("  1. Run: python start.py")
        print('  2. Search: python scripts/search.py "your query"')

    if not args.lang:
        snapshots = SCRIPT_DIR / "snapshots"
        if not snapshots.exists() or not list(snapshots.glob("*.snapshot")):
            print()
            print("  No language data installed yet. To download a language package:")
            print("    python setup.py --lang en")
            print(f"    Available languages: {', '.join(AVAILABLE_LANGS)}")


if __name__ == "__main__":
    main()
