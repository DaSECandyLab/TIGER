"""Fetch exact external sources and apply the recorded runtime patches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/source-lock.json"


def run(args, *, cwd, **kwargs):
    return subprocess.run(args, cwd=cwd, check=True, **kwargs)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_patch(source):
    if source.get("patch"):
        patch = ROOT / source["patch"]
        if sha256(patch) != source["patch_sha256"]:
            raise ValueError(f"Patch checksum differs: {patch}")
        return patch


def fetch(source, *, check_only):
    path = ROOT / source["path"]
    patch = verify_patch(source)
    if path.exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        if commit != source["commit"]:
            raise ValueError(f"Unexpected source revision at {path}: {commit}")
        if patch:
            run(["git", "apply", "--reverse", "--check", str(patch)], cwd=path)
        print(f"verified {source['name']}: {commit}", flush=True)
        return
    if check_only:
        raise FileNotFoundError(f"Fetch the pinned dependency first: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Populate a new directory before publishing it. Never reset an existing checkout.
    with tempfile.TemporaryDirectory(prefix=".source-", dir=path.parent) as temp:
        checkout = Path(temp) / "checkout"
        run(["git", "init", str(checkout)], cwd=ROOT)
        run(["git", "remote", "add", "origin", source["url"]], cwd=checkout)
        run(["git", "fetch", "--depth", "1", "origin", source["commit"]], cwd=checkout)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
        if patch:
            run(["git", "apply", "--check", str(patch)], cwd=checkout)
            run(["git", "apply", str(patch)], cwd=checkout)
        checkout.rename(path)


def patch_lmcache(source, python):
    verify_patch(source)
    code = 'from importlib.metadata import distribution; print(distribution("lmcache").version); print(distribution("lmcache").locate_file(""))'
    version, site = subprocess.check_output([python, "-c", code], text=True).strip().splitlines()
    if version != source["version"]:
        raise ValueError(f"Expected lmcache=={source['version']}, got {version}")
    site = Path(site)
    hashes = source["patched_files_sha256"]
    if all(sha256(site / name) == expected for name, expected in hashes.items()):
        print("LMCache runtime patches already match")
        return
    patch = ROOT / source["patch"]
    run(["patch", "--batch", "--forward", "-p1", "--dry-run", "-i", str(patch)], cwd=site)
    run(["patch", "--batch", "--forward", "-p1", "-i", str(patch)], cwd=site)
    if not all(sha256(site / name) == expected for name, expected in hashes.items()):
        raise ValueError("LMCache files differ after applying the pinned patch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing checkouts without network access")
    parser.add_argument("--only", nargs="+", choices=["duckdb", "extension-ci-tools", "vcpkg", "vllm"],
                        help="Fetch/verify only these dependencies (e.g. the three Flock build dependencies)")
    parser.add_argument("--patch-lmcache-python", help="Apply the pinned LMCache patch to this Python environment")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    if args.patch_lmcache_python:
        patch_lmcache(lock["lmcache"], args.patch_lmcache_python)
        return
    for source in lock["sources"]:
        if (args.only and source["name"] in args.only) or (not args.only and not source["optional"]):
            fetch(source, check_only=args.check)


if __name__ == "__main__":
    main()
