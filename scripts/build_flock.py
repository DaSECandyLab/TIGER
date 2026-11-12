"""Build the bundled modified Flock against the pinned DuckDB 1.5.1 source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .bootstrap_sources import ROOT, LOCK, fetch


def verify_snapshot() -> dict:
    manifest = json.loads((ROOT / "flock/SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = ROOT / "flock" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundled Flock snapshot changed: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")))
    parser.add_argument("--system-deps", action="store_true", help="Use installed libcurl/nlohmann-json instead of vcpkg")
    parser.add_argument("--check-source", action="store_true", help="Only check the bundled source hashes")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    snapshot = verify_snapshot()
    if args.check_source:
        print(json.dumps({"status": "passed", "files": len(snapshot["files_sha256"])}))
        return
    lock = json.loads(LOCK.read_text())
    names = {"duckdb", "extension-ci-tools"}
    if not args.system_deps:
        names.add("vcpkg")
    for source in lock["sources"]:
        if source["name"] in names:
            fetch(source, check_only=False)
    flock = ROOT / "flock"
    build = flock / "build/release"
    flags = [
        "-DCMAKE_BUILD_TYPE=Release", "-DEXTENSION_STATIC_BUILD=ON",
        "-DBUILD_UNITTESTS=OFF", "-DBUILD_SHELL=OFF", "-DFLOCK_BUILD_TESTS=OFF",
        "-DENABLE_EXTENSION_AUTOLOADING=OFF", "-DENABLE_EXTENSION_AUTOINSTALL=OFF",
        "-DOVERRIDE_GIT_DESCRIBE=v1.5.1",
        f"-DDUCKDB_EXTENSION_CONFIGS={flock / 'extension_config.cmake'}",
    ]
    if not args.system_deps:
        subprocess.run([str(flock / "vcpkg/bootstrap-vcpkg.sh"), "-disableMetrics"], cwd=flock, check=True)
        flags += [f"-DCMAKE_TOOLCHAIN_FILE={flock / 'vcpkg/scripts/buildsystems/vcpkg.cmake'}",
                  f"-DVCPKG_MANIFEST_DIR={flock}"]
    subprocess.run(["cmake", "-S", str(flock / "duckdb"), "-B", str(build), *flags], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "flock_loadable_extension",
                    "--parallel", str(args.jobs)], check=True)
    extension = build / "extension/flock/flock.duckdb_extension"
    report = {"status": "built", "duckdb_version": "v1.5.1", "extension": str(extension),
              "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
              "source_head": snapshot["source_head"], "source_files": len(snapshot["files_sha256"])}
    (build / "flock_build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
