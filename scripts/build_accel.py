"""Check and build the optional C++/Cython acceleration extensions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys


def _find_command(names):
    for name in names:
        path = shutil.which(name)
        if path is not None:
            return path
    return None


def _module_available(name):
    return importlib.util.find_spec(name) is not None


def build_report():
    if os.name == "nt":
        c_compiler = _find_command(("cl", "clang", "gcc"))
        cpp_compiler = _find_command(("cl", "clang++", "g++"))
    else:
        c_compiler = _find_command(("cc", "clang", "gcc"))
        cpp_compiler = _find_command(("c++", "clang++", "g++"))
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _module_available("numpy"),
        "cython": _module_available("Cython"),
        "pybind11": _module_available("pybind11"),
        "c_compiler": c_compiler,
        "cpp_compiler": cpp_compiler,
    }
    report["ready"] = all(
        (report["numpy"], report["cython"], report["pybind11"], c_compiler, cpp_compiler)
    )
    return report


def _print_report(report, as_json):
    if as_json:
        print(json.dumps(report, indent=2))
        return
    for key, value in report.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only check the build environment")
    parser.add_argument("--build", action="store_true", help="install the editable package with extensions")
    parser.add_argument("--numpy-only", action="store_true", help="install without native extensions")
    parser.add_argument("--json", action="store_true", help="print the environment report as JSON")
    args = parser.parse_args()

    if args.build and args.numpy_only:
        parser.error("--build and --numpy-only cannot be combined")
    if not args.check and not args.build and not args.numpy_only:
        args.check = True

    report = build_report()
    _print_report(report, args.json)
    if args.check:
        return 0 if report["ready"] else 2
    if args.numpy_only:
        env = os.environ.copy()
        env["MASTERML_DISABLE_EXTENSIONS"] = "1"
        command = [sys.executable, "-m", "pip", "install", "--editable", ".", "--no-build-isolation"]
    else:
        if not report["ready"]:
            print("Acceleration build prerequisites are missing; run with --numpy-only for fallback.", file=sys.stderr)
            return 2
        env = os.environ.copy()
        env.pop("MASTERML_DISABLE_EXTENSIONS", None)
        command = [sys.executable, "-m", "pip", "install", "--editable", ".[dev]", "--no-build-isolation"]
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
