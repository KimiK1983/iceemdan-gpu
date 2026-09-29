#!/usr/bin/env python3
"""Run independent acceptance stages. CUDA absence is BLOCKED, never PASS.

Examples:
  python tools/validate.py --stage 0
  python tools/validate.py --stage 1
  python tools/validate.py --stage all
  python tools/validate.py --stage all --mode cpu-surrogate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "0": ["tests/test_00_delivery.py", "tests/test_10_public_scripts.py"],
    "1": ["tests/test_01_contracts_extrema.py"],
    "2": ["tests/test_02_geometry.py"],
    "3": ["tests/test_03_emd.py"],
    "4": ["tests/test_04_ensemble.py"],
    "5": ["tests/test_05_integration.py", "tests/test_06_review.py"],
    "6": ["tests/test_07_batched_extrema.py"],
    "7": ["tests/test_08_conditional_graph.py"],
    "8": ["tests/test_09_numeric_contracts.py"],
}
SURROGATE_EXCLUSIONS = {"4": "test_default_rng_generates_on_device"}
HASH = "d697bfb841ab2b099a1961a4575b671ba08867748e5cee63225f55a4be365b3a"


def probe_cuda():
    try:
        import cupy as cp

        if getattr(cp, "_test_surrogate", False):
            raise RuntimeError("A test surrogate is not a CUDA runtime.")
        if cp.__version__ != "14.2.0":
            raise RuntimeError(f"Required CuPy 14.2.0, found {cp.__version__}")
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1:
            raise RuntimeError("No CUDA device found.")
        with cp.cuda.Device(0):
            cp.zeros(1, dtype="float64").item()
            cp.cuda.Device().synchronize()
            memory = cp.cuda.Device().mem_info
        return {
            "status": "AVAILABLE",
            "cupy": cp.__version__,
            "devices": n,
            "device0_mem_info": list(memory),
        }
    except Exception as exc:
        return {"status": "BLOCKED", "exception": type(exc).__name__, "reason": str(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    ap.add_argument("--mode", choices=["cuda", "cpu-surrogate"], default="cuda")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    discovered = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "tests").glob("test_*.py")
    }
    mapped = [path for paths in STAGES.values() for path in paths]
    if set(mapped) != discovered or len(mapped) != len(discovered):
        raise RuntimeError(
            f"Acceptance families omitted or duplicated: {sorted(discovered - set(mapped))}; unexpected mappings: {sorted(set(mapped) - discovered)}"
        )
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    output = (
        args.output or ROOT / "results" / f"validation_{args.mode}_{args.stage}.json"
    ).resolve()
    actual = hashlib.sha256((ROOT / "reference/ICEEMDAN_cpu.py").read_bytes()).hexdigest()
    if actual != HASH:
        raise RuntimeError("Frozen reference hash changed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{output.stem}_artifacts_", dir=output.parent))
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    report = {
        "mode": args.mode,
        "python": sys.version,
        "platform": platform.platform(),
        "reference_sha256": actual,
        "gpu_execution": False,
        "stages": [],
        "note": "CPU surrogate checks Python logic only; it is NOT CuPy or CUDA."
        if args.mode == "cpu-surrogate"
        else "Real CUDA tests only; missing GPU is a blocker.",
    }
    capability = None
    exitcode = 0
    for stage in selected:
        if exitcode:
            report["stages"].append({"stage": stage, "status": "NOT_RUN_DEPENDENCY"})
            continue
        if args.mode == "cpu-surrogate" and stage in ("6", "7", "8"):
            report["stages"].append({"stage": stage, "status": "NOT_APPLICABLE_CUDA"})
            if args.stage != "all":
                exitcode = 2
            continue
        if stage != "0" and args.mode == "cuda":
            if capability is None:
                capability = probe_cuda()
                report["cuda"] = capability
            if capability["status"] != "AVAILABLE":
                report["stages"].append(
                    {"stage": stage, "status": "BLOCKED", "reason": capability["reason"]}
                )
                exitcode = 2
                continue
            report["gpu_execution"] = True
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        if args.mode == "cpu-surrogate":
            env["ICEEMDAN_TEST_MODE"] = "cpu-surrogate"
        else:
            env.pop("ICEEMDAN_TEST_MODE", None)
        xml = run_dir / f"junit_{args.mode}_stage{stage}.xml"
        log = run_dir / f"pytest_{args.mode}_stage{stage}.txt"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *STAGES[stage],
            "-q",
            "--tb=short",
            f"--junitxml={xml}",
        ]
        if args.mode == "cpu-surrogate" and stage in SURROGATE_EXCLUSIONS:
            cmd += ["-k", f"not {SURROGATE_EXCLUSIONS[stage]}"]
        tic = time.perf_counter()
        completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        counters = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        if xml.exists():
            suites = ET.parse(xml).getroot()
            for suite in suites.findall("testsuite"):
                for name in counters:
                    counters[name] += int(suite.attrib.get(name, 0))
        success = completed.returncode == 0 and counters["tests"] > 0 and counters["skipped"] == 0
        status = (
            (
                "PASS_CPU_STATIC"
                if stage == "0"
                else "PASS_CUDA"
                if args.mode == "cuda"
                else "PASS_CPU_SURROGATE"
            )
            if success
            else "FAIL_OR_INCOMPLETE"
        )
        report["stages"].append(
            {
                "stage": stage,
                "status": status,
                **counters,
                "seconds": time.perf_counter() - tic,
                "log": str(log),
                "xml": str(xml) if xml.exists() else None,
                "surrogate_excluded": SURROGATE_EXCLUSIONS.get(stage)
                if args.mode == "cpu-surrogate"
                else None,
            }
        )
        print(f"Stage {stage}: {status} ({counters})", flush=True)
        if not success:
            exitcode = 1
    report["status"] = (
        "BLOCKED"
        if exitcode == 2
        else "FAILED"
        if exitcode
        else "PASSED_CPU_SURROGATE"
        if args.mode == "cpu-surrogate"
        else "PASSED_SELECTED_STAGES"
    )
    report["full_cuda_acceptance"] = (
        args.mode == "cuda"
        and args.stage == "all"
        and report["gpu_execution"]
        and exitcode == 0
        and all(item["status"] in ("PASS_CPU_STATIC", "PASS_CUDA") for item in report["stages"])
    )
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT: {output}")
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
