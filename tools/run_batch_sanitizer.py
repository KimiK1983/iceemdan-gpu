"""Run the bounded 4x4 Compute Sanitizer gate, one CUDA process at a time."""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("memcheck", "synccheck", "racecheck", "initcheck")
CASES = ("odd_i1", "block_i5", "three_stage", "colominas_i50")
PASS_PREFIX = "BATCH_CASE_PASS "


def classify(returncode, summaries, marker, timed_out=False, crashed=False):
    if timed_out:
        return "BLOCKED_TIMEOUT"
    if crashed:
        return "BLOCKED_CRASH"
    if not summaries:
        return "BLOCKED_NO_SUMMARY"
    if any(count > 0 for count in summaries):
        return "FAIL_SANITIZER"
    if returncode != 0:
        return "FAIL_CASE"
    if marker is None:
        return "BLOCKED_NO_PASS_MARKER"
    return "PASS"


def sanitizer_path(explicit):
    if explicit:
        path = explicit
    elif os.environ.get("COMPUTE_SANITIZER"):
        path = Path(os.environ["COMPUTE_SANITIZER"])
    elif found := shutil.which("compute-sanitizer"):
        path = Path(found)
    else:
        matches = sorted(
            ROOT.parent.glob(
                "ComputeSanitizer_*/extracted/*/compute-sanitizer/compute-sanitizer.exe"
            )
        )
        if not matches:
            raise FileNotFoundError("Compute Sanitizer not found; pass --sanitizer PATH.")
        path = matches[-1]
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def code_sha256():
    digest = hashlib.sha256()
    files = [*sorted((ROOT / "iceemdan_cupy").glob("*.py"))]
    files += [ROOT / "tools/sanitizer_batch_case.py", Path(__file__)]
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_cell(sanitizer, python, tool, case, timeout, output_dir):
    stem = f"{case}_{tool}"
    log = output_dir / f"{stem}.log"
    stdout_file = output_dir / f"{stem}.stdout.txt"
    command = [
        str(sanitizer),
        "--tool",
        tool,
        "--error-exitcode",
        "99",
        "--kill",
        "--log-file",
        str(log),
        str(python),
        str(ROOT / "tools/sanitizer_batch_case.py"),
        case,
    ]
    env = dict(os.environ)
    env.pop("ICEEMDAN_TEST_MODE", None)
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    timed_out = False
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            process.kill()
        stdout, _ = process.communicate(timeout=15)
    stdout_file.write_text(stdout, encoding="utf-8")
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    summaries = [int(x) for x in re.findall(r"ERROR SUMMARY:\s*(\d+)", log_text)]
    marker = None
    for line in stdout.splitlines():
        if line.startswith(PASS_PREFIX):
            try:
                candidate = json.loads(line[len(PASS_PREFIX) :])
                if candidate.get("case") == case:
                    marker = candidate
            except json.JSONDecodeError:
                pass
    crashed = "Windows fatal exception" in stdout or process.returncode in (
        0xC0000005,
        -1073741819,
    )
    status = classify(process.returncode, summaries, marker, timed_out, crashed)
    result = {
        "case": case,
        "tool": tool,
        "status": status,
        "returncode": process.returncode,
        "seconds": round(time.perf_counter() - start, 3),
        "error_summaries": summaries,
        "marker": marker,
        "command": command,
        "log": str(log.relative_to(ROOT)),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None,
        "stdout": str(stdout_file.relative_to(ROOT)),
        "stdout_sha256": hashlib.sha256(stdout_file.read_bytes()).hexdigest(),
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitizer", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--tool", choices=TOOLS)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    sanitizer = sanitizer_path(args.sanitizer)
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    output_dir = args.output_dir or ROOT / "results/batch_sanitizer" / dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    version = subprocess.run(
        [str(sanitizer), "--version"], capture_output=True, text=True, check=False
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "code_commit": commit,
        "code_sha256": code_sha256(),
        "sanitizer": str(sanitizer),
        "sanitizer_version": (version.stdout + version.stderr).strip(),
        "python": str(python),
        "timeout_seconds_per_cell": args.timeout,
        "cases": list(CASES if args.case is None else (args.case,)),
        "tools": list(TOOLS if args.tool is None else (args.tool,)),
        "results": [],
    }
    for case in report["cases"]:
        for tool in report["tools"]:
            print(f"RUN {case} {tool}", flush=True)
            result = run_cell(sanitizer, python, tool, case, args.timeout, output_dir)
            report["results"].append(result)
            (output_dir / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(f"{result['status']} {case} {tool} {result['seconds']}s", flush=True)
    return 0 if all(result["status"] == "PASS" for result in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
