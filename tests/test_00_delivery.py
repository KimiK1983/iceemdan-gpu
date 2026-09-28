import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_reference_intact():
    assert (
        hashlib.sha256((ROOT / "reference/ICEEMDAN_cpu.py").read_bytes()).hexdigest()
        == "d697bfb841ab2b099a1961a4575b671ba08867748e5cee63225f55a4be365b3a"
    )


def test_complete_modules_exist_and_compile():
    for name in ("runtime", "contracts", "geometry", "emd", "noise", "ensemble", "__init__"):
        p = ROOT / "iceemdan_cupy" / f"{name}.py"
        assert p.exists(), f"Missing stage implementation {p.name}"
        compile(p.read_text(encoding="utf-8"), str(p), "exec")


def test_every_cupy_symbol_documented():
    manifest = json.loads((ROOT / "docs/CUPY_API_LEDGER.json").read_text(encoding="utf-8"))[
        "operations"
    ]
    files = list((ROOT / "iceemdan_cupy").glob("*.py"))
    assert files
    missing = []
    for p in files:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if not isinstance(n, ast.Attribute):
                continue

            def dotted(node):
                if isinstance(node, ast.Name):
                    return node.id
                if isinstance(node, ast.Attribute):
                    parent = dotted(node.value)
                    return parent + "." + node.attr if parent else None
                return None

            text = dotted(n)
            if text and text.startswith("cp."):
                symbol = "cupy." + text[3:]
                if symbol in {
                    "cupy.__version__",
                    "cupy.cuda",
                    "cupy.cuda.runtime",
                    "cupy.linalg",
                    "cupy.random",
                    "cupy.ndarray",
                }:
                    continue
                if symbol not in manifest:
                    missing.append((p.name, n.lineno, symbol))
            elif n.attr in ("astype", "copy", "reshape", "sum", "item"):
                assert "cupy.ndarray." + n.attr in manifest
    assert not missing, missing


def test_no_scipy_cpu_numerical_imports_in_product():
    for p in (ROOT / "iceemdan_cupy").glob("*.py"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                assert not any(a.name == "scipy" or a.name.startswith("scipy.") for a in n.names)
            if isinstance(n, ast.ImportFrom):
                assert not (n.module or "").startswith("scipy")


def test_validation_maps_every_acceptance_family():
    from tools.validate import STAGES

    mapped = [file for stage in STAGES.values() for file in stage]
    discovered = [
        str(file.relative_to(ROOT)).replace("\\", "/")
        for file in (ROOT / "tests").glob("test_*.py")
    ]
    assert sorted(mapped) == sorted(discovered)
    assert "tests/test_07_batched_extrema.py" in mapped
    assert "tests/test_08_conditional_graph.py" in mapped


def test_validation_blocks_missing_cuda(monkeypatch, tmp_path):
    import sys

    from tools import validate

    output = tmp_path / "blocked.json"
    monkeypatch.setattr(sys, "argv", ["validate.py", "--stage", "1", "--output", str(output)])
    monkeypatch.setattr(validate, "probe_cuda", lambda: {"status": "BLOCKED", "reason": "test"})
    assert validate.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED"
    assert report["full_cuda_acceptance"] is False


def test_validation_preserves_artifacts_across_runs(monkeypatch, tmp_path):
    import sys
    from subprocess import CompletedProcess

    from tools import validate

    messages = iter(("first passed", "second passed"))

    def pytest_result(cmd, **kwargs):
        xml = next(Path(arg.split("=", 1)[1]) for arg in cmd if arg.startswith("--junitxml="))
        xml.write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>',
            encoding="utf-8",
        )
        return CompletedProcess(cmd, 0, next(messages), "")

    monkeypatch.setattr(validate.subprocess, "run", pytest_result)
    artifacts = []
    for name in ("first", "second"):
        output = tmp_path / f"{name}.json"
        monkeypatch.setattr(sys, "argv", ["validate.py", "--stage", "0", "--output", str(output)])
        assert validate.main() == 0
        stage = json.loads(output.read_text(encoding="utf-8"))["stages"][0]
        log = Path(stage["log"])
        xml = Path(stage["xml"])
        assert log.read_text(encoding="utf-8") == f"{name} passed"
        assert 'tests="1"' in xml.read_text(encoding="utf-8")
        artifacts.append((log, xml))
    assert set(artifacts[0]).isdisjoint(artifacts[1])
    assert artifacts[0][0].read_text(encoding="utf-8") == "first passed"
    assert artifacts[0][1].exists()
