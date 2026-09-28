"""Audit documented call sites and host numeric boundaries; never imports CuPy."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return parent + "." + node.attr if parent else None
    return None


def main():
    ledger = json.loads((ROOT / "docs/CUPY_API_LEDGER.json").read_text(encoding="utf-8"))[
        "operations"
    ]
    calls = []
    missing = []
    host = []
    metadata = {"cp.cuda", "cp.cuda.runtime", "cp.linalg", "cp.random", "cp.__version__"}
    opmap = {
        ast.Add: "cupy.add",
        ast.Sub: "cupy.subtract",
        ast.Mult: "cupy.multiply",
        ast.Div: "cupy.divide",
        ast.BitAnd: "cupy.bitwise_and",
        ast.BitOr: "cupy.bitwise_or",
        ast.USub: "cupy.negative",
        ast.Eq: "cupy.equal",
        ast.NotEq: "cupy.not_equal",
        ast.Lt: "cupy.less",
        ast.LtE: "cupy.less_equal",
        ast.Gt: "cupy.greater",
        ast.GtE: "cupy.greater_equal",
    }
    for p in sorted((ROOT / "iceemdan_cupy").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            symbol = None
            if isinstance(n, ast.Attribute):
                q = dotted(n)
                if q and q.startswith("cp.") and q not in metadata:
                    symbol = "cupy." + q[3:]
                elif n.attr in ("astype", "copy", "reshape", "sum", "item"):
                    symbol = "cupy.ndarray." + n.attr
            if isinstance(n, ast.BinOp) and type(n.op) in opmap:
                symbol = opmap[type(n.op)]
            if isinstance(n, ast.UnaryOp) and type(n.op) in opmap:
                symbol = opmap[type(n.op)]
            if isinstance(n, ast.Compare):
                for op in n.ops:
                    if type(op) in opmap:
                        name = opmap[type(op)]
                        calls.append(
                            {
                                "file": p.name,
                                "line": n.lineno,
                                "api": name,
                                "scope": "potential_array_operator; scalar operations are a safe over-approximation",
                            }
                        )
            if isinstance(n, ast.ImportFrom) and n.module == "cupyx.scipy.interpolate":
                for a in n.names:
                    name = "cupyx.scipy.interpolate." + a.name
                    calls.append(
                        {
                            "file": p.name,
                            "line": n.lineno,
                            "api": name,
                            "scope": "constructor and evaluation",
                        }
                    )
            if symbol:
                calls.append(
                    {
                        "file": p.name,
                        "line": n.lineno,
                        "api": symbol,
                        "scope": "source call/attribute/operator",
                    }
                )
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("numpy"):
                host.append(
                    {
                        "file": p.name,
                        "line": n.lineno,
                        "import": ast.unparse(n),
                        "purpose": "explicit PCG64 noise"
                        if n.module == "numpy.random"
                        else "dtype, finfo, bool scalar metadata",
                    }
                )
    for x in calls:
        if x["api"] not in ledger:
            missing.append(x)
    report = {
        "documentation_entries": len(ledger),
        "reviewed_sites": len(calls),
        "used_documented_symbols": len(set(x["api"] for x in calls)),
        "missing": missing,
        "numpy_boundaries": host,
        "sites": calls,
        "limitations": "Static call coverage is not a numerical proof. Dynamic ndarray methods and runtime context behavior are additionally reviewed in ledger; tests are separate.",
    }
    output = ROOT / "results/API_COVERAGE.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert not missing, missing
    print(json.dumps({k: v for k, v in report.items() if k not in ("sites",)}, indent=2))


if __name__ == "__main__":
    main()
