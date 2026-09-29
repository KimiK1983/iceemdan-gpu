"""Check the public parser and the small, versioned metric evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_cudb import decode_212

ROOT = Path(__file__).resolve().parents[1]


def test_decode_212_signed_limits_and_length():
    samples = decode_212(bytes((0x00, 0xF0, 0xFF, 0xFF, 0x87, 0x00)))
    np.testing.assert_array_equal(samples, [0, -1, 2047, -2048])
    with pytest.raises(ValueError, match="divisible by three"):
        decode_212(b"\x00")


def test_public_metric_evidence_contains_only_scalar_provenance():
    source_hash = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted((ROOT / "iceemdan_cupy").glob("*.py")))
    ).hexdigest()
    for dataset, script, expected_rows in (
        ("keele", "analyze_keele.py", 2),
        ("cudb", "analyze_cudb.py", 1),
    ):
        path = ROOT / "evidence" / "public_data_20260928" / dataset / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
        assert len(rows) == expected_rows
        script_hash = hashlib.sha256((ROOT / "scripts" / script).read_bytes()).hexdigest()
        for row in rows:
            assert all(not isinstance(value, (list, dict)) for value in row.values())
            assert row["trials"] == 100
            assert row["gpu_route"] == "batch_emd=True,graph_control=False"
            assert row["analysis_script_sha256"] == script_hash
            assert row["gpu_source_sha256"] == source_hash
            assert "C:\\Users" not in json.dumps(row)
