## WRITTEN BY AI ##
import asyncio
import csv
from pathlib import Path
from types import SimpleNamespace

from guidellm.benchmark.outputs.csv import GenerativeBenchmarkerCSV


def _make_report(benchmarks):
    """Build a minimal benchmark report for CSV output tests."""
    return SimpleNamespace(
        benchmarks=benchmarks,
        metadata=SimpleNamespace(model_dump_json=lambda: "{}"),
        args=SimpleNamespace(model_dump_json=lambda: "{}"),
    )


def _finalize_sync(out, report):
    """Run the async CSV finalizer in a synchronous test context."""
    return asyncio.run(out.finalize(report))


def _make_csv_output(tmp_path: Path, benchmarks):
    """Create a CSV output instance with only the fields this test needs."""
    report = _make_report(benchmarks)
    out = GenerativeBenchmarkerCSV(output_path=tmp_path)

    # Disable unrelated metric emitters so the test controls the output shape.
    for name in [
        "_add_benchmark_info",
        "_add_timing_info",
        "_add_request_counts",
        "_add_request_latency_metrics",
        "_add_server_throughput_metrics",
        "_add_modality_metrics",
        "_add_scheduler_info",
        "_add_runtime_info",
    ]:
        setattr(out, name, lambda *a, **k: None)

    def _add_run_info(self, benchmark, headers, values):
        for key, val in benchmark._test_fields:
            headers.append(list(key))
            values.append(val)

    out._add_run_info = _add_run_info.__get__(out, out.__class__)
    return out, report


def test_headers_merge_and_order(tmp_path: Path):
    """Ensure headers from multiple benchmarks are merged in first-seen order."""
    bench1 = SimpleNamespace(
        _test_fields=[
            (("GroupA", "Field1", ""), "v1"),
            (("GroupB", "Field2", ""), "v2"),
        ]
    )

    bench2 = SimpleNamespace(
        _test_fields=[
            (("GroupA", "Field1", ""), "v1_b2"),
            (("GroupC", "Field3", ""), "v3"),
        ]
    )

    out, report = _make_csv_output(tmp_path, [bench1, bench2])
    path = _finalize_sync(out, report)

    rows = list(csv.reader(path.open()))
    header_rows = rows[:3]

    reconstructed = [
        tuple(col[i] for col in header_rows) for i in range(len(header_rows[0]))
    ]

    assert reconstructed == [
        ("GroupA", "Field1", ""),
        ("GroupB", "Field2", ""),
        ("GroupC", "Field3", ""),
    ]


def test_values_alignment(tmp_path: Path):
    """Ensure missing columns are written as blanks for each aligned row."""
    bench1 = SimpleNamespace(
        _test_fields=[(("G", "A", ""), "a"), (("G", "B", ""), "b")]
    )
    bench2 = SimpleNamespace(_test_fields=[(("G", "A", ""), "a2")])

    out, report = _make_csv_output(tmp_path, [bench1, bench2])
    path = _finalize_sync(out, report)
    rows = list(csv.reader(path.open()))
    data_rows = rows[3:]

    assert data_rows[0] == ["a", "b"]
    assert data_rows[1] == ["a2", ""]
