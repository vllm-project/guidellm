import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from guidellm.benchmark.outputs.csv import GenerativeBenchmarkerCSV
from guidellm.schemas import StatusDistributionSummary


class TestAlignColumns:
    """
    Tests for _align_columns ensuring correct column merging and alignment
    when benchmarks have different sets of metrics.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.regression
    def test_headers_merge_in_first_seen_order(self):
        """
        Headers from multiple benchmarks are merged preserving first-seen order,
        producing the union of all columns.

        ## WRITTEN BY AI ##
        """
        headers_b1 = [["GroupA", "Field1", ""], ["GroupB", "Field2", ""]]
        headers_b2 = [["GroupA", "Field1", ""], ["GroupC", "Field3", ""]]
        values_b1 = ["v1", "v2"]
        values_b2 = ["v1_b2", "v3"]

        headers, rows = GenerativeBenchmarkerCSV._align_columns(
            [headers_b1, headers_b2], [values_b1, values_b2]
        )

        assert headers == [
            ["GroupA", "Field1", ""],
            ["GroupB", "Field2", ""],
            ["GroupC", "Field3", ""],
        ]
        assert rows[0] == ["v1", "v2", ""]
        assert rows[1] == ["v1_b2", "", "v3"]

    @pytest.mark.regression
    def test_missing_columns_filled_with_empty_string(self):
        """
        When the second benchmark is missing a column the first has, that
        position is filled with an empty string.

        ## WRITTEN BY AI ##
        """
        headers_b1 = [["G", "A", ""], ["G", "B", ""]]
        headers_b2 = [["G", "A", ""]]
        values_b1 = ["a", "b"]
        values_b2 = ["a2"]

        headers, rows = GenerativeBenchmarkerCSV._align_columns(
            [headers_b1, headers_b2], [values_b1, values_b2]
        )

        assert headers == [["G", "A", ""], ["G", "B", ""]]
        assert rows[0] == ["a", "b"]
        assert rows[1] == ["a2", ""]

    @pytest.mark.regression
    def test_first_benchmark_missing_columns(self):
        """
        When the first benchmark lacks columns that the second has, those
        columns are appended and the first row gets empty strings.

        ## WRITTEN BY AI ##
        """
        headers_b1 = [["G", "A", ""]]
        headers_b2 = [["G", "A", ""], ["G", "B", ""]]
        values_b1 = ["a1"]
        values_b2 = ["a2", "b2"]

        headers, rows = GenerativeBenchmarkerCSV._align_columns(
            [headers_b1, headers_b2], [values_b1, values_b2]
        )

        assert headers == [["G", "A", ""], ["G", "B", ""]]
        assert rows[0] == ["a1", ""]
        assert rows[1] == ["a2", "b2"]

    @pytest.mark.regression
    def test_identical_columns_no_padding(self):
        """
        When all benchmarks have the same columns, no padding is needed.

        ## WRITTEN BY AI ##
        """
        headers_b1 = [["G", "X", ""], ["G", "Y", ""]]
        headers_b2 = [["G", "X", ""], ["G", "Y", ""]]
        values_b1 = ["1", "2"]
        values_b2 = ["3", "4"]

        headers, rows = GenerativeBenchmarkerCSV._align_columns(
            [headers_b1, headers_b2], [values_b1, values_b2]
        )

        assert headers == [["G", "X", ""], ["G", "Y", ""]]
        assert rows[0] == ["1", "2"]
        assert rows[1] == ["3", "4"]

    @pytest.mark.smoke
    def test_empty_benchmarks_list(self):
        """
        No benchmarks produces empty headers and no data rows.

        ## WRITTEN BY AI ##
        """
        headers, rows = GenerativeBenchmarkerCSV._align_columns([], [])
        assert headers == []
        assert rows == []

    @pytest.mark.smoke
    def test_single_benchmark(self):
        """
        A single benchmark returns its headers and values unchanged.

        ## WRITTEN BY AI ##
        """
        headers_b1 = [["A", "B", "C"], ["D", "E", "F"]]
        values_b1 = [10, 20]

        headers, rows = GenerativeBenchmarkerCSV._align_columns(
            [headers_b1], [values_b1]
        )

        assert headers == [["A", "B", "C"], ["D", "E", "F"]]
        assert rows == [[10, 20]]

    @pytest.mark.regression
    def test_three_benchmarks_disjoint_columns(self):
        """
        Three benchmarks each with unique columns produces the full union
        with correct empty-fill for each row.

        ## WRITTEN BY AI ##
        """
        headers_b1 = [["G", "A", ""]]
        headers_b2 = [["G", "B", ""]]
        headers_b3 = [["G", "C", ""]]
        values_b1 = ["a"]
        values_b2 = ["b"]
        values_b3 = ["c"]

        headers, rows = GenerativeBenchmarkerCSV._align_columns(
            [headers_b1, headers_b2, headers_b3],
            [values_b1, values_b2, values_b3],
        )

        assert headers == [["G", "A", ""], ["G", "B", ""], ["G", "C", ""]]
        assert rows[0] == ["a", "", ""]
        assert rows[1] == ["", "b", ""]
        assert rows[2] == ["", "", "c"]


class TestHasDistributionData:
    """
    Tests for _has_distribution_data on GenerativeBenchmarkerCSV.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_returns_true_for_zero_valued_distribution(self):
        """
        _has_distribution_data returns True when a status has count > 0
        but total_sum == 0 (e.g. errored tool-call requests with count=0).

        ## WRITTEN BY AI ##
        """
        dist = StatusDistributionSummary.from_values(
            successful=[],
            incomplete=[],
            errored=[0.0, 0.0, 0.0],
        )
        csv_out = GenerativeBenchmarkerCSV.__new__(GenerativeBenchmarkerCSV)
        assert csv_out._has_distribution_data(dist) is True

    @pytest.mark.smoke
    def test_returns_false_for_empty_distribution(self):
        """
        _has_distribution_data returns False when all statuses have
        count == 0 (no data at all).

        ## WRITTEN BY AI ##
        """
        dist = StatusDistributionSummary.from_values(
            successful=[],
            incomplete=[],
            errored=[],
        )
        csv_out = GenerativeBenchmarkerCSV.__new__(GenerativeBenchmarkerCSV)
        assert csv_out._has_distribution_data(dist) is False

    @pytest.mark.smoke
    def test_returns_true_for_positive_distribution(self):
        """
        _has_distribution_data returns True for a normal positive distribution.

        ## WRITTEN BY AI ##
        """
        dist = StatusDistributionSummary.from_values(
            successful=[5.0, 10.0],
            incomplete=[],
            errored=[],
        )
        csv_out = GenerativeBenchmarkerCSV.__new__(GenerativeBenchmarkerCSV)
        assert csv_out._has_distribution_data(dist) is True


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_finalize_aligns_columns_in_written_csv(tmp_path: Path):
    """
    Integration test: finalize writes a CSV where all rows (headers + data)
    have the same column count, even when benchmarks produce different columns.

    Uses patching to control the column shape without constructing full
    benchmark objects.

    ## WRITTEN BY AI ##
    """
    report = SimpleNamespace(
        benchmarks=[
            SimpleNamespace(_test_fields=[(("G", "A", ""), "a1")]),
            SimpleNamespace(
                _test_fields=[(("G", "A", ""), "a2"), (("G", "B", ""), "b2")]
            ),
        ],
        metadata=SimpleNamespace(model_dump_json=lambda: "{}"),
        args=SimpleNamespace(model_dump_json=lambda: "{}"),
    )

    out = GenerativeBenchmarkerCSV(output_path=tmp_path)

    # Stub all emitters except _add_run_info so we control column shape
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

    path = await out.finalize(report)

    rows = list(csv.reader(path.open()))
    assert len(rows) == 5  # 3 header rows + 2 data rows

    # All rows must have the same column count
    col_counts = {len(row) for row in rows}
    assert len(col_counts) == 1, f"Expected uniform column count, got {col_counts}"

    # Data row for first benchmark should have blank in column B
    data_rows = rows[3:]
    assert data_rows[0] == ["a1", ""]
    assert data_rows[1] == ["a2", "b2"]


# Metrics read by GenerativeBenchmarkerCSV._add_request_latency_metrics.
_LATENCY_CSV_METRICS = (
    "request_latency",
    "request_dispatch_delay",
    "request_scheduled_latency",
    "request_streaming_iterations_count",
    "time_to_first_token_ms",
    "time_to_first_output_token_ms",
    "time_per_output_token_ms",
    "inter_token_latency_ms",
    "time_to_last_round_trip_ms",
    "avg_round_trip_time_ms",
)


def _latency_metric_groups(
    tmp_path: Path, schedule_metrics_missing: bool = False
) -> list[str]:
    """Emit the latency CSV section and return its column group names in order.

    ## WRITTEN BY AI ##
    """
    distribution = StatusDistributionSummary.from_values([1.0, 2.0, 3.0], [], [])
    metrics = dict.fromkeys(_LATENCY_CSV_METRICS, distribution)
    if schedule_metrics_missing:
        for name in ("request_dispatch_delay", "request_scheduled_latency"):
            metrics[name] = None
    benchmark = SimpleNamespace(metrics=SimpleNamespace(**metrics))

    output = GenerativeBenchmarkerCSV(output_path=tmp_path)
    headers: list[list[str]] = []
    values: list[str | int | float] = []
    output._add_request_latency_metrics(benchmark, headers, values)

    groups: list[str] = []
    for header in headers:
        if header[0] not in groups:
            groups.append(header[0])

    return groups


class TestRequestLatencyCSVMetrics:
    """
    Verify the latency section of the CSV export.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.sanity
    def test_exports_schedule_relative_metrics(self, tmp_path: Path):
        """
        Dispatch delay and scheduled latency reach the CSV even though they are
        omitted from the console table.

        ## WRITTEN BY AI ##
        """
        groups = _latency_metric_groups(tmp_path)

        assert "Dispatch Delay" in groups
        assert "Scheduled Latency" in groups

    @pytest.mark.regression
    def test_preserves_existing_column_order(self, tmp_path: Path):
        """
        Pre-existing latency columns keep their relative order, so the new
        groups are additions rather than a reshuffle.

        ## WRITTEN BY AI ##
        """
        groups = _latency_metric_groups(tmp_path)
        existing = [
            group
            for group in groups
            if group not in {"Dispatch Delay", "Scheduled Latency"}
        ]

        assert existing == [
            "Request Latency",
            "Streaming Iterations",
            "Time to First Token",
            "Time to First Output Token",
            "Time per Output Token",
            "Inter Token Latency",
            "Time To Last Round Trip",
            "Avg Round Trip Time",
        ]

    @pytest.mark.regression
    def test_omits_schedule_metrics_when_not_applicable(self, tmp_path: Path):
        """
        Schedule-relative metrics set to None produce no CSV columns.

        GenerativeMetrics.compile leaves these None for strategies without an
        arrival schedule, so a throughput row carries no misleading values.

        ## WRITTEN BY AI ##
        """
        groups = _latency_metric_groups(tmp_path, schedule_metrics_missing=True)

        assert "Dispatch Delay" not in groups
        assert "Scheduled Latency" not in groups
        assert "Request Latency" in groups
