"""
Generate a multi-run HTML fixture for the Node/jsdom smoke test.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import sys
from pathlib import Path

from guidellm.benchmark.outputs.html import build_report_view, render_html_report
from guidellm.scheduler import ConcurrentStrategy
from tests.unit.benchmark.html_report_fixtures import make_benchmark, report


def main() -> None:
    """
    Write a multi-run HTML fixture to the path given on the command line.

    ## WRITTEN BY AI ##
    """
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_fixture.py <output.html>")
    out = Path(sys.argv[1])
    built = report(
        make_benchmark(
            strategy=ConcurrentStrategy(streams=2),
            rps=1.0,
            tps=20.0,
            measure_start=1_700_000_000.0,
        ),
        make_benchmark(
            strategy=ConcurrentStrategy(streams=4),
            rps=2.0,
            tps=50.0,
            measure_start=1_700_000_010.0,
        ),
    )
    out.write_text(render_html_report(build_report_view(built)), encoding="utf-8")


if __name__ == "__main__":
    main()
