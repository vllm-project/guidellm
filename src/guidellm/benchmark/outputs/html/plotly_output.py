"""Plotly-based HTML output generator for benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from guidellm.benchmark.outputs.html.components.footer import FooterComponent
from guidellm.benchmark.outputs.html.components.header import HeaderComponent
from guidellm.benchmark.outputs.html.components.metrics_summary import (
    MetricsSummaryComponent,
)
from guidellm.benchmark.outputs.html.components.workload_details import (
    WorkloadDetailsComponent,
)
from guidellm.benchmark.outputs.html.components.workload_metrics import (
    WorkloadMetricsComponent,
)

# Import data building functions
from guidellm.benchmark.outputs.html.data_builder import build_ui_data
from guidellm.benchmark.outputs.html.theme import PlotlyTheme
from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import GenerativeBenchmarksReport

__all__ = ["GenerativeBenchmarkerHTML"]


@GenerativeBenchmarkerOutput.register("html")
class GenerativeBenchmarkerHTML(GenerativeBenchmarkerOutput):
    """
    Plotly-based HTML output formatter for benchmark results.

    Generates interactive HTML reports using Plotly charts.
    This eliminates JavaScript dependencies and security vulnerabilities while
    maintaining visual appearance and interactivity.

    :cvar DEFAULT_FILE: Default filename for HTML output
    """

    DEFAULT_FILE: ClassVar[str] = "benchmarks.html"

    output_path: Path = Field(
        default_factory=lambda: Path.cwd(),
        description=(
            "Directory or file path for saving the HTML report, "
            "defaults to current working directory"
        ),
    )

    @classmethod
    def validated_kwargs(
        cls, output_path: str | Path | None, **_kwargs
    ) -> dict[str, Any]:
        """
        Validate and normalize output path argument.

        :param output_path: Output file or directory path for the HTML report
        :return: Dictionary containing validated output_path if provided
        """
        validated: dict[str, Any] = {}
        if output_path is not None:
            validated["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
        return validated

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Generate and save the Plotly-based HTML benchmark report.

        Builds data structures, generates components, assembles HTML, and writes
        to the output path.

        :param report: Completed benchmark report containing all results
        :return: Path to the saved HTML report file
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / self.DEFAULT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build UI data using existing functions
        data = build_ui_data(report.benchmarks, report.args)

        # Generate HTML content
        html_content = self._assemble_html(data)

        # Write to file
        output_path.write_text(html_content, encoding="utf-8")

        return output_path

    def _assemble_html(self, data: dict[str, Any]) -> str:
        """
        Assemble complete HTML from components.

        :param data: UI data dictionary with run_info, workload_details, benchmarks
        :return: Complete HTML string
        """
        theme = PlotlyTheme()

        # Initialize components
        header_component = HeaderComponent(theme=theme)
        footer_component = FooterComponent(theme=theme)
        workload_details_component = WorkloadDetailsComponent(theme=theme)
        metrics_summary_component = MetricsSummaryComponent(theme=theme)
        workload_metrics_component = WorkloadMetricsComponent(theme=theme)

        # Generate component HTML
        header_html = header_component.generate(data["run_info"])
        footer_html = footer_component.generate()
        workload_details_html = workload_details_component.generate(
            data["workload_details"]
        )

        # For metrics components, pass benchmarks
        metrics_data = {"benchmarks": data["benchmarks"]}
        metrics_summary_html = metrics_summary_component.generate(metrics_data)
        workload_metrics_html = workload_metrics_component.generate(metrics_data)

        # Load base template
        template_path = Path(__file__).parent / "templates" / "base.html"
        template = template_path.read_text(encoding="utf-8")

        # Replace placeholders
        html = template.replace("{CSS_CONTENT}", theme.get_css())
        html = html.replace("{HEADER_CONTENT}", header_html)
        html = html.replace("{WORKLOAD_DETAILS_CONTENT}", workload_details_html)
        html = html.replace("{METRICS_SUMMARY_CONTENT}", metrics_summary_html)
        html = html.replace("{WORKLOAD_METRICS_CONTENT}", workload_metrics_html)
        return html.replace("{FOOTER_CONTENT}", footer_html)
