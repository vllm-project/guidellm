"""Workload metrics component for HTML reports."""

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from guidellm.benchmark.outputs.html.components.base import PlotlyComponentBase


class WorkloadMetricsComponent(PlotlyComponentBase):
    """Generates the 2x2 workload metrics grid."""

    def generate(self, data: dict[str, Any]) -> str:
        """Generate workload metrics HTML with 2x2 grid of charts.

        Args:
            data: Dictionary containing:
                - benchmarks: List of benchmark dicts with metrics

        Returns:
            HTML string with workload metrics section.
        """
        benchmarks = data.get("benchmarks", [])

        if not benchmarks:
            return """
            <div class="section">
                <h2>Workload Metrics</h2>
                <p>No benchmark data available</p>
            </div>
            """

        # Create 2x2 subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Time to First Token (TTFT)",
                "Inter-Token Latency (ITL)",
                "Time Per Request",
                "Throughput (tokens/sec)",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # Apply base theme
        fig = self._apply_theme_to_figure(fig)

        # Extract RPS values for x-axis
        [bm["requests_per_second"] for bm in benchmarks]

        # Add metric traces to each subplot
        self._add_metric_traces(fig, benchmarks, "ttft", 1, 1, "TTFT (ms)")
        self._add_metric_traces(fig, benchmarks, "itl", 1, 2, "ITL (ms)")
        self._add_metric_traces(
            fig, benchmarks, "time_per_request", 2, 1, "Latency (s)"
        )
        self._add_metric_traces(
            fig, benchmarks, "throughput", 2, 2, "Throughput (tokens/s)"
        )

        # Update axes
        fig.update_xaxes(title_text="Requests per Second (RPS)", row=2, col=1)
        fig.update_xaxes(title_text="Requests per Second (RPS)", row=2, col=2)
        fig.update_yaxes(title_text="Milliseconds", row=1, col=1)
        fig.update_yaxes(title_text="Milliseconds", row=1, col=2)
        fig.update_yaxes(title_text="Seconds", row=2, col=1)
        fig.update_yaxes(title_text="Tokens/Second", row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text="Workload Metrics",
            title_font_size=24,
            title_font_color=self.theme.SECONDARY,
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
            height=800,
        )

        # Convert to HTML
        chart_html = fig.to_html(include_plotlyjs=False, div_id="workload-metrics")

        return f"""
        <div class="section">
            {chart_html}
        </div>
        """

    def _add_metric_traces(
        self,
        fig: go.Figure,
        benchmarks: list[dict[str, Any]],
        metric_name: str,
        row: int,
        col: int,
        trace_prefix: str,
    ) -> None:
        """Add metric traces to a subplot.

        Args:
            fig: Plotly figure to add traces to.
            benchmarks: List of benchmark data dicts.
            metric_name: Name of the metric (e.g., 'ttft', 'itl').
            row: Subplot row number.
            col: Subplot column number.
            trace_prefix: Prefix for trace names.
        """
        rps_values = [bm["requests_per_second"] for bm in benchmarks]

        # Extract metric data
        mean_values = [bm[metric_name].get("mean", 0) for bm in benchmarks]

        # Add mean line (primary trace)
        fig.add_trace(
            go.Scatter(
                x=rps_values,
                y=mean_values,
                mode="lines+markers",
                name=f"{trace_prefix} Mean",
                line={"width": 3, "color": self.theme.PRIMARY},
                marker={"size": 8},
                hovertemplate=(
                    f"RPS: %{{x:.2f}}<br>{trace_prefix}: %{{y:.2f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Add percentile lines
        percentiles = ["p50", "p90", "p95", "p99"]
        colors = [
            self.theme.TERTIARY,
            self.theme.SECONDARY,
            self.theme.QUATERNARY,
            self.theme.ERROR,
        ]

        for pct, color in zip(percentiles, colors, strict=False):
            # Check if percentiles exist in the data
            if benchmarks[0][metric_name].get("percentiles"):
                pct_values = [
                    bm[metric_name].get("percentiles", {}).get(pct, 0)
                    for bm in benchmarks
                ]

                fig.add_trace(
                    go.Scatter(
                        x=rps_values,
                        y=pct_values,
                        mode="lines",
                        name=f"{trace_prefix} {pct.upper()}",
                        line={"width": 1.5, "dash": "dash", "color": color},
                        hovertemplate=(
                            f"RPS: %{{x:.2f}}<br>{pct.upper()}: "
                            "%{y:.2f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
