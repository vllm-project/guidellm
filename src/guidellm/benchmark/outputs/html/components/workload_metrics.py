"""Workload metrics component for HTML reports."""

from typing import Any

import plotly.graph_objects as go

from guidellm.benchmark.outputs.html.components.base import PlotlyComponentBase


class WorkloadMetricsComponent(PlotlyComponentBase):
    """Generates the 2x2 workload metrics grid with separate charts."""

    def generate(self, data: dict[str, Any]) -> str:
        """Generate workload metrics HTML with 4 separate charts.

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

        # Sort benchmarks by requests_per_second to prevent line crossovers
        benchmarks_sorted = sorted(
            benchmarks, key=lambda bm: bm.get("requests_per_second", 0)
        )

        # Create 4 separate figures
        ttft_fig = self._create_metric_chart(
            benchmarks_sorted,
            "ttft",
            "Time to First Token (TTFT)",
            "Milliseconds",
        )
        itl_fig = self._create_metric_chart(
            benchmarks_sorted,
            "itl",
            "Inter-Token Latency (ITL)",
            "Milliseconds",
        )
        tpr_fig = self._create_metric_chart(
            benchmarks_sorted,
            "time_per_request",
            "Time Per Request",
            "Seconds",
        )
        throughput_fig = self._create_metric_chart(
            benchmarks_sorted,
            "throughput",
            "Throughput",
            "Tokens/Second",
        )

        # Convert each figure to HTML
        ttft_html = ttft_fig.to_html(include_plotlyjs=False, div_id="ttft-chart")
        itl_html = itl_fig.to_html(include_plotlyjs=False, div_id="itl-chart")
        tpr_html = tpr_fig.to_html(include_plotlyjs=False, div_id="tpr-chart")
        throughput_html = throughput_fig.to_html(
            include_plotlyjs=False, div_id="throughput-chart"
        )

        # Return HTML with 2x2 grid layout
        return f"""
        <div class="section">
            <h2>Workload Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>TIME TO FIRST TOKEN</h3>
                    {ttft_html}
                </div>
                <div class="metric-card">
                    <h3>INTER-TOKEN LATENCY</h3>
                    {itl_html}
                </div>
                <div class="metric-card">
                    <h3>TIME PER REQUEST</h3>
                    {tpr_html}
                </div>
                <div class="metric-card">
                    <h3>THROUGHPUT</h3>
                    {throughput_html}
                </div>
            </div>
        </div>
        """

    def _create_metric_chart(
        self,
        benchmarks: list[dict[str, Any]],
        metric_key: str,
        title: str,
        yaxis_title: str,
    ) -> go.Figure:
        """Create a single metric chart with mean + percentile lines.

        Args:
            benchmarks: Sorted list of benchmark data dicts.
            metric_key: Name of the metric (e.g., 'ttft', 'itl').
            title: Chart title.
            yaxis_title: Y-axis title.

        Returns:
            Plotly figure with metric chart.
        """
        fig = self._create_figure(
            title=title,
            xaxis_title="Requests per Second",
            yaxis_title=yaxis_title,
        )

        # Extract data
        rps_values = [bm["requests_per_second"] for bm in benchmarks]
        metric_data = [bm[metric_key] for bm in benchmarks]

        # Add mean line (solid, primary color, thicker)
        mean_values = [m["mean"] for m in metric_data]
        fig.add_trace(
            go.Scatter(
                x=rps_values,
                y=mean_values,
                mode="lines+markers",
                name="mean",
                line={"color": self.theme.PRIMARY, "width": 3},
                marker={"size": 6, "color": self.theme.PRIMARY},
                hovertemplate="RPS: %{x:.2f}<br>Mean: %{y:.2f}<extra></extra>",
            )
        )

        # Add percentile lines (dashed, thinner)
        percentiles = ["p50", "p90", "p95", "p99"]
        colors = [
            self.theme.TERTIARY,  # p50 - teal
            self.theme.SECONDARY,  # p90 - purple
            self.theme.QUATERNARY,  # p95 - yellow
            self.theme.ERROR,  # p99 - red
        ]

        for pct, color in zip(percentiles, colors, strict=False):
            # Check if percentiles exist in the data
            if metric_data[0].get("percentiles"):
                pct_values = [m.get("percentiles", {}).get(pct, 0) for m in metric_data]
                fig.add_trace(
                    go.Scatter(
                        x=rps_values,
                        y=pct_values,
                        mode="lines+markers",
                        name=pct,
                        line={"color": color, "width": 2, "dash": "dot"},
                        marker={"size": 4, "color": color},
                        hovertemplate=(
                            f"RPS: %{{x:.2f}}<br>{pct.upper()}: "
                            "%{y:.2f}<extra></extra>"
                        ),
                    )
                )

        # Update layout with individual legend
        fig.update_layout(
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.15,  # Below the chart
                "xanchor": "center",
                "x": 0.5,
            },
            height=400,  # Fixed height for consistency
            margin={"l": 60, "r": 20, "t": 60, "b": 100},  # Extra bottom for legend
        )

        return fig
