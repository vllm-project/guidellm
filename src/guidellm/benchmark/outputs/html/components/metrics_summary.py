"""Metrics summary component for HTML reports."""

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from guidellm.benchmark.outputs.html.components.base import PlotlyComponentBase


class MetricsSummaryComponent(PlotlyComponentBase):
    """Generates the SLO metrics summary dashboard with interactive RPS slider."""

    def generate(self, data: dict[str, Any]) -> str:
        """Generate metrics summary HTML with SLO dashboard.

        Args:
            data: Dictionary containing:
                - benchmarks: List of benchmark dicts with metrics
                - thresholds: Optional dict with SLO thresholds

        Returns:
            HTML string with metrics summary section.
        """
        benchmarks = data.get("benchmarks", [])
        thresholds = data.get("thresholds", {})

        if not benchmarks:
            return """
            <div class="section">
                <h2>Metrics Summary</h2>
                <p>No benchmark data available</p>
            </div>
            """

        # Create the interactive dashboard figure
        fig = self._create_dashboard_figure(benchmarks, thresholds)

        # Convert to HTML
        chart_html = fig.to_html(include_plotlyjs=False, div_id="metrics-summary")

        return f"""
        <div class="section">
            <h2>SLO Metrics Summary</h2>
            <p style="color: {self.theme.TEXT_SECONDARY}; margin-bottom: 1rem;">
                Use the slider below to explore metrics at different RPS rates.
                Default percentile: P95
            </p>
            {chart_html}
        </div>
        """

    def _create_dashboard_figure(
        self, benchmarks: list[dict[str, Any]], thresholds: dict[str, float]
    ) -> go.Figure:
        """Create interactive dashboard with RPS slider.

        Args:
            benchmarks: List of benchmark data.
            thresholds: Optional SLO thresholds.

        Returns:
            Plotly figure with interactive dashboard.
        """
        # Create 2x2 subplot
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "TTFT (ms)",
                "ITL (ms)",
                "Time Per Request (s)",
                "Throughput (tokens/s)",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # Apply theme
        fig = self._apply_theme_to_figure(fig)

        # Extract RPS values
        rps_values = [bm["requests_per_second"] for bm in benchmarks]

        # For each RPS value, create traces (all hidden except default)
        default_idx = len(benchmarks) // 2  # Default to middle RPS value

        # Create traces for each metric at each RPS
        self._add_metric_bar_traces(
            fig, benchmarks, "ttft", 1, 1, "TTFT", default_idx, thresholds.get("ttft")
        )
        self._add_metric_bar_traces(
            fig, benchmarks, "itl", 1, 2, "ITL", default_idx, thresholds.get("itl")
        )
        self._add_metric_bar_traces(
            fig,
            benchmarks,
            "time_per_request",
            2,
            1,
            "Latency",
            default_idx,
            thresholds.get("time_per_request"),
        )
        self._add_metric_bar_traces(
            fig,
            benchmarks,
            "throughput",
            2,
            2,
            "Throughput",
            default_idx,
            thresholds.get("throughput"),
        )

        # Create slider steps
        steps = []
        for i, rps in enumerate(rps_values):
            step = {
                "method": "update",
                "args": [
                    {"visible": self._create_visibility_array(i, len(benchmarks))},
                    {
                        "title": f"SLO Metrics at {rps:.2f} RPS"
                    },  # Update title with current RPS
                ],
                "label": f"{rps:.1f}",
            }
            steps.append(step)

        sliders = [
            {
                "active": default_idx,
                "yanchor": "top",
                "y": -0.15,
                "xanchor": "left",
                "x": 0.1,
                "currentvalue": {
                    "prefix": "RPS: ",
                    "visible": True,
                    "xanchor": "center",
                    "font": {"size": 16, "color": self.theme.PRIMARY},
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.8,
                "steps": steps,
            }
        ]

        # Update layout
        fig.update_layout(
            title=f"SLO Metrics at {rps_values[default_idx]:.2f} RPS",
            title_font_size=24,
            title_font_color=self.theme.SECONDARY,
            sliders=sliders,
            showlegend=False,
            height=700,
            margin={"b": 120},  # Extra margin for slider
        )

        return fig

    def _add_metric_bar_traces(
        self,
        fig: go.Figure,
        benchmarks: list[dict[str, Any]],
        metric_name: str,
        row: int,
        col: int,
        label: str,
        default_idx: int,
        threshold: float | None = None,
    ) -> None:
        """Add bar chart traces for a metric across all RPS values.

        Args:
            fig: Figure to add traces to.
            benchmarks: List of benchmark data.
            metric_name: Metric name (e.g., 'ttft').
            row: Subplot row.
            col: Subplot column.
            label: Label for the metric.
            default_idx: Index of default visible trace.
            threshold: Optional SLO threshold value.
        """
        percentiles = ["p50", "p90", "p95", "p99"]
        percentile_labels = ["P50", "P90", "P95 (default)", "P99"]
        colors = [
            self.theme.TERTIARY,
            self.theme.SECONDARY,
            self.theme.PRIMARY,
            self.theme.ERROR,
        ]

        for i, bm in enumerate(benchmarks):
            metric_data = bm[metric_name]
            percentiles_data = metric_data.get("percentiles", {})

            # Get values for each percentile
            pct_values = [percentiles_data.get(p, 0) for p in percentiles]

            # Add bar trace
            visible = i == default_idx

            fig.add_trace(
                go.Bar(
                    x=percentile_labels,
                    y=pct_values,
                    marker_color=colors,
                    visible=visible,
                    hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
                    text=[f"{v:.2f}" for v in pct_values],
                    textposition="outside",
                    textfont={"color": self.theme.TEXT_PRIMARY},
                ),
                row=row,
                col=col,
            )

            # Add threshold line if provided
            if threshold and visible:
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color=self.theme.SUCCESS,
                    annotation_text=f"Threshold: {threshold}",
                    annotation_position="right",
                    row=row,
                    col=col,
                )

    def _create_visibility_array(
        self, active_idx: int, num_benchmarks: int
    ) -> list[bool]:
        """Create visibility array for slider step.

        Each metric has num_benchmarks traces, so visibility array needs
        to show the active_idx trace for each of the 4 metrics.

        Args:
            active_idx: Index of RPS value to show.
            num_benchmarks: Total number of benchmarks.

        Returns:
            List of boolean visibility values.
        """
        visibility = []
        for _metric_idx in range(4):  # 4 metrics (TTFT, ITL, TPR, Throughput)
            for bm_idx in range(num_benchmarks):
                visibility.append(bm_idx == active_idx)
        return visibility
