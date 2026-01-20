"""Workload details component for HTML reports."""

from typing import Any

import plotly.graph_objects as go

from guidellm.benchmark.outputs.html.components.base import PlotlyComponentBase

# Maximum characters to display for sample text
_SAMPLE_MAX_LENGTH = 100


class WorkloadDetailsComponent(PlotlyComponentBase):
    """Generates the workload details 3-panel layout."""

    def generate(self, data: dict[str, Any]) -> str:
        """Generate workload details HTML with charts.

        Args:
            data: Dictionary containing:
                - prompts: Dict with 'samples' and 'token_distributions'
                - generations: Dict with 'samples' and 'token_distributions'
                - requests_over_time: Dict with request timing data
                - server: Dict with 'target' URL
                - rate_type: Benchmark rate type

        Returns:
            HTML string with workload details section including charts.
        """
        prompts_data = data.get("prompts", {})
        generations_data = data.get("generations", {})
        requests_data = data.get("requests_over_time", {})
        server_data = data.get("server", {})
        rate_type = data.get("rate_type", "N/A")
        num_benchmarks = requests_data.get("num_benchmarks", 0)

        # Build HTML sections
        prompts_html = self._generate_prompts_section(prompts_data)
        server_html = self._generate_server_section(
            server_data, rate_type, num_benchmarks
        )
        generations_html = self._generate_generations_section(generations_data)

        # Build charts
        prompt_tokens_fig = self._create_histogram_chart(
            prompts_data.get("token_distributions", {}),
            "Prompt Token Distribution",
            "length (tokens)",
        )
        output_tokens_fig = self._create_histogram_chart(
            generations_data.get("token_distributions", {}),
            "Output Token Distribution",
            "length (tokens)",
        )
        requests_fig = self._create_requests_over_time_chart(requests_data)

        # Convert figures to HTML
        prompt_chart_html = prompt_tokens_fig.to_html(
            include_plotlyjs=False, div_id="prompt-tokens-chart"
        )
        output_chart_html = output_tokens_fig.to_html(
            include_plotlyjs=False, div_id="output-tokens-chart"
        )
        requests_chart_html = requests_fig.to_html(
            include_plotlyjs=False, div_id="requests-over-time-chart"
        )

        return f"""
        <div class="section">
            <h2>Workload Details</h2>
            <div class="grid-3col">
                <div>
                    <h3>Prompts</h3>
                    {prompts_html}
                    {prompt_chart_html}
                </div>
                <div>
                    <h3>Server Configuration</h3>
                    {server_html}
                    {requests_chart_html}
                </div>
                <div>
                    <h3>Generations</h3>
                    {generations_html}
                    {output_chart_html}
                </div>
            </div>
        </div>
        """

    def _generate_prompts_section(self, prompts_data: dict[str, Any]) -> str:
        """Generate HTML for prompts samples.

        Args:
            prompts_data: Dict with 'samples' and 'token_distributions'.

        Returns:
            HTML string for prompts section.
        """
        samples = prompts_data.get("samples", [])
        token_stats = prompts_data.get("token_distributions", {}).get("statistics", {})
        mean_tokens = token_stats.get("mean", 0) if token_stats else 0

        # Sample prompt header
        header_html = '<div class="section-header">Sample Prompt</div>'

        if not samples:
            samples_html = "<p>No prompt samples available</p>"
        else:
            samples_html = "".join(
                f'<div class="sample-box">'
                f"{sample[:_SAMPLE_MAX_LENGTH]}"
                f"{'...' if len(sample) > _SAMPLE_MAX_LENGTH else ''}</div>"
                for sample in samples[:5]
            )

        # Mean prompt length
        mean_html = f"""
        <div class="mean-container">
            <div class="mean-label">Mean Prompt Length</div>
            <div class="mean-value-primary">{mean_tokens:.2f} tokens</div>
        </div>
        """

        return f"{header_html}{samples_html}{mean_html}"

    def _generate_server_section(
        self, server_data: dict[str, Any], rate_type: str, num_benchmarks: int
    ) -> str:
        """Generate HTML for server configuration.

        Args:
            server_data: Dict with 'target' URL.
            rate_type: Rate type string.
            num_benchmarks: Number of benchmarks.

        Returns:
            HTML string for server section.
        """
        target = server_data.get("target", "N/A")

        # Parse URL to extract protocol and port
        protocol = "N/A"
        port = "N/A"
        if target != "N/A" and "://" in target:
            protocol = target.split("://")[0]
            rest = target.split("://")[1]
            if ":" in rest:
                port = rest.split(":")[1].split("/")[0]
            else:
                port = "80" if protocol == "http" else "443"

        return f"""
        <div class="flex-col">
            <div class="info-item">
                <div class="info-label">Target</div>
                <div class="info-value-primary">{target}</div>
            </div>
            <div class="grid-2col">
                <div class="info-item">
                    <div class="info-label">Type</div>
                    <div class="info-value-primary">{protocol}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Port</div>
                    <div class="info-value-primary">{port}</div>
                </div>
            </div>
            <div class="info-item">
                <div class="info-label">Number of Benchmarks</div>
                <div class="info-value-primary">{num_benchmarks}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Rate Type</div>
                <div class="info-value">
                    <span class="badge badge-primary">{rate_type}</span>
                </div>
            </div>
        </div>
        """

    def _generate_generations_section(self, generations_data: dict[str, Any]) -> str:
        """Generate HTML for generation samples.

        Args:
            generations_data: Dict with 'samples' and 'token_distributions'.

        Returns:
            HTML string for generations section.
        """
        samples = generations_data.get("samples", [])
        token_stats = generations_data.get("token_distributions", {}).get(
            "statistics", {}
        )
        mean_tokens = token_stats.get("mean", 0) if token_stats else 0

        # Sample generated header
        header_html = '<div class="section-header">Sample Generated</div>'

        if not samples:
            samples_html = "<p>No generation samples available</p>"
        else:
            samples_html = "".join(
                f'<div class="sample-box">'
                f"{sample[:_SAMPLE_MAX_LENGTH]}"
                f"{'...' if len(sample) > _SAMPLE_MAX_LENGTH else ''}</div>"
                for sample in samples[:5]
            )

        # Mean generated length
        mean_html = f"""
        <div class="mean-container">
            <div class="mean-label">Mean Generated Length</div>
            <div class="mean-value-primary">{mean_tokens:.2f} tokens</div>
        </div>
        """

        return f"{header_html}{samples_html}{mean_html}"

    def _create_histogram_chart(
        self, distribution_data: dict[str, Any], title: str, xaxis_title: str
    ) -> go.Figure:
        """Create a histogram chart with statistics overlay.

        Args:
            distribution_data: Dict with 'buckets' and 'statistics'.
            title: Chart title.
            xaxis_title: X-axis title.

        Returns:
            Plotly figure with histogram.
        """
        buckets = distribution_data.get("buckets", [])
        statistics = distribution_data.get("statistics", {})

        fig = self._create_figure(
            title=title, xaxis_title=xaxis_title, yaxis_title="Count"
        )

        if not buckets:
            return fig

        # Extract bucket data
        x_values = [b["value"] for b in buckets]
        counts = [b["count"] for b in buckets]

        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=counts,
                name="Count",
                marker_color=self.theme.PRIMARY,
                hovertemplate="Tokens: %{x}<br>Count: %{y}<extra></extra>",
            )
        )

        # Add mean line if available
        if statistics and "mean" in statistics:
            mean = statistics["mean"]
            fig.add_vline(
                x=mean,
                line_dash="dash",
                line_color=self.theme.SECONDARY,
                annotation_text=f"Mean: {mean:.1f}",
                annotation_position="top",
            )

        return fig

    def _create_requests_over_time_chart(
        self, requests_data: dict[str, Any]
    ) -> go.Figure:
        """Create requests over time bar chart.

        Args:
            requests_data: Dict with 'requests_over_time' containing 'buckets'.

        Returns:
            Plotly figure with requests over time.
        """
        requests_over_time = requests_data.get("requests_over_time", {})
        buckets = requests_over_time.get("buckets", [])
        bucket_width = requests_over_time.get("bucket_width", 1.0)

        fig = self._create_figure(
            title="Requests Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Request Count",
        )

        if not buckets:
            return fig

        # Extract bucket data
        x_values = [b["value"] for b in buckets]
        counts = [b["count"] for b in buckets]

        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=counts,
                name="Requests",
                marker_color=self.theme.TERTIARY,
                hovertemplate="Time: %{x:.1f}s<br>Requests: %{y}<extra></extra>",
                width=bucket_width * 0.8,  # Make bars slightly narrower than buckets
            )
        )

        return fig
