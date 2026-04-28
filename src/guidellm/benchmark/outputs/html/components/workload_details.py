"""Workload details component for HTML reports."""

from typing import Any

import plotly.graph_objects as go

from guidellm.benchmark.outputs.html.components.base import PlotlyComponentBase

# Maximum characters to display for sample text
_SAMPLE_MAX_LENGTH = 200


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
        rate_types = data.get("rate_types", ["N/A"])
        num_benchmarks = requests_data.get("num_benchmarks", 0)
        total_requests = requests_data.get("total_requests", 0)
        requests_per_benchmark = requests_data.get(
            "requests_per_benchmark",
            {"successful": [], "incomplete": [], "errored": []},
        )

        # Extract samples for unified JavaScript rotation
        prompt_samples = prompts_data.get("samples", [])[:5]
        generation_samples = generations_data.get("samples", [])[:5]

        # Prepare samples for JavaScript (escape quotes and truncate)
        prompt_samples_js = (
            [
                s[:_SAMPLE_MAX_LENGTH].replace("\\", "\\\\").replace('"', '\\"')
                for s in prompt_samples
            ]
            if prompt_samples
            else []
        )

        generation_samples_js = (
            [
                s[:_SAMPLE_MAX_LENGTH].replace("\\", "\\\\").replace('"', '\\"')
                for s in generation_samples
            ]
            if generation_samples
            else []
        )

        # Build HTML sections
        prompts_html = self._generate_prompts_section(prompts_data, prompt_samples_js)
        server_html = self._generate_server_section(
            server_data, rate_types, num_benchmarks, total_requests
        )
        generations_html = self._generate_generations_section(
            generations_data, generation_samples_js
        )

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
        requests_fig = self._create_requests_per_benchmark_chart(requests_per_benchmark)

        # Convert figures to HTML
        prompt_chart_html = prompt_tokens_fig.to_html(
            include_plotlyjs=False, div_id="prompt-tokens-chart"
        )
        output_chart_html = output_tokens_fig.to_html(
            include_plotlyjs=False, div_id="output-tokens-chart"
        )
        requests_chart_html = requests_fig.to_html(
            include_plotlyjs=False, div_id="requests-per-benchmark-chart"
        )

        # Create unified JavaScript for synchronized sample rotation
        unified_script = ""
        if prompt_samples_js and generation_samples_js:
            unified_script = f"""
            <script>
            (function() {{
                const promptSamples = {prompt_samples_js};
                const generationSamples = {generation_samples_js};
                let currentIndex = 0;

                const promptDisplayEl = document.getElementById(
                    'prompt-sample-display'
                );
                const generationDisplayEl = document.getElementById(
                    'generation-sample-display'
                );

                function rotateSamples() {{
                    // Fade out both displays
                    promptDisplayEl.classList.add('fade-out');
                    generationDisplayEl.classList.add('fade-out');

                    // Wait for fade out, then update both and fade in
                    setTimeout(function() {{
                        currentIndex = (currentIndex + 1) % promptSamples.length;
                        promptDisplayEl.textContent =
                            promptSamples[currentIndex] + '...';
                        generationDisplayEl.textContent =
                            generationSamples[currentIndex] + '...';
                        promptDisplayEl.classList.remove('fade-out');
                        generationDisplayEl.classList.remove('fade-out');
                    }}, 500);
                }}

                setInterval(rotateSamples, 3000);
            }})();
            </script>
            """

        return f"""
        <div class="section">
            <h2>Workload Details</h2>
            <div class="grid-3col">
                <h3>Prompts</h3>
                <h3>Run Configuration</h3>
                <h3>Outputs</h3>

                <div class="content-section">{prompts_html}</div>
                <div class="content-section">{server_html}</div>
                <div class="content-section">{generations_html}</div>

                <div class="chart-section">{prompt_chart_html}</div>
                <div class="chart-section">{requests_chart_html}</div>
                <div class="chart-section">{output_chart_html}</div>
            </div>
            {unified_script}
        </div>
        """

    def _generate_prompts_section(
        self, prompts_data: dict[str, Any], samples_js: list[str]
    ) -> str:
        """Generate HTML for prompts samples.

        Args:
            prompts_data: Dict with 'samples' and 'token_distributions'.
            samples_js: Prepared JavaScript-safe sample strings.

        Returns:
            HTML string for prompts section.
        """
        token_stats = prompts_data.get("token_distributions", {}).get("statistics", {})
        mean_tokens = token_stats.get("mean", 0) if token_stats else 0

        # Sample prompt header
        header_html = '<div class="section-header">Sample Prompt</div>'

        if not samples_js:
            samples_html = "<p>No prompt samples available</p>"
        else:
            samples_html = f"""
            <div class="sample-carousel">
                <div class="sample-display" id="prompt-sample-display">
                    {samples_js[0]}...
                </div>
            </div>
            """

        # Mean prompt length
        mean_html = f"""
        <div class="mean-container">
            <div class="mean-label">Mean Prompt Length</div>
            <div class="mean-value-primary">{mean_tokens:.2f} tokens</div>
        </div>
        """

        return f"{header_html}{samples_html}{mean_html}"

    def _generate_server_section(
        self,
        _server_data: dict[str, Any],
        rate_types: list[str],
        num_benchmarks: int,
        total_requests: int,
    ) -> str:
        """Generate HTML for benchmark configuration.

        Args:
            _server_data: Dict with server data (unused but kept for compatibility).
            rate_types: List of rate type strings in execution order.
            num_benchmarks: Number of benchmarks.
            total_requests: Total number of requests across all benchmarks.

        Returns:
            HTML string for benchmark configuration section.
        """
        # Generate multiple rate type badges
        rate_badges = " ".join(
            f'<span class="badge badge-primary">{rt}</span>' for rt in rate_types
        )

        rate_label = "Profile" + ("s" if len(rate_types) > 1 else "")

        return f"""
        <div class="flex-col">
            <div class="info-item">
                <div class="info-label">Number of Benchmarks</div>
                <div class="info-value-primary">{num_benchmarks}</div>
            </div>
            <div class="info-item">
                <div class="info-label">{rate_label}</div>
                <div class="info-value">
                    {rate_badges}
                </div>
            </div>
        </div>
        <div class="mean-container">
            <div class="mean-label">Total Request Count</div>
            <div class="mean-value-primary">{total_requests}</div>
        </div>
        """

    def _generate_generations_section(
        self, generations_data: dict[str, Any], samples_js: list[str]
    ) -> str:
        """Generate HTML for generation samples.

        Args:
            generations_data: Dict with 'samples' and 'token_distributions'.
            samples_js: Prepared JavaScript-safe sample strings.

        Returns:
            HTML string for generations section.
        """
        token_stats = generations_data.get("token_distributions", {}).get(
            "statistics", {}
        )
        mean_tokens = token_stats.get("mean", 0) if token_stats else 0

        # Sample output header
        header_html = '<div class="section-header">Sample Output</div>'

        if not samples_js:
            samples_html = "<p>No output samples available</p>"
        else:
            samples_html = f"""
            <div class="sample-carousel">
                <div class="sample-display" id="generation-sample-display">
                    {samples_js[0]}...
                </div>
            </div>
            """

        # Mean output length
        mean_html = f"""
        <div class="mean-container">
            <div class="mean-label">Mean Output Length</div>
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
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Count",
            autosize=True,
            width=None,
        )

        if not buckets:
            return fig

        # Extract bucket data
        x_values = [b["value"] for b in buckets]
        counts = [b["count"] for b in buckets]

        # Calculate appropriate bar width with max limit
        if len(x_values) > 1:
            data_range = max(x_values) - min(x_values)
            calculated_width = data_range / len(x_values) * 0.8
            max_width = 20  # Maximum bar width in data units
            bar_width = min(calculated_width, max_width)
        else:
            bar_width = 10  # Single bar default width

        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=counts,
                name="Count",
                marker_color=self.theme.PRIMARY,
                hovertemplate="Tokens: %{x}<br>Count: %{y}<extra></extra>",
                width=bar_width,
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

    def _create_requests_per_benchmark_chart(
        self, requests_per_benchmark: dict[str, list[int]]
    ) -> go.Figure:
        """Create requests per benchmark stacked bar chart.

        Args:
            requests_per_benchmark: Dict with 'successful', 'incomplete',
                and 'errored' lists.

        Returns:
            Plotly figure with stacked requests per benchmark.
        """
        fig = self._create_figure(
            title="Requests per Benchmark",
            xaxis_title="Benchmark Index",
            yaxis_title="Request Count",
            autosize=True,
            width=None,
        )

        successful = requests_per_benchmark.get("successful", [])
        incomplete = requests_per_benchmark.get("incomplete", [])
        errored = requests_per_benchmark.get("errored", [])

        if not successful and not incomplete and not errored:
            return fig

        # Create x values as benchmark indices (1-based for display)
        num_benchmarks = max(len(successful), len(incomplete), len(errored))
        x_values = list(range(1, num_benchmarks + 1))

        # Add stacked bars - order matters for stacking
        # Add successful requests bar using primary theme color
        if successful:
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=successful,
                    name="Successful",
                    marker_color=self.theme.PRIMARY,
                    hovertemplate=(
                        "Benchmark: %{x}<br>Successful: %{y}<extra></extra>"
                    ),
                )
            )

        # Add incomplete requests bar using secondary theme color (lavender)
        if incomplete:
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=incomplete,
                    name="Incomplete",
                    marker_color=self.theme.SECONDARY,
                    hovertemplate=(
                        "Benchmark: %{x}<br>Incomplete: %{y}<extra></extra>"
                    ),
                )
            )

        # Add errored requests bar using error theme color
        if errored:
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=errored,
                    name="Errored",
                    marker_color=self.theme.ERROR,
                    hovertemplate=("Benchmark: %{x}<br>Errored: %{y}<extra></extra>"),
                )
            )

        # Enable stacked bar mode with legend at bottom
        fig.update_layout(
            barmode="stack",
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.2,
                "xanchor": "center",
                "x": 0.5,
            },
            margin={"b": 80},
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
            autosize=True,
            width=None,
        )

        if not buckets:
            return fig

        # Extract bucket data
        x_values = [b["value"] for b in buckets]
        counts = [b["count"] for b in buckets]

        # Calculate bar width with max limit
        calculated_width = bucket_width * 0.8
        max_width = 50  # Maximum width in seconds for requests chart
        bar_width = min(calculated_width, max_width)

        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=counts,
                name="Requests",
                marker_color=self.theme.TERTIARY,
                hovertemplate="Time: %{x:.1f}s<br>Requests: %{y}<extra></extra>",
                width=bar_width,
            )
        )

        return fig
