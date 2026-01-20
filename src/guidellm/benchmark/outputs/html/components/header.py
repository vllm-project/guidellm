"""Header component for HTML reports."""

from typing import Any

from guidellm.benchmark.outputs.html.components.base import PlotlyComponentBase
from guidellm.utils.functions import safe_format_timestamp


class HeaderComponent(PlotlyComponentBase):
    """Generates the page header with run metadata."""

    def generate(self, data: dict[str, Any]) -> str:
        """Generate header HTML.

        Args:
            data: Dictionary containing:
                - model: Dict with 'name' key
                - timestamp: ISO format timestamp or datetime
                - dataset: Dict with 'name' key (optional)
                - task: Task name (optional)

        Returns:
            HTML string for the header section.
        """
        model_name = data.get("model", {}).get("name", "N/A")
        timestamp = safe_format_timestamp(
            data.get("timestamp"),
            format_="%B %d %Y at %H:%M:%S",
        )
        dataset_name = data.get("dataset", {}).get("name", "N/A")
        task = data.get("task", "N/A")

        return f"""
        <div class="header">
            <h1>GuideLLM Benchmark Report</h1>
            <div class="info-row">
                <div class="info-item">
                    <div class="info-label">Model</div>
                    <div class="info-value">{model_name}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Timestamp</div>
                    <div class="info-value">{timestamp}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Dataset</div>
                    <div class="info-value">{dataset_name}</div>
                </div>
                {
            f'''<div class="info-item">
                    <div class="info-label">Task</div>
                    <div class="info-value">{task}</div>
                </div>'''
            if task != "N/A"
            else ""
        }
            </div>
        </div>
        """
