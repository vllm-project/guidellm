"""Base classes for HTML component generation."""

from abc import ABC, abstractmethod
from typing import Any

import plotly.graph_objects as go

from guidellm.benchmark.outputs.html.theme import PlotlyTheme


class PlotlyComponentBase(ABC):
    """Abstract base class for Plotly-based HTML components."""

    def __init__(self, theme: PlotlyTheme | None = None):
        """Initialize the component.

        Args:
            theme: Optional PlotlyTheme instance. If None, uses default theme.
        """
        self.theme = theme or PlotlyTheme()

    @abstractmethod
    def generate(self, data: dict[str, Any]) -> str | go.Figure:
        """Generate the component output.

        Args:
            data: Data dictionary containing component-specific data.

        Returns:
            Either an HTML string or a Plotly Figure object.
        """
        ...

    def _apply_theme_to_figure(self, fig: go.Figure) -> go.Figure:
        """Apply theme to a Plotly figure.

        Args:
            fig: Plotly figure to style.

        Returns:
            Styled figure.
        """
        layout_updates = self.theme.get_base_layout()
        fig.update_layout(**layout_updates)
        return fig

    def _create_figure(self, **layout_kwargs: Any) -> go.Figure:
        """Create a new figure with theme applied.

        Args:
            **layout_kwargs: Additional layout parameters to merge with theme.

        Returns:
            New Plotly figure with theme applied.
        """
        fig = go.Figure()
        base_layout = self.theme.get_base_layout()
        base_layout.update(layout_kwargs)
        fig.update_layout(**base_layout)
        return fig
