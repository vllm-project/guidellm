"""Theme configuration for Plotly-based HTML reports.

This module provides Material-UI dark theme colors and Plotly layout configurations
that match the original Next.js UI appearance.
"""

from typing import Any


class PlotlyTheme:
    """Material-UI dark theme configuration for Plotly charts."""

    # Material-UI dark theme colors
    BACKGROUND = "#121212"
    SURFACE = "#1e1e1e"
    PRIMARY = "#90caf9"  # Blue
    SECONDARY = "#ce93d8"  # Purple
    TERTIARY = "#80cbc4"  # Teal
    QUATERNARY = "#fff59d"  # Yellow
    SUCCESS = "#66bb6a"  # Green
    ERROR = "#f44336"  # Red
    TEXT_PRIMARY = "rgba(255, 255, 255, 0.87)"
    TEXT_SECONDARY = "rgba(255, 255, 255, 0.6)"
    TEXT_DISABLED = "rgba(255, 255, 255, 0.38)"

    # Font family
    FONT_FAMILY = "Spezia, Roboto, -apple-system, BlinkMacSystemFont, sans-serif"

    # Chart colors palette (for multi-line charts)
    CHART_COLORS = [
        PRIMARY,
        SECONDARY,
        TERTIARY,
        QUATERNARY,
        "#ef5350",  # Red
        "#ab47bc",  # Deep purple
        "#42a5f5",  # Light blue
        "#26a69a",  # Teal
    ]

    @classmethod
    def get_base_layout(cls) -> dict[str, Any]:
        """Get base Plotly layout configuration.

        Returns:
            Dictionary with Plotly layout settings for dark theme.
        """
        return {
            "paper_bgcolor": cls.BACKGROUND,
            "plot_bgcolor": cls.SURFACE,
            "font": {
                "family": cls.FONT_FAMILY,
                "size": 12,
                "color": cls.TEXT_PRIMARY,
            },
            "xaxis": {
                "gridcolor": "rgba(255, 255, 255, 0.1)",
                "zerolinecolor": "rgba(255, 255, 255, 0.2)",
                "color": cls.TEXT_SECONDARY,
            },
            "yaxis": {
                "gridcolor": "rgba(255, 255, 255, 0.1)",
                "zerolinecolor": "rgba(255, 255, 255, 0.2)",
                "color": cls.TEXT_SECONDARY,
            },
            "legend": {
                "font": {"color": cls.TEXT_PRIMARY},
                "bgcolor": "rgba(0, 0, 0, 0.5)",
            },
            "hovermode": "closest",
            "hoverlabel": {
                "bgcolor": cls.SURFACE,
                "font": {"family": cls.FONT_FAMILY, "color": cls.TEXT_PRIMARY},
            },
            "shapes": [],
            "margin": {"pad": 0},
        }

    @classmethod
    def get_css(cls) -> str:
        """Get CSS stylesheet for HTML reports.

        Returns:
            CSS string with Material-UI dark theme styles.
        """
        return f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: {cls.FONT_FAMILY};
            background: linear-gradient(105deg, black, {cls.SURFACE});
            color: {cls.TEXT_PRIMARY};
            line-height: 1.6;
            padding: 2rem;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-weight: 500;
            margin-bottom: 1rem;
        }}

        h1 {{
            font-size: 2.5rem;
            color: {cls.PRIMARY};
        }}

        h2 {{
            font-size: 2rem;
            color: {cls.SECONDARY};
        }}

        h3 {{
            font-size: 1.5rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            margin-bottom: 2rem;
            padding: 1.5rem;
            background-color: {cls.SURFACE};
            border-radius: 8px;
        }}

        .section {{
            margin-bottom: 2rem;
            padding: 1.5rem;
            background-color: {cls.SURFACE};
            border-radius: 8px;
        }}

        .footer {{
            margin-top: 2rem;
            padding: 1rem;
            text-align: center;
            color: {cls.TEXT_SECONDARY};
            font-size: 0.875rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.875rem;
            font-weight: 500;
            margin: 0.25rem;
        }}

        .badge-primary {{
            background-color: {cls.PRIMARY};
            color: #000;
        }}

        .badge-secondary {{
            background-color: {cls.SECONDARY};
            color: #000;
        }}

        .badge-success {{
            background-color: {cls.SUCCESS};
            color: #fff;
        }}

        .badge-error {{
            background-color: {cls.ERROR};
            color: #fff;
        }}

        .info-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }}

        .info-item {{
            flex: 1;
            min-width: 200px;
        }}

        .info-label {{
            color: {cls.TEXT_SECONDARY};
            font-size: 0.875rem;
            margin-bottom: 0.25rem;
        }}

        .info-value {{
            color: {cls.TEXT_PRIMARY};
            font-size: 1rem;
            font-weight: 500;
        }}

        a {{
            color: {cls.PRIMARY};
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        /* Workload details styles */
        .section-header {{
            margin-bottom: 0.5rem;
            color: rgba(255,255,255,0.6);
            font-size: 0.875rem;
            text-transform: uppercase;
        }}

        .sample-box {{
            padding: 0.5rem;
            margin: 0.25rem 0;
            background-color: rgba(255,255,255,0.05);
            border-radius: 4px;
            font-size: 0.875rem;
            color: rgba(255, 255, 255, 0.7);
        }}

        .sample-carousel {{
            margin: 0.5rem 0;
        }}

        .sample-display {{
            padding: 0.5rem;
            background-color: rgba(255,255,255,0.05);
            border-radius: 4px;
            font-size: 0.875rem;
            color: rgba(255, 255, 255, 0.7);
            min-height: 4rem;
            display: flex;
            align-items: center;
            opacity: 1;
            transition: opacity 0.5s ease-in-out;
        }}

        .sample-display.fade-out {{
            opacity: 0;
        }}

        .mean-container {{
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        .mean-label {{
            color: rgba(255,255,255,0.6);
            font-size: 0.875rem;
            text-transform: uppercase;
        }}

        .mean-value-primary {{
            color: {cls.PRIMARY};
            font-size: 2rem;
            font-weight: 500;
        }}

        .info-value-primary {{
            color: {cls.PRIMARY};
            word-break: break-all;
        }}

        .grid-3col {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: auto auto 1fr;
            gap: 1.5rem;
            column-gap: 1.5rem;
            row-gap: 1rem;
            margin-top: 1rem;
            align-items: start;
        }}

        .grid-3col h3 {{
            align-self: start;
        }}

        .grid-3col .content-section {{
            align-self: start;
        }}

        .grid-3col .chart-section {{
            align-self: stretch;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            min-height: 300px;
        }}

        .grid-3col .chart-section > div {{
            width: 100% !important;
        }}

        .flex-col {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }}

        .grid-2col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }}

        /* Workload metrics grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            grid-template-rows: repeat(2, 1fr);
            gap: 2rem;
            margin-top: 1.5rem;
        }}

        .metric-card {{
            background-color: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .metric-card h3 {{
            font-size: 0.875rem;
            font-weight: 500;
            letter-spacing: 0.05em;
            color: rgba(255, 255, 255, 0.6);
            margin-bottom: 1rem;
            text-transform: uppercase;
        }}

        /* Plotly chart styling */
        .plotly {{
            border-radius: 8px;
            overflow: hidden;
        }}

        .js-plotly-plot {{
            border-radius: 8px;
            overflow: hidden;
        }}

        .plotly .main-svg {{
            border-radius: 8px;
        }}
        """
