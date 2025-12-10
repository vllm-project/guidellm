"""
Console utilities for rich terminal output and status updates.

Provides an extended Rich console with custom formatting for status messages,
progress tracking, and tabular data display. Includes predefined color schemes,
status levels, icons, and styles for consistent terminal output across the
application. Supports multi-step operations with spinners and context managers
for clean progress reporting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from rich.console import Console as RichConsole
from rich.padding import Padding
from rich.status import Status
from rich.text import Text

__all__ = [
    "Colors",
    "Console",
    "ConsoleUpdateStep",
    "StatusIcons",
    "StatusLevel",
    "StatusStyles",
]

StatusLevel = Annotated[
    Literal[
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        "notset",
        "success",
    ],
    "Status level for console messages indicating severity or state",
]


class Colors:
    """
    Color constants for console styling.

    Provides standardized color schemes for different message types and branding.
    Colors are defined using Rich console color names or hex values.

    :cvar info: Color for informational messages
    :cvar progress: Color for progress indicators
    :cvar success: Color for successful operations
    :cvar warning: Color for warning messages
    :cvar error: Color for error messages
    :cvar primary: Primary brand color
    :cvar secondary: Secondary brand color
    :cvar tertiary: Tertiary brand color
    """

    # Core states
    info: str = "light_steel_blue"
    progress: str = "dark_slate_gray1"
    success: str = "chartreuse1"
    warning: str = "#FDB516"
    error: str = "orange_red1"

    # Branding
    primary: str = "#30A2FF"
    secondary: str = "#FDB516"
    tertiary: str = "#008080"


StatusIcons: Annotated[
    Mapping[str, str],
    "Mapping of status levels to unicode icon characters for visual indicators",
] = {
    "debug": "…",
    "info": "ℹ",
    "warning": "⚠",
    "error": "✖",
    "critical": "‼",
    "notset": "⟳",
    "success": "✔",
}

StatusStyles: Annotated[
    Mapping[str, str],
    "Mapping of status levels to Rich console style strings for colored output",
] = {
    "debug": "dim",
    "info": f"bold {Colors.info}",
    "warning": f"bold {Colors.warning}",
    "error": f"bold {Colors.error}",
    "critical": "bold red reverse",
    "notset": f"bold {Colors.progress}",
    "success": f"bold {Colors.success}",
}


@dataclass
class ConsoleUpdateStep:
    """
    Context manager for multi-step progress operations with spinner.

    Displays animated spinner during operation execution and allows dynamic
    status updates. Automatically stops spinner on exit and prints final
    status message. Designed for use with Python's `with` statement.

    Example:
    ::
        console = Console()
        with console.print_update_step("Processing data") as step:
            step.update("Loading files", "info")
            # ... do work ...
            step.finish("Completed successfully", status_level="success")

    :param console: The Console instance to use for output
    :param title: Initial progress message to display
    :param details: Optional additional details to show after completion
    :param status_level: Initial status level determining style and icon
    :param spinner: Spinner animation style name from Rich's spinner set
    """

    console: Console
    title: str
    details: Any | None = None
    status_level: StatusLevel = "info"
    spinner: str = "dots"
    _status: Status | None = None

    def __enter__(self) -> ConsoleUpdateStep:
        if self.console.quiet:
            return self

        style = StatusStyles.get(self.status_level, "bold")
        self._status = self.console.status(
            f"[{style}]{self.title}[/]",
            spinner=self.spinner,
        )
        self._status.__enter__()
        return self

    def update(self, title: str, status_level: StatusLevel | None = None):
        """
        Update the progress message and optionally the status level.

        :param title: New progress message to display
        :param status_level: Optional new status level to apply
        """
        self.title = title
        if status_level is not None:
            self.status_level = status_level

        if self._status:
            style = StatusStyles.get(self.status_level, "bold")
            self._status.update(status=f"[{style}]{title}[/]")

    def finish(
        self,
        title: str,
        details: Any | None = None,
        status_level: StatusLevel = "info",
    ):
        """
        Stop the spinner and print the final status message.

        :param title: Final completion message to display
        :param details: Optional additional information to show below message
        :param status_level: Status level for final message styling
        """
        self.title = title
        self.status_level = status_level

        if self._status:
            self._status.stop()

        self.console.print_update(title, details, status_level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._status:
            self._status.__exit__(exc_type, exc_val, exc_tb)


class Console(RichConsole):
    """
    Extended Rich console with custom formatting and status reporting.

    Enhances Rich's Console with specialized methods for status messages,
    progress tracking with spinners, and formatted table output. Provides
    consistent styling through predefined status levels, icons, and colors.
    Supports quiet mode to suppress non-critical output.

    Example:
    ::
        console = Console()
        console.print_update("Starting process", status="info")
        with console.print_update_step("Loading data") as step:
            step.update("Processing items")
            step.finish("Complete", status_level="success")
    """

    def print_update(
        self,
        title: str,
        details: Any | None = None,
        status: StatusLevel = "info",
    ):
        """
        Print a status message with icon and optional details.

        :param title: Main status message to display
        :param details: Optional additional details shown indented below message
        :param status: Status level determining icon and styling
        """
        icon = StatusIcons.get(status, "•")
        style = StatusStyles.get(status, "bold")
        line = Text.assemble(f"{icon} ", (title, style))
        self.print(line)
        self.print_update_details(details)

    def print_update_details(self, details: Any | None):
        """
        Print additional details indented below a status message.

        :param details: Content to display, converted to string and styled dimly
        """
        if details:
            block = Padding(
                Text.from_markup(str(details)),
                (0, 0, 0, 2),
                style=StatusStyles.get("debug", "dim"),
            )
            self.print(block)

    def print_update_step(
        self,
        title: str,
        status: StatusLevel = "info",
        details: Any | None = None,
        spinner: str = "dots",
    ) -> ConsoleUpdateStep:
        """
        Create a context manager for multi-step progress with spinner.

        :param title: Initial progress message to display
        :param status: Initial status level for styling
        :param details: Optional details to show after completion
        :param spinner: Spinner animation style name
        :return: ConsoleUpdateStep context manager for progress tracking
        """
        return ConsoleUpdateStep(
            console=self,
            title=title,
            details=details,
            status_level=status,
            spinner=spinner,
        )

    def print_tables(
        self,
        header_cols_groups: Sequence[Sequence[str | list[str]]],
        value_cols_groups: Sequence[Sequence[str | list[str]]],
        title: str | None = None,
        widths: Sequence[int] | None = None,
    ):
        """
        Print multiple tables with uniform column widths.

        :param header_cols_groups: List of header column groups for each table
        :param value_cols_groups: List of value column groups for each table
        :param title: Optional title to display before tables
        :param widths: Optional minimum column widths to enforce
        """
        if title is not None:
            self.print_update(title, None, "info")

        # Format all groups to determine uniform widths
        widths = widths or None
        headers = []
        values = []

        # Process all tables to get consistent widths
        for value_cols in value_cols_groups:
            formatted, widths = self._format_table_columns(value_cols, widths)
            values.append(formatted)
        for header_cols in header_cols_groups:
            formatted, widths = self._format_table_headers(header_cols, widths)
            headers.append(formatted)

        # Print each table
        for ind, (header, value) in enumerate(zip(headers, values, strict=False)):
            is_last = ind == len(headers) - 1
            self.print_table(
                header,
                value,
                widths=widths,
                apply_formatting=False,
                print_bottom_divider=is_last,
            )

    def print_table(
        self,
        header_cols: Sequence[str | list[str]],
        value_cols: Sequence[str | list[str]],
        title: str | None = None,
        widths: Sequence[int] | None = None,
        apply_formatting: bool = True,
        print_bottom_divider: bool = True,
    ):
        """
        Print a formatted table with headers and values.

        :param header_cols: List of header columns, each string or list of strings
        :param value_cols: List of value columns, each string or list of strings
        :param title: Optional title to display before table
        :param widths: Optional minimum column widths to enforce
        :param apply_formatting: Whether to calculate widths and format columns
        :param print_bottom_divider: Whether to print bottom border line
        """
        if title is not None:
            self.print_update(title, None, "info")

        # Format data
        values: list[list[str]]
        headers: list[list[str]]
        final_widths: list[int]

        if apply_formatting:
            values, final_widths = self._format_table_columns(value_cols, widths)
            headers, final_widths = self._format_table_headers(
                header_cols, final_widths
            )
        else:
            values = [col if isinstance(col, list) else [col] for col in value_cols]
            headers = [col if isinstance(col, list) else [col] for col in header_cols]
            final_widths = list(widths) if widths else []

        # Print table structure
        self.print_table_divider(final_widths, "=")
        self.print_table_headers(headers, final_widths)
        self.print_table_divider(final_widths, "-")
        self.print_table_values(values, final_widths)

        if print_bottom_divider:
            self.print_table_divider(final_widths, "=")

    def print_table_divider(self, widths: Sequence[int], char: str):
        """
        Print a horizontal divider line across table columns.

        :param widths: Column widths for divider line
        :param char: Character to use for divider line (e.g., '=', '-')
        """
        self.print_table_row(
            [""] * len(widths),
            widths=widths,
            spacer=char,
            cell_style="bold",
            divider_style="bold",
            edge_style="bold",
        )

    def print_table_headers(self, headers: Sequence[list[str]], widths: Sequence[int]):
        """
        Print header rows with support for column spanning.

        :param headers: List of header columns, each containing header row values
        :param widths: Column widths for proper alignment
        """
        if not headers or not headers[0]:
            return

        for row_idx in range(len(headers[0])):
            # Calculate widths for this header row, accounting for merged cells.
            row_widths = list(widths)
            for col_idx in range(len(headers)):
                if not headers[col_idx][row_idx]:
                    continue

                # Find span end
                span_end = col_idx + 1
                while span_end < len(headers) and not headers[span_end][row_idx]:
                    row_widths[span_end] = 0
                    span_end += 1

                # Set combined width for the first cell in span
                row_widths[col_idx] = sum(
                    widths[col] for col in range(col_idx, span_end)
                )

            # Print the header row
            self.print_table_row(
                values=[headers[col][row_idx] for col in range(len(headers))],
                widths=row_widths,
                cell_style="bold",
                divider_style="bold",
                edge_style="bold",
            )

    def print_table_values(self, values: Sequence[list[str]], widths: Sequence[int]):
        """
        Print all data rows in the table.

        :param values: List of value columns, each containing row values
        :param widths: Column widths for proper alignment
        """
        if not values:
            return

        for row_idx in range(len(values[0])):
            # Print the value row
            self.print_table_row(
                values=[values[col][row_idx] for col in range(len(values))],
                widths=widths,
                divider="|",
                edge_style="bold",
            )

    def print_table_row(
        self,
        values: Sequence[str],
        widths: Sequence[int] | None = None,
        spacer: str = " ",
        divider: str = "|",
        cell_style: str = "",
        value_style: str = "",
        divider_style: str = "",
        edge_style: str = "",
    ):
        """
        Print a single table row with custom styling.

        :param values: Cell values for the row
        :param widths: Column widths, defaults to value lengths
        :param spacer: Character for padding cells
        :param divider: Character separating columns
        :param cell_style: Rich style string for entire cells
        :param value_style: Rich style string for cell values only
        :param divider_style: Rich style string for column dividers
        :param edge_style: Rich style string for table edges
        """
        widths = widths or [len(val) for val in values]

        # Build styled cells
        cells = []
        for val, width in zip(values, widths, strict=True):
            cell = val.ljust(width, spacer)
            if value_style and val:
                cell = cell.replace(val, f"[{value_style}]{val}[/{value_style}]")
            if cell_style:
                cell = f"[{cell_style}]{cell}[/{cell_style}]"
            cells.append(cell)

        # Build and print row
        edge = f"[{edge_style}]{divider}[/{edge_style}]" if edge_style else divider
        inner = (
            f"[{divider_style}]{divider}[/{divider_style}]"
            if divider_style
            else divider
        )
        line = edge + inner.join(cells) + edge
        self.print(line, overflow="ignore", crop=False)

    def _format_table_headers(
        self,
        headers: Sequence[str | list[str]],
        col_widths: Sequence[int] | None = None,
        spacer: str = " ",
        min_padding: int = 1,
    ) -> tuple[list[list[str]], list[int]]:
        formatted, header_widths = self._format_table_columns(
            headers, col_widths, spacer, min_padding
        )

        if not formatted or not formatted[0]:
            return formatted, []

        # Merge identical adjacent headers row by row
        widths = list(col_widths) if col_widths else header_widths
        for row_idx in range(len(formatted[0])):
            last_value = None
            start_col = -1

            for col_idx in range(len(formatted) + 1):
                cur_value = (
                    formatted[col_idx][row_idx] if col_idx < len(formatted) else None
                )

                # Check if we should continue merging
                if (
                    col_idx < len(formatted)
                    and cur_value != ""
                    and cur_value == last_value
                    and (
                        row_idx == 0
                        or headers[start_col][row_idx - 1]
                        == headers[col_idx][row_idx - 1]
                    )
                ):
                    continue

                # Finalize previous
                if start_col >= 0:
                    # Clear merged cells to keep only the first
                    for col in range(start_col + 1, col_idx):
                        formatted[col][row_idx] = ""

                    # Adjust widths of columns in the merged span, if needed
                    if (required := len(formatted[start_col][row_idx])) > (
                        current := sum(widths[col] for col in range(start_col, col_idx))
                    ):
                        diff = required - current
                        cols_count = col_idx - start_col
                        per_col = diff // cols_count
                        extra = diff % cols_count

                        for col in range(start_col, col_idx):
                            widths[col] += per_col
                            if extra > 0:
                                widths[col] += 1
                                extra -= 1

                # Start new merge
                last_value = cur_value
                start_col = col_idx

        return formatted, widths

    def _format_table_columns(
        self,
        columns: Sequence[str | list[str]],
        col_widths: Sequence[int] | None = None,
        spacer: str = " ",
        min_padding: int = 1,
    ) -> tuple[list[list[str]], list[int]]:
        if not columns:
            return [], []

        # Normalize to list of lists
        max_rows = max(len(col) if isinstance(col, list) else 1 for col in columns)

        formatted = []
        for col in columns:
            col_list = col if isinstance(col, list) else [col]
            # Pad to max height
            col_list = col_list + [""] * (max_rows - len(col_list))
            # Add cell padding
            padding = spacer * min_padding
            col_list = [
                f"{padding}{item}{padding}" if item else "" for item in col_list
            ]
            formatted.append(col_list)

        # Calculate widths
        widths = [max(len(row) for row in col) for col in formatted]

        # Apply minimum widths if provided
        if col_widths is not None:
            widths = [
                max(width, min_w)
                for width, min_w in zip(widths, col_widths, strict=True)
            ]

        return formatted, widths
