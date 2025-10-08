from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rich.console import Console as RichConsole
from rich.padding import Padding
from rich.status import Status
from rich.text import Text

__all__ = [
    "Colors",
    "Console",
    "ConsoleUpdateStep",
    "StatusIcons",
    "StatusStyles",
]


class Colors:
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


StatusIcons: Mapping[str, str] = {
    "debug": "…",
    "info": "ℹ",
    "warning": "⚠",
    "error": "✖",
    "critical": "‼",
    "notset": "⟳",
    "success": "✔",
}

StatusStyles: Mapping[str, str] = {
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
    console: Console
    title: str
    details: Any | None = None
    status_level: Literal[
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        "notset",
        "success",
    ] = "info"
    spinner: str = "dots"
    _status: Status | None = None

    def __enter__(self):
        if self.console.quiet:
            return self

        self._status = self.console.status(
            f"[{StatusStyles.get(self.status_level, 'bold')}]{self.title}[/]",
            spinner=self.spinner,
        )
        self._status.__enter__()
        return self

    def update(
        self,
        title: str,
        status_level: Literal[
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "notset",
            "success",
        ]
        | None = None,
    ):
        self.title = title
        if status_level is not None:
            self.status_level = status_level
        if self._status:
            self._status.update(
                status=f"[{StatusStyles.get(self.status_level, 'bold')}]{title}[/]"
            )

    def finish(
        self,
        title: str,
        details: Any | None = None,
        status_level: Literal[
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "notset",
            "success",
        ] = "info",
    ):
        self.title = title
        self.status_level = status_level
        if self._status:
            self._status.stop()
        self.console.print_update(title, details, status_level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._status:
            return self._status.__exit__(exc_type, exc_val, exc_tb)
        return False


class Console(RichConsole):
    def print_update(
        self,
        title: str,
        details: str | None = None,
        status: Literal[
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "notset",
            "success",
        ] = "info",
    ) -> None:
        icon = StatusIcons.get(status, "•")
        style = StatusStyles.get(status, "bold")
        line = Text.assemble(f"{icon} ", (title, style))
        self.print(line)
        self.print_update_details(details)

    def print_update_details(self, details: Any | None):
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
        status: Literal[
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "notset",
            "success",
        ] = "info",
        details: Any | None = None,
        spinner: str = "dots",
    ) -> ConsoleUpdateStep:
        return ConsoleUpdateStep(
            console=self,
            title=title,
            details=details,
            status_level=status,
            spinner=spinner,
        )
