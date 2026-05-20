"""Data model and render helpers for function inventory reports."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TextIO

FunctionKind = Literal[
    "function",
    "method",
    "staticmethod",
    "classmethod",
    "property",
    "nested_function",
]
ApiSurface = Literal["exported_api", "public_internal", "private"]


@dataclass(frozen=True, order=True)
class FunctionKey:
    """Stable module-qualified function identity for audit rows.

    Inputs: module path and qualified function name.
    Outputs: hashable key whose display text survives line-number churn.
    """

    module: str
    qualname: str

    def display_name(self) -> str:
        """Return the module-qualified function name used in reports."""
        return f"{self.module}.{self.qualname}"


@dataclass
class FunctionMetric:
    """Mutable metric record for one source function.

    Inputs: source span, docs, typing, complexity, use sites, and test sites.
    Outputs: CSV-ready counts that keep repeated same-line uses deduplicated.
    """

    key: FunctionKey
    path: Path
    line: int
    end_line: int
    kind: FunctionKind
    public: bool
    total_lines: int
    code_lines: int
    docstring_lines: int
    docstring_span_lines: int
    cyclomatic_complexity: int
    domain: str
    api_surface: ApiSurface
    git_file_commit_count: int = 0
    git_file_churn_lines: int = 0
    git_blame_commit_count: int = 0
    direct_call_sites: set[str] = field(default_factory=set)
    ambiguous_name_sites: set[str] = field(default_factory=set)
    direct_test_functions: set[str] = field(default_factory=set)
    direct_test_modules: set[str] = field(default_factory=set)

    def row(self) -> dict[str, str]:
        """Render this metric as one flat CSV/Markdown row."""
        return {
            "function": self.key.display_name(),
            "module": self.key.module,
            "kind": self.kind,
            "path": self.path.as_posix(),
            "line": str(self.line),
            "code_lines": str(self.code_lines),
            "docstring_lines": str(self.docstring_lines),
            "cyclomatic_complexity": str(self.cyclomatic_complexity),
            "domain": self.domain,
            "api_surface": self.api_surface,
            "git_file_commit_count": str(self.git_file_commit_count),
            "git_file_churn_lines": str(self.git_file_churn_lines),
            "git_blame_commit_count": str(self.git_blame_commit_count),
            "direct_call_site_count": str(len(self.direct_call_sites)),
            "direct_test_function_count": str(len(self.direct_test_functions)),
            "risk_score": f"{self.risk_score():.1f}",
        }

    def risk_score(self) -> float:
        """Return a rough prioritization score for review and refactor queues."""
        api_weight = {
            "exported_api": 2.0,
            "public_internal": 1.5,
            "private": 1.0,
        }[self.api_surface]
        test_weight = 2.0 if not self.direct_test_functions else 1.0
        churn_weight = 1.0 + min(self.git_file_commit_count, 20) / 20
        return (self.code_lines + 5 * self.cyclomatic_complexity) * (
            api_weight * test_weight * churn_weight
        )


FIELDNAMES = (
    "function",
    "module",
    "kind",
    "path",
    "line",
    "code_lines",
    "docstring_lines",
    "cyclomatic_complexity",
    "domain",
    "api_surface",
    "git_file_commit_count",
    "git_file_churn_lines",
    "git_blame_commit_count",
    "direct_call_site_count",
    "direct_test_function_count",
    "risk_score",
)


def write_csv(rows: list[dict[str, str]], output: TextIO) -> None:
    """Write exact per-function inventory rows as CSV."""
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], output: TextIO) -> None:
    """Write a compact Markdown table for quick inventory review."""
    columns = (
        "function",
        "code_lines",
        "docstring_lines",
        "cyclomatic_complexity",
        "api_surface",
        "domain",
        "git_file_commit_count",
        "direct_test_function_count",
        "risk_score",
    )
    output.write("| " + " | ".join(columns) + " |\n")
    output.write("| " + " | ".join("---" for _ in columns) + " |\n")
    for row in rows:
        cells = (_escape_markdown(row[column]) for column in columns)
        output.write("| " + " | ".join(cells) + " |\n")


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|")
