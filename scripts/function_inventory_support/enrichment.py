"""Git and test-timing enrichment helpers for function inventories."""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

from function_inventory_support.model import FunctionKey, FunctionMetric


def _attach_git_metrics(
    root: Path,
    metrics: dict[FunctionKey, FunctionMetric],
) -> None:
    file_stats = _git_file_stats(root, {metric.path for metric in metrics.values()})
    blame_by_file = _git_blame_commits_by_file(
        root,
        {metric.path for metric in metrics.values()},
    )
    for metric in metrics.values():
        commit_count, churn_lines = file_stats.get(metric.path, (0, 0))
        metric.git_file_commit_count = commit_count
        metric.git_file_churn_lines = churn_lines
        line_commits = blame_by_file.get(metric.path, ())
        if line_commits:
            start = max(metric.line - 1, 0)
            end = min(metric.end_line, len(line_commits))
            metric.git_blame_commit_count = len(set(line_commits[start:end]))


def _git_file_stats(
    root: Path,
    paths: set[Path],
) -> dict[Path, tuple[int, int]]:
    if not (root / ".git").exists() or not paths:
        return {}
    path_texts = sorted(path.as_posix() for path in paths)
    output = _git_output(
        root, ["log", "--format=commit:%H", "--numstat", "--", *path_texts]
    )
    if output is None:
        return {}
    commits_by_path: dict[str, set[str]] = defaultdict(set)
    churn_by_path: dict[str, int] = defaultdict(int)
    current_commit = ""
    for line in output.splitlines():
        if line.startswith("commit:"):
            current_commit = line.removeprefix("commit:")
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, changed_path = parts[0], parts[1], parts[-1]
        if changed_path not in path_texts:
            continue
        commits_by_path[changed_path].add(current_commit)
        churn_by_path[changed_path] += _numstat_count(added) + _numstat_count(deleted)
    return {
        Path(path): (len(commits_by_path[path]), churn_by_path[path])
        for path in path_texts
    }


def _git_blame_commits_by_file(
    root: Path,
    paths: set[Path],
) -> dict[Path, tuple[str, ...]]:
    if not (root / ".git").exists():
        return {}
    result: dict[Path, tuple[str, ...]] = {}
    for path in sorted(paths):
        output = _git_output(root, ["blame", "--line-porcelain", "--", path.as_posix()])
        if output is None:
            continue
        commits: list[str] = []
        for line in output.splitlines():
            if re.fullmatch(r"[0-9a-f]{40} .+", line):
                commits.append(line.split(" ", maxsplit=1)[0])
        result[path] = tuple(commits)
    return result


def _git_output(root: Path, args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def _numstat_count(value: str) -> int:
    if value == "-":
        return 0
    return int(value)
