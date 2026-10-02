"""
quality.py
==========
Data-quality bookkeeping and the reports/data_quality.md writer.

Cleaning functions record what they did into a QualityLog; every number in
the report is computed from that log or from the data, never typed by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class RuleResult:
    dataset: str
    rule: str
    rows_affected: int
    action: str          # "dropped", "flagged", "fixed", "check"
    detail: str = ""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class DatasetSummary:
    dataset: str
    source: str
    snapshot: str
    rows_ingested: int
    nulls: dict[str, int]
    null_handling: dict[str, str]
    exact_duplicates_removed: int = 0
    rows_dropped_invalid: int = 0
    rows_flagged: int = 0
    final_rows: int = 0

    @property
    def pct_retained(self) -> float:
        return 100.0 * self.final_rows / self.rows_ingested if self.rows_ingested else 0.0


@dataclass
class QualityLog:
    rules: list[RuleResult] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    summaries: dict[str, DatasetSummary] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def rule(self, dataset: str, rule: str, rows: int, action: str, detail: str = "") -> None:
        self.rules.append(RuleResult(dataset, rule, int(rows), action, detail))

    def check(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name, bool(passed), detail))

    def rules_for(self, dataset: str) -> list[RuleResult]:
        return [r for r in self.rules if r.dataset == dataset]


def null_counts(df: pd.DataFrame) -> dict[str, int]:
    """Nulls per column, treating empty/whitespace-only strings as null."""
    out = {}
    for c in df.columns:
        s = df[c]
        n = s.isna()
        if s.dtype == object:
            n = n | s.astype(str).str.strip().eq("")
        out[c] = int(n.sum())
    return out


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def render_report(log: QualityLog, generated_at: datetime | None = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    lines = [
        "# Data Quality Report",
        "",
        f"_Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} by `traffic_safety.clean`. "
        "All numbers are computed by the pipeline from the raw snapshots listed below._",
        "",
        "## Summary",
        "",
        "| Dataset | Rows ingested | Exact duplicates removed | Invalid rows dropped | Rows flagged (kept) | Final rows | % retained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in log.summaries.values():
        lines.append(
            f"| {s.dataset} | {s.rows_ingested:,} | {s.exact_duplicates_removed:,} | "
            f"{s.rows_dropped_invalid:,} | {s.rows_flagged:,} | {s.final_rows:,} | {_fmt_pct(s.pct_retained)} |"
        )

    passed = sum(c.passed for c in log.checks)
    lines += ["", f"**Validation checks:** {passed} of {len(log.checks)} passed.", ""]

    for s in log.summaries.values():
        lines += [
            f"## `{s.dataset}`",
            "",
            f"- Source: {s.source}",
            f"- Raw snapshot: `{s.snapshot}`",
            "",
            "### Nulls per column (raw)",
            "",
            "| Column | Nulls | Handling |",
            "|---|---:|---|",
        ]
        for col, n in s.nulls.items():
            lines.append(f"| `{col}` | {n:,} | {s.null_handling.get(col, 'n/a — no nulls')} |")
        lines += ["", "### Cleaning rules applied", "", "| Rule | Rows affected | Action | Detail |", "|---|---:|---|---|"]
        for r in log.rules_for(s.dataset):
            lines.append(f"| {r.rule} | {r.rows_affected:,} | {r.action} | {r.detail} |")
        lines.append("")

    lines += ["## Validation checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    for c in log.checks:
        lines.append(f"| {c.name} | {'PASS' if c.passed else '**FAIL**'} | {c.detail} |")

    if log.notes:
        lines += ["", "## Known data issues and how they are handled", ""]
        lines += [f"{i}. {n}" for i, n in enumerate(log.notes, 1)]

    lines.append("")
    return "\n".join(lines)


def write_report(log: QualityLog, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(log))
