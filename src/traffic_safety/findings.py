"""
findings.py
===========
Runs the named queries in sql/analysis/findings.sql against DuckDB and writes
reports/findings_evidence.md (query text + live result) so every number in the
findings brief can be traced to the query that produced it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import REPORTS_DIR, SQL_DIR
from .load import load_duckdb, query_view

FINDINGS_SQL = SQL_DIR / "analysis" / "findings.sql"


def parse_named_queries(text: str) -> dict[str, str]:
    """Split a file into {name: sql} on '-- name: <id>' markers."""
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_findings(con, sql_path: Path = FINDINGS_SQL) -> dict[str, tuple[str, pd.DataFrame]]:
    out = {}
    for name, sql in parse_named_queries(sql_path.read_text()).items():
        con.execute(f"CREATE OR REPLACE TEMP VIEW _finding AS {sql.rstrip(';')}")
        out[name] = (sql, query_view(con, "_finding"))
    return out


def write_evidence(results: dict[str, tuple[str, pd.DataFrame]], path: Path = REPORTS_DIR / "findings_evidence.md") -> Path:
    lines = ["# Findings evidence", "",
             f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python -m traffic_safety.findings` "
             "from `sql/analysis/findings.sql`. Each result below is the live output of the query above it._", ""]
    for name, (sql, df) in results.items():
        lines += [f"## `{name}`", "", "```sql", sql, "```", "", df.to_markdown(index=False), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    con = load_duckdb()
    res = run_findings(con)
    print(write_evidence(res))
    for name, (_, df) in res.items():
        print(f"\n== {name}\n{df.to_string(index=False)}")
