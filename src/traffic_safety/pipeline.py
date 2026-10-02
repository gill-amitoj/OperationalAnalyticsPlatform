"""
pipeline.py
===========
End-to-end run:  ingest -> clean/validate -> load (DuckDB, + BigQuery if
configured) -> Tableau extracts -> Plotly dashboard.

    python -m traffic_safety.pipeline            # use cache when source unchanged
    python -m traffic_safety.pipeline --force    # re-download everything
    python -m traffic_safety.pipeline --skip-ingest
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from . import clean, dashboard, ingest
from .config import TABLEAU_DIR
from .load import bigquery_settings, load_bigquery, load_duckdb, query_views

log = logging.getLogger("traffic_safety")


def export_tableau(views: dict[str, pd.DataFrame], out_dir: Path = TABLEAU_DIR) -> list[Path]:
    """One clean CSV per KPI view (file name = view name without the v_ prefix)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in views.items():
        path = out_dir / f"{name.removeprefix('v_')}.csv"
        # Views have no ORDER BY; sort so weekly refreshes only diff when data changes.
        df.sort_values(list(df.columns), kind="mergesort").to_csv(path, index=False)
        paths.append(path)
    return paths


def run(force: bool = False, skip_ingest: bool = False) -> None:
    if not skip_ingest:
        ingest.ingest_all(force=force)
    frames = clean.run()
    con = load_duckdb(frames)
    views = query_views(con)
    if bigquery_settings():
        load_bigquery(frames)
    else:
        log.info("BigQuery not configured (GCP_PROJECT_ID unset); DuckDB only")
    paths = export_tableau(views)
    log.info("Wrote %d Tableau extracts to %s", len(paths), TABLEAU_DIR)
    log.info("Dashboard: %s", dashboard.build(views))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the cache is current")
    parser.add_argument("--skip-ingest", action="store_true", help="use the latest raw snapshots as-is")
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(force=args.force, skip_ingest=args.skip_ingest)


if __name__ == "__main__":
    main()
