"""
load.py
=======
Loads cleaned tables into a SQL warehouse and creates the KPI views.

- DuckDB (default, local file data/traffic_safety.duckdb): no account needed.
- BigQuery (when GCP_PROJECT_ID is set): uses Application Default Credentials
  or GOOGLE_APPLICATION_CREDENTIALS. Credentials are never stored in the repo.

The view SQL in sql/views/ is written once and shared by both engines; table and
view names are written as {placeholders} and filled in per engine.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

from .config import DATA_PROCESSED, DATASETS, DUCKDB_PATH, SQL_DIR

log = logging.getLogger(__name__)

VIEW_DIR = SQL_DIR / "views"
TABLES = list(DATASETS)


def view_files(view_dir: Path = VIEW_DIR) -> list[Path]:
    return sorted(view_dir.glob("*.sql"))


def view_names(view_dir: Path = VIEW_DIR) -> list[str]:
    names = []
    for f in view_files(view_dir):
        m = re.search(r"CREATE OR REPLACE VIEW \{(\w+)\}", f.read_text())
        if m:
            names.append(m.group(1))
    return names


def render_sql(sql: str, qualify) -> str:
    """Replace {name} placeholders with engine-qualified identifiers."""
    return re.sub(r"\{(\w+)\}", lambda m: qualify(m.group(1)), sql)


def load_processed(processed_dir: Path = DATA_PROCESSED) -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(processed_dir / f"{t}.parquet") for t in TABLES}


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------

def load_duckdb(frames: dict[str, pd.DataFrame] | None = None, db_path: Path | str = DUCKDB_PATH,
                view_dir: Path = VIEW_DIR):
    """Create tables + views in DuckDB and return the open connection."""
    import duckdb

    frames = frames if frames is not None else load_processed()
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for name, df in frames.items():
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    for f in view_files(view_dir):
        con.execute(render_sql(f.read_text(), lambda n: n))
    log.info("DuckDB: loaded %d tables and %d views into %s", len(frames), len(view_files(view_dir)), db_path)
    return con


# ---------------------------------------------------------------------------
# BigQuery
# ---------------------------------------------------------------------------

def bigquery_settings() -> dict | None:
    project = os.getenv("GCP_PROJECT_ID")
    if not project:
        return None
    return {
        "project": project,
        "dataset": os.getenv("BQ_DATASET", "edmonton_traffic_safety"),
        "location": os.getenv("BQ_LOCATION", "northamerica-northeast1"),
    }


def load_bigquery(frames: dict[str, pd.DataFrame] | None = None, view_dir: Path = VIEW_DIR) -> str:
    """Load tables (WRITE_TRUNCATE) and create views in BigQuery. Returns the dataset id."""
    from google.cloud import bigquery

    settings = bigquery_settings()
    if settings is None:
        raise RuntimeError("GCP_PROJECT_ID is not set; see README 'BigQuery setup'")
    frames = frames if frames is not None else load_processed()
    client = bigquery.Client(project=settings["project"], location=settings["location"])
    dataset_id = f"{settings['project']}.{settings['dataset']}"
    ds = bigquery.Dataset(dataset_id)
    ds.location = settings["location"]
    client.create_dataset(ds, exists_ok=True)

    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    for name, df in frames.items():
        client.load_table_from_dataframe(df, f"{dataset_id}.{name}", job_config=job_config).result()
        log.info("BigQuery: loaded %s.%s (%d rows)", dataset_id, name, len(df))

    for f in view_files(view_dir):
        client.query(render_sql(f.read_text(), lambda n: f"`{dataset_id}.{n}`")).result()
    log.info("BigQuery: created %d views in %s", len(view_files(view_dir)), dataset_id)
    return dataset_id


def query_view(con, name: str) -> pd.DataFrame:
    """Read one view. DuckDB returns SUM(BIGINT) as HUGEINT, which pandas reads as float; cast those back."""
    rel = con.sql(f"SELECT * FROM {name}")
    df = rel.df()
    for col, typ in zip(rel.columns, rel.types):
        if str(typ) == "HUGEINT":
            df[col] = df[col].astype("Int64")
    return df


def query_views(con, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read every KPI view from an open DuckDB connection."""
    return {n: query_view(con, n) for n in (names or view_names())}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    con = load_duckdb()
    for n, df in query_views(con).items():
        print(f"{n:32s} {len(df):>6,d} rows")
    if bigquery_settings():
        load_bigquery()
