"""
ingest.py
=========
Pulls datasets from the data.edmonton.ca Socrata (SODA 2.0) API into an
immutable local raw cache.

- Paginates with $limit/$offset ordered by the stable system id (:id).
- Retries transient HTTP failures (429/5xx) with exponential backoff.
- Verifies the downloaded row count against the API's count(*); the SODA
  endpoint has been observed returning a header-only page transiently,
  which would otherwise end pagination early and silently truncate data.
- Writes each snapshot as data/raw/<key>/<key>_<UTC timestamp>.csv plus a
  JSON manifest, then marks the CSV read-only. Raw files are never modified.
- Skips the download when the latest snapshot is already current
  (same source rowsUpdatedAt and row count), unless force=True.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import stat
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import DATA_RAW, DATASETS, Dataset

log = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 50_000
MAX_COUNT_ATTEMPTS = 3
TIMEOUT_SECONDS = 60


class IngestError(RuntimeError):
    pass


@dataclass
class Manifest:
    dataset_key: str
    socrata_id: str
    source_url: str
    snapshot_file: str
    ingested_at_utc: str
    row_count: int
    sha256: str
    source_rows_updated_at: int | None
    columns: list[str]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without network)
# ---------------------------------------------------------------------------

def merge_csv_pages(pages: list[str]) -> str:
    """Concatenate CSV page bodies, keeping the header from the first page only.

    Every page must carry the same header; a mismatch means the schema changed
    mid-download and the snapshot is rejected.
    """
    if not pages:
        raise IngestError("No pages to merge")
    header, _, _ = pages[0].partition("\n")
    parts = [pages[0] if pages[0].endswith("\n") else pages[0] + "\n"]
    for page in pages[1:]:
        page_header, _, body = page.partition("\n")
        if page_header != header:
            raise IngestError(f"Header changed between pages: {page_header!r} != {header!r}")
        if body:
            parts.append(body if body.endswith("\n") else body + "\n")
    return "".join(parts)


def count_csv_rows(text: str) -> int:
    """Count data rows (excluding header), honouring quoted newlines."""
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    return sum(1 for _ in reader)


def csv_header(text: str) -> list[str]:
    return next(csv.reader(io.StringIO(text)), [])


def snapshot_name(key: str, ts: datetime) -> str:
    # Microseconds keep back-to-back runs (e.g. --force twice) from colliding; names still sort chronologically.
    return f"{key}_{ts.strftime('%Y%m%dT%H%M%S_%fZ')}.csv"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def make_session(app_token: str | None = None, retries: int = 5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "traffic-safety-analytics (portfolio project)"
    if app_token:
        session.headers["X-App-Token"] = app_token
    return session


def fetch_metadata(session: requests.Session, ds: Dataset) -> dict:
    resp = session.get(ds.metadata_url, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def fetch_row_count(session: requests.Session, ds: Dataset) -> int:
    url = ds.resource_url.replace(".csv", ".json")
    resp = session.get(url, params={"$select": "count(*) AS n"}, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return int(resp.json()[0]["n"])


def fetch_pages(session: requests.Session, ds: Dataset, page_size: int = DEFAULT_PAGE_SIZE) -> list[str]:
    pages: list[str] = []
    offset = 0
    while True:
        params = {"$limit": page_size, "$offset": offset, "$order": ":id"}
        resp = session.get(ds.resource_url, params=params, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        text = resp.text
        n = count_csv_rows(text)
        log.info("%s: offset=%d rows=%d", ds.key, offset, n)
        pages.append(text)
        if n < page_size:
            break
        offset += page_size
    return pages


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def dataset_dir(key: str, raw_root: Path = DATA_RAW) -> Path:
    return raw_root / key


def latest_manifest(key: str, raw_root: Path = DATA_RAW) -> Manifest | None:
    manifests = sorted(dataset_dir(key, raw_root).glob(f"{key}_*.manifest.json"))
    if not manifests:
        return None
    return Manifest(**json.loads(manifests[-1].read_text()))


def latest_snapshot(key: str, raw_root: Path = DATA_RAW) -> Path:
    m = latest_manifest(key, raw_root)
    if m is None:
        raise FileNotFoundError(f"No raw snapshot for {key!r}; run ingestion first")
    return dataset_dir(key, raw_root) / m.snapshot_file


def write_snapshot(key: str, ds: Dataset, text: str, metadata: dict,
                   raw_root: Path = DATA_RAW, now: datetime | None = None) -> Manifest:
    now = now or datetime.now(timezone.utc)
    out_dir = dataset_dir(key, raw_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = snapshot_name(key, now)
    path = out_dir / fname
    if path.exists():
        raise IngestError(f"Refusing to overwrite existing raw snapshot {path}")
    data = text.encode("utf-8")
    path.write_bytes(data)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only

    manifest = Manifest(
        dataset_key=key,
        socrata_id=ds.socrata_id,
        source_url=ds.portal_url,
        snapshot_file=fname,
        ingested_at_utc=now.isoformat(timespec="seconds"),
        row_count=count_csv_rows(text),
        sha256=hashlib.sha256(data).hexdigest(),
        source_rows_updated_at=metadata.get("rowsUpdatedAt"),
        columns=csv_header(text),
    )
    (out_dir / fname.replace(".csv", ".manifest.json")).write_text(json.dumps(asdict(manifest), indent=2))
    (out_dir / fname.replace(".csv", ".metadata.json")).write_text(json.dumps(metadata, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ingest_dataset(key: str, session: requests.Session | None = None, force: bool = False,
                   raw_root: Path = DATA_RAW, page_size: int = DEFAULT_PAGE_SIZE) -> tuple[Manifest, bool]:
    """Ingest one dataset. Returns (manifest, downloaded)."""
    ds = DATASETS[key]
    session = session or make_session(os.getenv("SOCRATA_APP_TOKEN") or None)

    metadata = fetch_metadata(session, ds)
    expected = fetch_row_count(session, ds)

    cached = latest_manifest(key, raw_root)
    if (not force and cached is not None
            and cached.source_rows_updated_at == metadata.get("rowsUpdatedAt")
            and cached.row_count == expected):
        log.info("%s: cache is current (%d rows), skipping download", key, cached.row_count)
        return cached, False

    for attempt in range(1, MAX_COUNT_ATTEMPTS + 1):
        text = merge_csv_pages(fetch_pages(session, ds, page_size))
        got = count_csv_rows(text)
        if got == expected:
            break
        log.warning("%s: row count mismatch (got %d, expected %d), attempt %d/%d",
                    key, got, expected, attempt, MAX_COUNT_ATTEMPTS)
    else:
        raise IngestError(f"{key}: row count mismatch after {MAX_COUNT_ATTEMPTS} attempts")

    manifest = write_snapshot(key, ds, text, metadata, raw_root)
    log.info("%s: wrote %s (%d rows)", key, manifest.snapshot_file, manifest.row_count)
    return manifest, True


def ingest_all(force: bool = False, raw_root: Path = DATA_RAW) -> dict[str, Manifest]:
    session = make_session(os.getenv("SOCRATA_APP_TOKEN") or None)
    return {key: ingest_dataset(key, session, force, raw_root)[0] for key in DATASETS}


def load_raw(path: Path) -> pd.DataFrame:
    """Load a raw snapshot with every column as string, so cleaning sees the source as-is."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for k, m in ingest_all().items():
        print(f"{k:14s} {m.row_count:>7,d} rows  {m.snapshot_file}")
