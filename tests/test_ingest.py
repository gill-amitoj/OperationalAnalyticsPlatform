"""Ingestion: CSV page parsing, snapshot immutability, pagination, retries on short reads, cache."""

import json
import os
import stat
from datetime import datetime, timezone

import pytest

from traffic_safety import ingest
from traffic_safety.config import DATASETS
from traffic_safety.ingest import IngestError

HEADER = '"year","location_type","rank","location_description","collision_count"\n'


def page(*rows: str) -> str:
    return HEADER + "".join(r + "\n" for r in rows)


# --- pure parsing -----------------------------------------------------------

def test_merge_csv_pages_keeps_single_header():
    merged = ingest.merge_csv_pages([page('"2019","A","1","X","5"'), page('"2019","B","2","Y","4"')])
    assert merged.count('"year"') == 1
    assert ingest.count_csv_rows(merged) == 2


def test_merge_csv_pages_ignores_empty_trailing_page():
    merged = ingest.merge_csv_pages([page('"2019","A","1","X","5"'), HEADER])
    assert ingest.count_csv_rows(merged) == 1


def test_merge_csv_pages_rejects_schema_change():
    other = '"year","rank"\n"2019","1"\n'
    with pytest.raises(IngestError, match="Header changed"):
        ingest.merge_csv_pages([page('"2019","A","1","X","5"'), other])


def test_merge_csv_pages_requires_pages():
    with pytest.raises(IngestError):
        ingest.merge_csv_pages([])


def test_count_csv_rows_handles_quoted_newlines_and_commas():
    text = page('"2019","A","1","LINE ONE\nLINE TWO, WITH COMMA","5"')
    assert ingest.count_csv_rows(text) == 1
    assert ingest.csv_header(text) == ["year", "location_type", "rank", "location_description", "collision_count"]


def test_snapshot_name_is_utc_timestamped():
    ts = datetime(2026, 10, 2, 2, 5, 29, 123456, tzinfo=timezone.utc)
    assert ingest.snapshot_name("temporal", ts) == "temporal_20261002T020529_123456Z.csv"


# --- snapshot writing --------------------------------------------------------

def test_write_snapshot_is_read_only_with_manifest(tmp_path):
    ds = DATASETS["top_locations"]
    text = page('"2019","A","1","X","5"', '"2019","B","2","Y","4"')
    m = ingest.write_snapshot("top_locations", ds, text, {"rowsUpdatedAt": 123}, raw_root=tmp_path)
    path = tmp_path / "top_locations" / m.snapshot_file
    assert path.read_text() == text
    assert not os.stat(path).st_mode & stat.S_IWUSR
    manifest = json.loads((tmp_path / "top_locations" / m.snapshot_file.replace(".csv", ".manifest.json")).read_text())
    assert manifest["row_count"] == 2
    assert manifest["source_rows_updated_at"] == 123
    assert len(manifest["sha256"]) == 64


def test_write_snapshot_refuses_overwrite(tmp_path):
    ds = DATASETS["top_locations"]
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ingest.write_snapshot("top_locations", ds, page(), {}, raw_root=tmp_path, now=now)
    with pytest.raises(IngestError, match="Refusing to overwrite"):
        ingest.write_snapshot("top_locations", ds, page(), {}, raw_root=tmp_path, now=now)


def test_latest_snapshot_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.latest_snapshot("temporal", raw_root=tmp_path)


def test_load_raw_keeps_values_as_strings(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text(page('"2019","A","01","X",""'))
    df = ingest.load_raw(p)
    assert df.loc[0, "rank"] == "01"          # leading zero preserved: no silent type coercion
    assert df["collision_count"].isna().all()  # blank -> NA


# --- orchestration with a fake HTTP session (no network) --------------------

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = payload if isinstance(payload, str) else ""
        self.encoding = None

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    """Serves metadata, count(*), and CSV pages; `pages` is a list of page lists, one per download attempt."""

    def __init__(self, rows, page_size, rows_updated_at=1, short_first_attempt=False):
        self.rows, self.page_size, self.rows_updated_at = rows, page_size, rows_updated_at
        self.short_first_attempt = short_first_attempt
        self.csv_calls = 0
        self.attempt_offsets = []

    def get(self, url, params=None, timeout=None):
        if url.endswith(".json") and "/api/views/" in url:
            return FakeResponse({"rowsUpdatedAt": self.rows_updated_at})
        if url.endswith(".json"):
            return FakeResponse([{"n": str(len(self.rows))}])
        self.csv_calls += 1
        offset, limit = params["$offset"], params["$limit"]
        if offset == 0:
            self.attempt_offsets.append([])
        self.attempt_offsets[-1].append(offset)
        chunk = self.rows[offset:offset + limit]
        if self.short_first_attempt and len(self.attempt_offsets) == 1 and offset > 0:
            chunk = []  # transient header-only page mid-download
        return FakeResponse(page(*chunk))


ROWS = [f'"2019","Intersection","{i}","LOC {i}","{100 - i}"' for i in range(1, 8)]


def test_ingest_paginates_and_verifies_count(tmp_path):
    s = FakeSession(ROWS, page_size=3)
    m, downloaded = ingest.ingest_dataset("top_locations", session=s, raw_root=tmp_path, page_size=3)
    assert downloaded
    assert m.row_count == 7
    assert s.attempt_offsets == [[0, 3, 6]]


def test_ingest_retries_when_a_page_comes_back_empty(tmp_path):
    s = FakeSession(ROWS, page_size=3, short_first_attempt=True)
    m, _ = ingest.ingest_dataset("top_locations", session=s, raw_root=tmp_path, page_size=3)
    assert m.row_count == 7
    assert len(s.attempt_offsets) == 2


def test_ingest_fails_after_repeated_short_reads(tmp_path, monkeypatch):
    class AlwaysShort(FakeSession):
        def get(self, url, params=None, timeout=None):
            if params and params.get("$offset", 0) > 0:
                return FakeResponse(page())
            return super().get(url, params, timeout)

    with pytest.raises(IngestError, match="row count mismatch"):
        ingest.ingest_dataset("top_locations", session=AlwaysShort(ROWS, 3), raw_root=tmp_path, page_size=3)


def test_ingest_uses_cache_when_source_unchanged(tmp_path):
    ingest.ingest_dataset("top_locations", session=FakeSession(ROWS, 50), raw_root=tmp_path)
    s = FakeSession(ROWS, 50)
    _, downloaded = ingest.ingest_dataset("top_locations", session=s, raw_root=tmp_path)
    assert not downloaded and s.csv_calls == 0


def test_ingest_redownloads_when_source_updated(tmp_path):
    ingest.ingest_dataset("top_locations", session=FakeSession(ROWS, 50, rows_updated_at=1), raw_root=tmp_path)
    s = FakeSession(ROWS, 50, rows_updated_at=2)
    _, downloaded = ingest.ingest_dataset("top_locations", session=s, raw_root=tmp_path)
    assert downloaded and s.csv_calls == 1
