"""
Shared fixtures. All tests run offline on the small hand-built CSVs in
tests/fixtures/ (test data shaped like the real Socrata exports, NOT real data).

Fixture design (so KPI expectations can be checked by hand):
- temporal_raw.csv: years 2021–2023, January and February only.
  Monthly totals: 2021 Jan 12 / Feb 5; 2022 Jan 13 / Feb 5; 2023 Jan 14 / Feb 31.
  Includes one hour-0 row (2023 Feb, PDO 1), one hour-24 row (2022 Jan, Fatal 1),
  and one 'Major' row (2021 Jan, 2 collisions at 01:00–02:00, Sunday).
- severity_raw.csv: totals 17 / 18 / 45 reconcile exactly with temporal by year and severity.
- top_locations_raw.csv: 2022–2023, with 2022+ labels, a typo, an 'AND' intersection
  written in reverse order, a tie, and a 'ST.' (Saint) name.
"""

from pathlib import Path

import pandas as pd
import pytest

from traffic_safety import clean
from traffic_safety.ingest import load_raw
from traffic_safety.quality import QualityLog

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def qlog() -> QualityLog:
    return QualityLog()


@pytest.fixture
def raw_temporal() -> pd.DataFrame:
    return load_raw(FIXTURES / "temporal_raw.csv")


@pytest.fixture
def raw_top() -> pd.DataFrame:
    return load_raw(FIXTURES / "top_locations_raw.csv")


@pytest.fixture
def raw_severity() -> pd.DataFrame:
    return load_raw(FIXTURES / "severity_raw.csv")


@pytest.fixture
def cleaned(raw_temporal, raw_top, raw_severity, qlog) -> dict[str, pd.DataFrame]:
    return {
        "temporal": clean.clean_temporal(raw_temporal, qlog),
        "top_locations": clean.clean_top_locations(raw_top, qlog),
        "severity": clean.clean_severity(raw_severity, qlog),
    }


@pytest.fixture
def con(cleaned):
    from traffic_safety.load import load_duckdb

    c = load_duckdb(cleaned, db_path=":memory:")
    yield c
    c.close()


def temporal_row(**overrides) -> dict:
    """One valid raw temporal row (all strings, like the source); override fields to make it dirty."""
    row = {
        "collision_report_year": "2022", "season": "WINTER", "collision_report_month_name": "JANUARY",
        "collision_report_hour": "9", "hour_group_name": "AM Peak", "is_weekend": "WEEKDAY",
        "day_of_week_name": "MONDAY", "collision_classification": "PDO", "collisions": "3",
    }
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries to open a network connection."""
    import socket

    def guard(*args, **kwargs):
        raise RuntimeError("Network access is not allowed in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
