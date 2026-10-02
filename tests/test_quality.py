"""Data-quality checks: null counting, internal consistency, cross-dataset reconciliation, report rendering."""

import pandas as pd

from traffic_safety import clean
from traffic_safety.quality import DatasetSummary, QualityLog, null_counts, render_report


def checks(qlog, prefix=""):
    return {c.name: c for c in qlog.checks if c.name.startswith(prefix)}


def test_null_counts_treat_blank_strings_as_null():
    df = pd.DataFrame({"a": ["x", "", "  ", None], "b": [1, 2, None, 4]})
    assert null_counts(df) == {"a": 3, "b": 1}


def test_pct_retained():
    s = DatasetSummary("t", "src", "snap", rows_ingested=200, nulls={}, null_handling={}, final_rows=150)
    assert s.pct_retained == 75.0
    assert DatasetSummary("t", "s", "s", 0, {}, {}).pct_retained == 0.0


def test_severity_internal_checks_pass_on_consistent_data(cleaned, qlog):
    clean.severity_internal_checks(cleaned["severity"], qlog)
    assert all(c.passed for c in qlog.checks)


def test_severity_internal_checks_catch_broken_totals(cleaned, qlog):
    sev = cleaned["severity"].copy()
    sev.loc[sev.year == 2022, "total_collisions"] += 1
    clean.severity_internal_checks(sev, qlog)
    c = checks(qlog)["severity: PDO + fatal/injury collisions = total_collisions"]
    assert not c.passed and "2022" in c.detail


def test_fatality_breakdown_gap_is_reported_but_allowed(cleaned, qlog):
    sev = cleaned["severity"].copy()
    sev.loc[sev.year == 2022, "fatality_pedestrian"] = 0  # 1 fatality now unattributed
    clean.severity_internal_checks(sev, qlog)
    c = checks(qlog)["severity: fatalities by road user ≤ total_fatalities"]
    assert c.passed and "{2022: 1}" in c.detail


def test_fatality_breakdown_exceeding_total_fails(cleaned, qlog):
    sev = cleaned["severity"].copy()
    sev.loc[sev.year == 2022, "fatality_pedestrian"] = 3
    clean.severity_internal_checks(sev, qlog)
    assert not checks(qlog)["severity: fatalities by road user ≤ total_fatalities"].passed


def test_reconcile_passes_on_matching_fixture(cleaned, qlog):
    clean.reconcile(cleaned["temporal"], cleaned["severity"], qlog)
    assert len(qlog.checks) == 5
    assert all(c.passed for c in qlog.checks)


def test_reconcile_detects_missing_temporal_rows(cleaned, qlog):
    t = cleaned["temporal"]
    t = t[~((t.year == 2023) & (t.severity == "Minor"))]
    clean.reconcile(t, cleaned["severity"], qlog)
    by_name = checks(qlog)
    assert not by_name["temporal total = severity total_collisions (each year)"].passed
    assert not by_name["temporal 'Minor' = severity minor_injury_collisions (each year)"].passed
    assert by_name["temporal 'PDO' = severity pdo_collisions (each year)"].passed


def test_report_contains_real_numbers(raw_temporal, cleaned):
    qlog = QualityLog()
    clean.clean_temporal(raw_temporal, qlog)
    qlog.summaries["temporal"] = DatasetSummary(
        "temporal", "src", "snap.csv", rows_ingested=9, nulls={"collisions": 0}, null_handling={},
        rows_flagged=2, final_rows=9)
    qlog.check("demo check", False, "broken")
    qlog.notes.append("A note.")
    md = render_report(qlog)
    assert "| temporal | 9 | 0 | 0 | 2 | 9 | 100.00% |" in md
    assert "Hour code 24" in md and "| 1 | flagged |" in md
    assert "**FAIL**" in md and "0 of 1 passed" in md
    assert "1. A note." in md


def test_build_notes_uses_computed_numbers(cleaned, qlog):
    clean.build_notes(cleaned["temporal"], cleaned["top_locations"], cleaned["severity"], qlog)
    text = " ".join(qlog.notes)
    assert "Hour code 24 (23:01–24:00) has 1 collisions" in text
    assert "1 row (1 collision) has hour code 0" in text
