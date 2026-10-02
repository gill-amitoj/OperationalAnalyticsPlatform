"""Every cleaning rule, tested in isolation on small frames."""

import pandas as pd
import pytest

from conftest import temporal_row
from traffic_safety import clean


def rule(qlog, name_part):
    hits = [r for r in qlog.rules if name_part in r.rule]
    assert hits, f"no rule matching {name_part!r}"
    return hits[0]


def temporal(*rows):
    return pd.DataFrame(list(rows))


# --- generic helpers ---------------------------------------------------------

def test_standardize_columns():
    df = pd.DataFrame(columns=[" Collision Report Year ", "Day-Of-Week", "rank"])
    assert list(clean.standardize_columns(df).columns) == ["collision_report_year", "day_of_week", "rank"]


def test_strip_text_trims_collapses_and_blanks_to_na():
    df = pd.DataFrame({"a": ["  YELLOWHEAD   TRAIL ", "   ", None]})
    out = clean.strip_text(df)
    assert out.loc[0, "a"] == "YELLOWHEAD TRAIL"
    assert out["a"].isna().sum() == 2


# --- temporal ------------------------------------------------------------------

def test_exact_duplicates_removed(qlog):
    out = clean.clean_temporal(temporal(temporal_row(), temporal_row(), temporal_row(collisions="4")), qlog)
    assert len(out) == 2
    assert rule(qlog, "Exact duplicate").rows_affected == 1


def test_whitespace_and_case_variants_count_as_duplicates(qlog):
    out = clean.clean_temporal(temporal(temporal_row(), temporal_row(day_of_week_name=" monday ")), qlog)
    assert len(out) == 1


@pytest.mark.parametrize("bad", ["", "abc", "2.5", None])
def test_unparseable_collisions_dropped(qlog, bad):
    out = clean.clean_temporal(temporal(temporal_row(), temporal_row(collisions=bad, collision_report_hour="10",
                                                                     hour_group_name="AM Non-Peak")), qlog)
    assert len(out) == 1
    assert rule(qlog, "Missing or non-integer").rows_affected == 1


@pytest.mark.parametrize("bad", ["0", "-3"])
def test_non_positive_counts_dropped(qlog, bad):
    out = clean.clean_temporal(temporal(temporal_row(collisions=bad)), qlog)
    assert out.empty
    assert rule(qlog, "Non-positive collision count").rows_affected == 1


def test_future_year_dropped(qlog):
    future = str(clean.current_year() + 1)
    out = clean.clean_temporal(temporal(temporal_row(), temporal_row(collision_report_year=future)), qlog)
    assert out.year.tolist() == [2022]
    assert rule(qlog, "Future or implausible year").rows_affected == 1


def test_unknown_month_dropped(qlog):
    out = clean.clean_temporal(temporal(temporal_row(collision_report_month_name="SMARCH")), qlog)
    assert out.empty


def test_hour_outside_range_dropped(qlog):
    out = clean.clean_temporal(temporal(temporal_row(collision_report_hour="25")), qlog)
    assert out.empty
    assert rule(qlog, "Hour code outside").rows_affected == 1


def test_hour_code_mapping_follows_source_definition(qlog):
    # Source: hour code h covers (h-1):01 to h:00, so code 9 -> 08:00–09:00 and code 1 -> 00:00–01:00.
    out = clean.clean_temporal(temporal(
        temporal_row(),
        temporal_row(collision_report_hour="1", hour_group_name="Evening")), qlog)
    got = dict(zip(out.hour_code, out.hour_of_day))
    assert got == {1: 0, 9: 8}
    assert out.set_index("hour_code").loc[9, "hour_label"] == "08:00–09:00"


def test_hour_zero_kept_but_flagged_and_unassigned(qlog):
    out = clean.clean_temporal(temporal(temporal_row(collision_report_hour="0", hour_group_name="Invalid")), qlog)
    assert len(out) == 1
    assert pd.isna(out.loc[0, "hour_of_day"])
    assert out.loc[0, "hour_label"] == "Unknown"
    assert "invalid_hour" in out.loc[0, "dq_flags"]


def test_hour_24_mapped_to_23_and_flagged(qlog):
    out = clean.clean_temporal(temporal(temporal_row(collision_report_hour="24", hour_group_name="Evening")), qlog)
    assert out.loc[0, "hour_of_day"] == 23
    assert "midnight_bucket" in out.loc[0, "dq_flags"]


def test_hour_group_mismatch_flagged(qlog):
    out = clean.clean_temporal(temporal(temporal_row(hour_group_name="PM Peak")), qlog)
    assert "hour_group_mismatch" in out.loc[0, "dq_flags"]


@pytest.mark.parametrize("month,season,ok", [("MARCH", "WINTER", True), ("MARCH", "SPRING", True),
                                             ("MARCH", "SUMMER", False), ("JULY", "WINTER", False)])
def test_season_consistency(qlog, month, season, ok):
    out = clean.clean_temporal(temporal(temporal_row(collision_report_month_name=month, season=season)), qlog)
    assert ("season_mismatch" in out.loc[0, "dq_flags"]) is not ok


def test_day_type_mismatch_flagged(qlog):
    out = clean.clean_temporal(temporal(temporal_row(day_of_week_name="SUNDAY")), qlog)  # labelled WEEKDAY
    assert "day_type_mismatch" in out.loc[0, "dq_flags"]


def test_unknown_day_flagged(qlog):
    out = clean.clean_temporal(temporal(temporal_row(day_of_week_name="FUNDAY")), qlog)
    assert "unknown_day" in out.loc[0, "dq_flags"]


def test_duplicate_dimension_key_flagged_not_merged(qlog):
    out = clean.clean_temporal(temporal(temporal_row(collisions="3"), temporal_row(collisions="5")), qlog)
    assert len(out) == 2
    assert out.dq_flags.str.contains("duplicate_key").all()


def test_severity_harmonized(qlog):
    rows = [temporal_row(collision_classification=c, collision_report_hour=str(h), hour_group_name=g)
            for c, h, g in [("Major", 9, "AM Peak"), ("Serious", 8, "AM Peak"), ("PDO", 10, "AM Non-Peak"),
                            ("Minor", 11, "AM Non-Peak"), ("Fatal", 12, "AM Non-Peak")]]
    out = clean.clean_temporal(temporal(*rows), qlog).set_index("severity_source")
    assert out.loc["MAJOR", "severity"] == "Serious"
    assert out.loc["SERIOUS", "severity"] == "Serious"
    assert out.loc["FATAL", "is_fatal_or_serious"] and not out.loc["MINOR", "is_fatal_or_serious"]
    assert rule(qlog, "'MAJOR' harmonized").rows_affected == 1


def test_unknown_severity_flagged(qlog):
    out = clean.clean_temporal(temporal(temporal_row(collision_classification="Catastrophic")), qlog)
    assert "unknown_severity" in out.loc[0, "dq_flags"]


@pytest.mark.parametrize("year,period", [(2021, "Pre-CRC"), (2022, "Transition"), (2023, "CRC")])
def test_reporting_period(qlog, year, period):
    out = clean.clean_temporal(temporal(temporal_row(collision_report_year=str(year))), qlog)
    assert out.loc[0, "reporting_period"].startswith(period)


def test_clean_fixture_keeps_every_row(raw_temporal, qlog):
    out = clean.clean_temporal(raw_temporal, qlog)
    assert len(out) == len(raw_temporal) == 9
    assert out.collisions.sum() == 80
    assert out.dq_flagged.sum() == 2  # hour 0 and hour 24


# --- top locations -------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("YELLOWHEAD TRAIL NW BETWEEN 107-121 STREEET NW", "YELLOWHEAD TRAIL NW BETWEEN 107-121 STREET NW"),
    ("GATEWAY BOULEVARD NE BETWEN 34-39A AVENUE NW", "GATEWAY BOULEVARD NE BETWEEN 34-39A AVENUE NW"),
    ("ANTHONY HENDA DRIVE NW", "ANTHONY HENDAY DRIVE NW"),
    ("GROAT ROAD NW BETWEEN STONY PLAIN RD-107 AVENUE NW", "GROAT ROAD NW BETWEEN STONY PLAIN ROAD-107 AVENUE NW"),
    ("119 ST-RABBIT HILL ROAD NW", "119 STREET-RABBIT HILL ROAD NW"),
    ("ST. ALBERT TRAIL NW BETWEEN 134-137 AVENUE NW", "ST. ALBERT TRAIL NW BETWEEN 134-137 AVENUE NW"),
    ("ANTHONY HENDAY DRIVE NW WESTBOUND, EAST AND WEST OF 111 STREET NW", "ANTHONY HENDAY DRIVE NW WESTBOUND EAST AND WEST OF 111 STREET NW"),
])
def test_normalize_location_name_midblock(raw, expected):
    assert clean.normalize_location_name(raw, is_intersection=False) == expected


def test_and_becomes_ampersand_only_for_intersections():
    assert clean.normalize_location_name("YELLOWHEAD TRAIL NW AND FORT ROAD NW", True) == "YELLOWHEAD TRAIL NW & FORT ROAD NW"
    assert " AND " in clean.normalize_location_name("BETWEEN ON AND OFF RAMPS", False)


def test_location_key_is_order_independent():
    a = clean.location_key("YELLOWHEAD TRAIL NW & FORT ROAD NW", True)
    b = clean.location_key("FORT ROAD NW & YELLOWHEAD TRAIL NW", True)
    assert a == b


def test_top_locations_groups_and_keys(raw_top, qlog):
    out = clean.clean_top_locations(raw_top, qlog)
    assert set(out.location_group) == {"Intersection", "Midblock"}
    keys_2023 = out[(out.year == 2023) & (out.location_group == "Intersection")].location_key.tolist()
    keys_2022 = out[(out.year == 2022) & (out.location_group == "Intersection")].location_key.tolist()
    assert "23 AVENUE NW & 91 STREET NW" in keys_2022 and "23 AVENUE NW & 91 STREET NW" in keys_2023
    assert rule(qlog, "grouped as 'Midblock'").rows_affected == 2  # MID AVENUE + MID STREET


def test_rank_recomputed_with_ties(raw_top, qlog):
    out = clean.clean_top_locations(raw_top, qlog)
    r = out[(out.year == 2023) & (out.location_group == "Intersection")].set_index("location_name")["rank"]
    assert r["107 AVENUE NW & 142 STREET NW"] == 1
    assert r["91 STREET NW & 23 AVENUE NW"] == 2 and r["ST. ALBERT TRAIL NW & 137 AVENUE NW"] == 2
    assert out.rank_source.tolist() != out["rank"].tolist()  # source tie ranked 3; recomputed 2


def test_unknown_location_type_flagged(qlog):
    raw = pd.DataFrame([{"year": "2023", "location_type": "ROUNDABOUT", "rank": "1",
                         "location_description": "X", "collision_count": "5"}])
    out = clean.clean_top_locations(raw, qlog)
    assert "unknown_location_type" in out.loc[0, "dq_flags"]
    assert out.loc[0, "location_group"] == "Unknown" and out.loc[0, "rank"] == 1


def test_duplicate_location_in_year_flagged(qlog):
    rows = [{"year": "2023", "location_type": "INTERSECTION", "rank": "1",
             "location_description": d, "collision_count": "5"}
            for d in ["A STREET NW & B AVENUE NW", "B AVENUE NW AND A STREET NW"]]
    out = clean.clean_top_locations(pd.DataFrame(rows), qlog)
    assert out.dq_flags.str.contains("duplicate_location").all()


@pytest.mark.parametrize("field,value", [("collision_count", "0"), ("rank", "-1"), ("location_description", "")])
def test_top_locations_invalid_rows_dropped(qlog, field, value):
    row = {"year": "2023", "location_type": "INTERSECTION", "rank": "1",
           "location_description": "X", "collision_count": "5"}
    row[field] = value
    assert clean.clean_top_locations(pd.DataFrame([row]), qlog).empty


# --- severity ------------------------------------------------------------------

def test_severity_rates_recomputed(raw_severity, qlog):
    out = clean.clean_severity(raw_severity, qlog).set_index("year")
    assert out.loc[2023, "collisions_per_100k"] == pytest.approx(45 / 1_100_000 * 100_000, abs=0.05)
    assert out.loc[2021, "fatal_serious_collisions"] == 2
    assert out.loc[2021, "ksi_persons"] == 2


def test_severity_renames_ambiguous_columns(raw_severity, qlog):
    out = clean.clean_severity(raw_severity, qlog)
    assert "fatalities_intersection_pct" in out and "fatalities_intersection_1" not in out
    assert "pdo_collisions" in out


def test_severity_negative_values_dropped(raw_severity, qlog):
    raw = raw_severity.copy()
    raw.loc[0, "total_collisions"] = "-5"
    out = clean.clean_severity(raw, qlog)
    assert 2021 not in out.year.tolist()
