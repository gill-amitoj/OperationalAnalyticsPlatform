"""KPI SQL views on the fixture data (DuckDB in-memory). Expected values are worked out by hand in conftest."""

import pandas as pd
import pytest

from traffic_safety.findings import parse_named_queries, run_findings
from traffic_safety.load import query_view, query_views, render_sql, view_names
from traffic_safety.pipeline import export_tableau


def v(con, name, **filters):
    df = query_view(con, name)
    for k, val in filters.items():
        df = df[df[k] == val]
    return df.reset_index(drop=True)


def test_all_views_created(con):
    names = view_names()
    assert len(names) == 12
    assert set(names) <= {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}


def test_render_sql_qualifies_placeholders():
    sql = "SELECT * FROM {temporal} JOIN {v_kpi_monthly}"
    assert render_sql(sql, lambda n: f"`p.d.{n}`") == "SELECT * FROM `p.d.temporal` JOIN `p.d.v_kpi_monthly`"


def test_annual_summary_yoy(con):
    a = v(con, "v_kpi_annual_summary").set_index("year")
    assert pd.isna(a.loc[2021, "collisions_yoy_pct"])
    assert a.loc[2022, "collisions_yoy_pct"] == pytest.approx(5.9)      # (18-17)/17
    assert a.loc[2023, "collisions_yoy_pct"] == pytest.approx(150.0)    # (45-18)/18
    assert a.loc[2022, "fatal_serious_yoy_pct"] == pytest.approx(-50.0)  # 2 -> 1
    assert a.loc[2023, "fatal_serious_collisions_per_100k"] == 0


def test_monthly_leave_one_out_baseline_and_outlier(con):
    m = v(con, "v_kpi_monthly").set_index(["year", "month"])
    assert len(m) == 6
    assert m.loc[(2023, 2), "collisions"] == 31
    assert m.loc[(2023, 2), "same_month_other_years_avg"] == 5.0          # (5 + 5) / 2
    assert m.loc[(2023, 2), "ratio_to_other_years"] == pytest.approx(6.2)
    assert m.loc[(2021, 1), "same_month_other_years_avg"] == 13.5         # (13 + 14) / 2
    assert m.is_outlier_month.sum() == 1
    assert str(m.loc[(2023, 2), "month_start"])[:10] == "2023-02-01"


def test_day_hour_excludes_invalid_hour(con):
    dh = v(con, "v_kpi_day_hour")
    assert dh.collisions.sum() == 79       # 80 minus the hour-0 collision
    assert dh.hour_of_day.notna().all()
    assert dh[dh.hour_of_day == 23].is_midnight_bucket.all()


def test_hourly_shares_and_severity_rate(con):
    h = v(con, "v_kpi_hourly").set_index("hour_of_day")
    assert h.collisions.sum() == 79
    assert h.loc[8, "pct_of_collisions"] == pytest.approx(round(100 * 36 / 79, 2))
    assert h.loc[1, "fatal_serious_per_1000"] == 1000.0     # the single 'Major' row
    assert h.loc[8, "fatal_serious_per_1000"] == 0.0
    assert h.pct_of_collisions.sum() == pytest.approx(100, abs=0.05)


def test_hour_group_normalizes_by_period_length(con):
    g = v(con, "v_kpi_hour_group").set_index("hour_group")
    assert g.loc["Evening", "hours_in_period"] == 12
    # 3 collisions / 12 h = 0.25; SQL ROUND is half-away-from-zero (DuckDB and BigQuery) -> 0.3
    assert g.loc["Evening", "collisions_per_hour"] == pytest.approx(0.3)
    assert g.loc["PM Peak", "collisions_per_hour"] == pytest.approx(13.3)   # 40 / 3


def test_day_of_week_average_uses_all_years(con):
    d = v(con, "v_kpi_day_of_week").set_index("day_of_week")
    assert d.loc["MONDAY", "collisions"] == 37
    assert d.loc["MONDAY", "avg_collisions_per_year"] == pytest.approx(12.3)
    assert d.loc["SUNDAY", "avg_collisions_per_year"] == pytest.approx(0.7)  # 2 over 3 years, not 2 over 1


def test_severity_trend_shares_sum_to_100(con):
    s = v(con, "v_kpi_severity_trend")
    assert s.groupby("year").pct_of_year.sum().round(1).eq(100.0).all()
    assert set(s.severity) == {"PDO", "Minor", "Serious", "Fatal"}


def test_location_persistence(con):
    p = v(con, "v_kpi_location_persistence").set_index("location_key")
    top = p.loc["107 AVENUE NW & 142 STREET NW"]
    assert (top.years_listed, top.collisions_in_listed_years, top.avg_collisions_per_listed_year) == (2, 244, 122.0)
    assert top.best_rank == 1 and top.years_in_top_10 == 2
    assert p.loc["23 AVENUE NW & 91 STREET NW", "years_listed"] == 2  # matched despite 'AND' + reversed order
    assert p.loc["QUESNELL BRIDGE", "location_group"] == "Midblock"


def test_vulnerable_road_users_long_format(con):
    r = v(con, "v_kpi_vulnerable_road_users")
    assert len(r) == 9
    ped = r[(r.road_user == "Pedestrian")].set_index("year")
    assert ped.loc[2021, "serious_injuries"] == 1 and ped.loc[2022, "fatalities"] == 1


def test_intersection_midblock_share(con):
    i = v(con, "v_kpi_intersection_midblock").set_index("year")
    assert i.loc[2021, "pct_injuries_at_intersections"] == 75.0
    assert (i.fatalities_unattributed == 0).all()


def test_drilldown_baseline(con):
    d = v(con, "v_kpi_month_hour_drilldown", year=2023, month=2).set_index("hour_of_day")
    assert d.loc[16, "collisions"] == 30
    assert d.loc[16, "same_month_other_years_avg"] == 5.0


def test_views_return_integers_not_floats(con):
    hg = v(con, "v_kpi_hour_group")
    assert str(hg.collisions.dtype) == "Int64"


def test_findings_queries_parse_and_run(con):
    named = parse_named_queries("-- name: a\nSELECT 1;\n\n-- name: b\nSELECT 2;")
    assert named == {"a": "SELECT 1;", "b": "SELECT 2;"}
    results = run_findings(con)
    assert "f1_night_severity" in results
    night = results["f1_night_severity"][1].iloc[0]
    assert night.night_collisions == 2 and night.night_per_1000 == 1000.0


def test_tableau_export_one_sorted_csv_per_view(con, tmp_path):
    paths = export_tableau(query_views(con), tmp_path)
    assert len(paths) == 12
    monthly = pd.read_csv(tmp_path / "kpi_monthly.csv")
    assert list(monthly[["year", "month"]].itertuples(index=False, name=None)) == sorted(
        monthly[["year", "month"]].itertuples(index=False, name=None))
