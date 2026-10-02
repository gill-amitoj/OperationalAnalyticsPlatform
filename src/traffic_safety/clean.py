"""
clean.py
========
Cleans and validates the three Annual Collision Report tables.

Policy
------
- Drop a row only when it cannot be used at all (unparseable/non-positive
  counts, impossible or future years, unknown month). Everything else is kept
  and *flagged* in `dq_flags`, so totals still reconcile with the City's
  published figures.
- Every rule records how many rows it touched in a QualityLog, which feeds
  reports/data_quality.md.
- Fixes to text values are explicit lookup tables (no fuzzy matching), so each
  correction can be reviewed.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DATA_PROCESSED, DATASETS, REPORTS_DIR
from .ingest import latest_manifest, latest_snapshot, load_raw
from .quality import DatasetSummary, QualityLog, is_text, null_counts, write_report

log = logging.getLogger(__name__)

MONTHS = {m: i for i, m in enumerate(
    ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
     "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"], start=1)}

DAYS = {d: i for i, d in enumerate(
    ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"], start=1)}

# The source splits March/June/September/December between two seasons
# (astronomical seasons change around the 20th), so those months allow two.
SEASONS_BY_MONTH = {
    1: {"WINTER"}, 2: {"WINTER"}, 3: {"WINTER", "SPRING"}, 4: {"SPRING"}, 5: {"SPRING"},
    6: {"SPRING", "SUMMER"}, 7: {"SUMMER"}, 8: {"SUMMER"}, 9: {"SUMMER", "FALL"},
    10: {"FALL"}, 11: {"FALL"}, 12: {"FALL", "WINTER"},
}


def expected_hour_group(hour_code: int) -> str:
    """Hour group per the dataset's field definition (hour_code h = (h-1):01 to h:00)."""
    if 7 <= hour_code <= 9:
        return "AM Peak"
    if 10 <= hour_code <= 12:
        return "AM Non-Peak"
    if 13 <= hour_code <= 15:
        return "PM Non-Peak"
    if 16 <= hour_code <= 18:
        return "PM Peak"
    if 1 <= hour_code <= 6 or 19 <= hour_code <= 24:
        return "Evening"
    return "Invalid"


# Severity labels. 2019 uses "Serious"; 2020+ uses "Major". Evidence that they are
# the same category: the Temporal "Major" totals equal the Severity table's
# serious_injury_collisions for every year 2020-2023 (checked in reconcile()).
SEVERITY_MAP = {"PDO": "PDO", "MINOR": "Minor", "MAJOR": "Serious", "SERIOUS": "Serious", "FATAL": "Fatal"}
SEVERITY_RANK = {"PDO": 1, "Minor": 2, "Serious": 3, "Fatal": 4}

# Collision Reporting Centres replaced EPS reports in September 2022 (dataset description).
CRC_LAUNCH_YEAR = 2022

LOCATION_GROUP = {
    "INTERSECTION": "Intersection",
    "MIDBLOCK": "Midblock",
    "MID AVENUE": "Midblock",
    "MID STREET": "Midblock",
    # Ranked inside the 2022 midblock ranking (tied at rank 4 with midblock sites).
    "SOUTH OF INTERSECTION": "Midblock",
}

# Unambiguous spelling errors observed in location_description.
LOCATION_TYPO_FIXES = {
    r"\bSTREEET\b": "STREET",
    r"\bBETWEN\b": "BETWEEN",
    r"\bWESBOUND\b": "WESTBOUND",
    r"\bTERWILELGAR\b": "TERWILLEGAR",
    r"\bANTONY HENDAY\b": "ANTHONY HENDAY",
    r"\bANTHONY HENDA\b": "ANTHONY HENDAY",
}
# Abbreviations. "ST." (with a period) is "Saint" (St. Albert Trail) and is left alone.
LOCATION_ABBREVIATIONS = {
    r"\bST\b(?!\.)": "STREET",
    r"\bRD\b": "ROAD",
    r"\bAVE\b": "AVENUE",
    r"\bDR\b": "DRIVE",
    r"\bBLVD\b": "BOULEVARD",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """lower_snake_case column names."""
    out = df.copy()
    out.columns = [re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    return out


def strip_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and collapse internal whitespace in every text column; blank -> NA."""
    out = df.copy()
    for c in out.columns:
        if is_text(out[c]):
            s = out[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            out[c] = s.mask(s == "")
    return out


def remove_exact_duplicates(df: pd.DataFrame, dataset: str, qlog: QualityLog) -> pd.DataFrame:
    n = int(df.duplicated().sum())
    qlog.rule(dataset, "Exact duplicate rows", n, "dropped", "identical across every column")
    return df.drop_duplicates().reset_index(drop=True)


def to_int(df: pd.DataFrame, cols: list[str], dataset: str, qlog: QualityLog) -> pd.DataFrame:
    """Parse integer columns; rows with missing/unparseable values are dropped."""
    out = df.copy()
    bad = pd.Series(False, index=out.index)
    for c in cols:
        parsed = pd.to_numeric(out[c], errors="coerce")
        non_int = parsed.notna() & (parsed != parsed.round())
        bad |= parsed.isna() | non_int
        out[c] = parsed
    qlog.rule(dataset, f"Missing or non-integer value in {', '.join(cols)}", int(bad.sum()), "dropped")
    out = out[~bad].copy()
    for c in cols:
        out[c] = out[c].astype("int64")
    return out.reset_index(drop=True)


def drop_where(df: pd.DataFrame, mask: pd.Series, dataset: str, rule: str, qlog: QualityLog,
               detail: str = "") -> pd.DataFrame:
    qlog.rule(dataset, rule, int(mask.sum()), "dropped", detail)
    return df[~mask].reset_index(drop=True)


def add_flag(df: pd.DataFrame, mask: pd.Series, flag: str, dataset: str, rule: str,
             qlog: QualityLog, detail: str = "") -> pd.DataFrame:
    out = df.copy()
    if "dq_flags" not in out:
        out["dq_flags"] = ""
    out.loc[mask, "dq_flags"] = (out.loc[mask, "dq_flags"] + ";" + flag).str.lstrip(";")
    qlog.rule(dataset, rule, int(mask.sum()), "flagged", detail)
    return out


def current_year() -> int:
    return datetime.now(timezone.utc).year


def reporting_period(year: pd.Series) -> pd.Series:
    return pd.Series(
        pd.cut(year, [-1, CRC_LAUNCH_YEAR - 1, CRC_LAUNCH_YEAR, 10_000],
               labels=["Pre-CRC (police reports)", "Transition (CRC from Sep 2022)", "CRC"]).astype(str),
        index=year.index)


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

TEMPORAL_RENAME = {
    "collision_report_year": "year",
    "collision_report_month_name": "month_name",
    "collision_report_hour": "hour_code",
    "hour_group_name": "hour_group",
    "is_weekend": "day_type",
    "day_of_week_name": "day_of_week",
    "collision_classification": "severity_source",
}
TEMPORAL_DIMS = ["year", "season", "month_name", "hour_code", "hour_group", "day_type", "day_of_week", "severity_source"]


def clean_temporal(raw: pd.DataFrame, qlog: QualityLog, ds: str = "temporal") -> pd.DataFrame:
    df = standardize_columns(raw).rename(columns=TEMPORAL_RENAME)
    df = strip_text(df)
    for c in ["season", "month_name", "day_type", "day_of_week", "severity_source"]:
        df[c] = df[c].str.upper()
    df = remove_exact_duplicates(df, ds, qlog)
    df = to_int(df, ["year", "hour_code", "collisions"], ds, qlog)

    df = drop_where(df, df["collisions"] <= 0, ds, "Non-positive collision count", qlog,
                    "a count row must represent at least one collision")
    df = drop_where(df, (df["year"] > current_year()) | (df["year"] < 1990), ds,
                    "Future or implausible year", qlog, f"outside 1990–{current_year()}")
    df = drop_where(df, ~df["month_name"].isin(MONTHS), ds, "Unknown month name", qlog)
    df = drop_where(df, ~df["hour_code"].between(0, 24), ds, "Hour code outside 0–24", qlog)

    df["month"] = df["month_name"].map(MONTHS).astype("int64")
    df["day_of_week_num"] = df["day_of_week"].map(DAYS).astype("Int64")
    df["hour_of_day"] = (df["hour_code"] - 1).where(df["hour_code"].between(1, 24)).astype("Int64")
    df["hour_label"] = [f"{int(h):02d}:00–{int(h) + 1:02d}:00" if pd.notna(h) else "Unknown"
                        for h in df["hour_of_day"]]
    df["severity"] = df["severity_source"].map(SEVERITY_MAP)
    df["severity_rank"] = df["severity"].map(SEVERITY_RANK).astype("Int64")
    df["is_fatal_or_serious"] = df["severity"].isin(["Fatal", "Serious"])
    df["is_weekend"] = df["day_of_week_num"] >= 6
    df["reporting_period"] = reporting_period(df["year"])
    df["dq_flags"] = ""

    df = add_flag(df, df["hour_code"] == 0, "invalid_hour", ds, "Hour code 0 (source hour group = 'Invalid')", qlog,
                  "kept in totals; excluded from hour-of-day KPIs")
    df = add_flag(df, df["hour_code"] == 24, "midnight_bucket", ds,
                  "Hour code 24 (23:01–24:00), possible default-midnight times", qlog,
                  "kept as hour 23:00–24:00; caveated in hour KPIs")
    df = add_flag(df, df["day_of_week_num"].isna(), "unknown_day", ds, "Unknown day-of-week name", qlog)
    df = add_flag(df, df["severity"].isna(), "unknown_severity", ds, "Unknown severity label", qlog)
    exp_group = df["hour_code"].map(expected_hour_group)
    df = add_flag(df, exp_group != df["hour_group"], "hour_group_mismatch", ds,
                  "Hour group inconsistent with hour code", qlog, "per field definition")
    season_ok = [s in SEASONS_BY_MONTH[m] for s, m in zip(df["season"], df["month"])]
    df = add_flag(df, ~pd.Series(season_ok, index=df.index), "season_mismatch", ds,
                  "Season inconsistent with month", qlog)
    exp_day_type = df["is_weekend"].map({True: "WEEKEND", False: "WEEKDAY"})
    df = add_flag(df, exp_day_type != df["day_type"], "day_type_mismatch", ds,
                  "Weekday/weekend label inconsistent with day name", qlog)
    dup_key = df.duplicated(TEMPORAL_DIMS, keep=False)
    df = add_flag(df, dup_key, "duplicate_key", ds, "Same dimension combination appears more than once", qlog,
                  "would double count; investigate if non-zero")
    qlog.rule(ds, "Severity label 'MAJOR' harmonized to 'Serious'",
              int((df["severity_source"] == "MAJOR").sum()), "fixed", "label used 2020+; 2019 uses 'SERIOUS'")

    df["dq_flagged"] = df["dq_flags"] != ""
    cols = ["year", "month", "month_name", "season", "day_of_week_num", "day_of_week", "day_type", "is_weekend",
            "hour_code", "hour_of_day", "hour_label", "hour_group", "severity", "severity_source", "severity_rank",
            "is_fatal_or_serious", "reporting_period", "collisions", "dq_flags", "dq_flagged"]
    return df[cols].sort_values(["year", "month", "day_of_week_num", "hour_code", "severity_rank"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top locations
# ---------------------------------------------------------------------------

def normalize_location_name(name: str, is_intersection: bool) -> str:
    s = re.sub(r"\s+", " ", str(name).strip().upper())
    for pat, rep in LOCATION_TYPO_FIXES.items():
        s = re.sub(pat, rep, s)
    for pat, rep in LOCATION_ABBREVIATIONS.items():
        s = re.sub(pat, rep, s)
    s = s.replace(",", "")
    if is_intersection:
        s = re.sub(r"\s+AND\s+", " & ", s)
    return re.sub(r"\s+", " ", s).strip()


def location_key(normalized: str, is_intersection: bool) -> str:
    """Order-independent key so 'A & B' and 'B & A' match."""
    if is_intersection and " & " in normalized:
        return " & ".join(sorted(p.strip() for p in normalized.split(" & ")))
    return normalized


def clean_top_locations(raw: pd.DataFrame, qlog: QualityLog, ds: str = "top_locations") -> pd.DataFrame:
    df = standardize_columns(raw).rename(columns={"rank": "rank_source", "location_type": "location_type_source",
                                                  "location_description": "location_description_source"})
    df = strip_text(df)
    df["location_type_source"] = df["location_type_source"].str.upper()
    df = remove_exact_duplicates(df, ds, qlog)
    df = to_int(df, ["year", "rank_source", "collision_count"], ds, qlog)
    df = drop_where(df, df["collision_count"] <= 0, ds, "Non-positive collision count", qlog)
    df = drop_where(df, df["rank_source"] <= 0, ds, "Non-positive rank", qlog)
    df = drop_where(df, (df["year"] > current_year()) | (df["year"] < 1990), ds, "Future or implausible year", qlog)
    df = drop_where(df, df["location_description_source"].isna(), ds, "Missing location description", qlog)

    df["location_group"] = df["location_type_source"].map(LOCATION_GROUP)
    df["dq_flags"] = ""
    unknown = df["location_group"].isna()
    df = add_flag(df, unknown, "unknown_location_type", ds, "Unrecognized location type", qlog,
                  "grouped as 'Unknown' and ranked separately")
    df.loc[unknown, "location_group"] = "Unknown"
    relabelled = df["location_type_source"].isin(["MID AVENUE", "MID STREET", "SOUTH OF INTERSECTION"])
    qlog.rule(ds, "Location type 'MID AVENUE' / 'MID STREET' / 'SOUTH OF INTERSECTION' grouped as 'Midblock'",
              int(relabelled.sum()), "fixed", "2022+ labels; these share one ranking in the source")

    is_int = df["location_group"] == "Intersection"
    df["location_name"] = [normalize_location_name(n, i) for n, i in zip(df["location_description_source"], is_int)]
    upper_ws = df["location_description_source"].str.upper().str.replace(r"\s+", " ", regex=True)
    qlog.rule(ds, "Location name normalized (typos, abbreviations, 'AND'→'&', commas)",
              int((df["location_name"] != upper_ws).sum()), "fixed", "explicit lookup tables in clean.py")
    df["location_key"] = [location_key(n, i) for n, i in zip(df["location_name"], is_int)]

    df["rank"] = (df.groupby(["year", "location_group"])["collision_count"]
                    .rank(method="min", ascending=False).astype("int64"))
    qlog.rule(ds, "Rank recomputed from collision_count (standard competition ranking within year × group)",
              int((df["rank"] != df["rank_source"]).sum()), "fixed",
              "source ranking method differs by year; original kept as rank_source")
    dup = df.duplicated(["year", "location_group", "location_key"], keep=False)
    df = add_flag(df, dup, "duplicate_location", ds, "Same location listed twice in one year", qlog)
    df["reporting_period"] = reporting_period(df["year"])
    df["dq_flagged"] = df["dq_flags"] != ""
    cols = ["year", "location_group", "location_type_source", "rank", "rank_source", "location_name", "location_key",
            "location_description_source", "collision_count", "reporting_period", "dq_flags", "dq_flagged"]
    return df[cols].sort_values(["year", "location_group", "rank", "location_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Severity (one row per year)
# ---------------------------------------------------------------------------

SEVERITY_RENAME = {
    "fatalities_intersection_1": "fatalities_intersection_pct",
    "fatalities_midblock_percent": "fatalities_midblock_pct",
    "injuries_intersection_percent": "injuries_intersection_pct",
    "injuries_midblock_percent": "injuries_midblock_pct",
    "injuries_fatalities_per_1": "injuries_fatalities_per_1_000",
    "fatalities_per_1_000": "fatalities_per_1_000",
    "property_damage_only_pdo": "pdo_collisions",
    "total_serious_minor_injury": "total_serious_minor_injury_collisions",
}


def clean_severity(raw: pd.DataFrame, qlog: QualityLog, ds: str = "severity") -> pd.DataFrame:
    df = standardize_columns(raw).rename(columns=SEVERITY_RENAME)
    df = strip_text(df)
    df = remove_exact_duplicates(df, ds, qlog)
    df = to_int(df, ["year"], ds, qlog)
    df = drop_where(df, (df["year"] > current_year()) | (df["year"] < 1990), ds, "Future or implausible year", qlog)
    num_cols = [c for c in df.columns if c != "year"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    neg = (df[num_cols] < 0).any(axis=1)
    df = drop_where(df, neg, ds, "Negative count or rate", qlog)
    df["dq_flags"] = ""
    df = add_flag(df, df.duplicated("year", keep=False), "duplicate_year", ds, "Year appears more than once", qlog)

    # Recompute rates at full precision; the source rates are rounded to 2 decimals
    # (e.g. fatalities_per_1_000 = 0.01–0.04), which is too coarse to compare years.
    per100k = 100_000 / df["population"]
    df["collisions_per_100k"] = (df["total_collisions"] * per100k).round(1)
    df["fatal_serious_collisions"] = df["fatal_collisions"] + df["serious_injury_collisions"]
    df["fatal_serious_collisions_per_100k"] = (df["fatal_serious_collisions"] * per100k).round(2)
    df["ksi_persons"] = df["total_fatalities"] + df["total_serious_injuries"]
    df["ksi_per_100k"] = (df["ksi_persons"] * per100k).round(2)
    df["fatalities_per_100k"] = (df["total_fatalities"] * per100k).round(2)
    qlog.rule(ds, "Rates recomputed per 100,000 population at full precision", len(df), "fixed",
              "source rates are rounded to 2 dp per 1,000")
    df["reporting_period"] = reporting_period(df["year"])
    df["dq_flagged"] = df["dq_flags"] != ""
    return df.sort_values("year").reset_index(drop=True)


def severity_internal_checks(sev: pd.DataFrame, qlog: QualityLog) -> None:
    def chk(name, diff, tol=0):
        bad = sev.loc[diff.abs() > tol, "year"].tolist()
        qlog.check(f"severity: {name}", not bad, "all years match" if not bad else f"mismatch in {bad}")

    chk("PDO + fatal/injury collisions = total_collisions",
        sev.pdo_collisions + sev.total_fatal_injury_collisions - sev.total_collisions)
    chk("fatal + serious + minor collisions = total fatal/injury collisions",
        sev.fatal_collisions + sev.serious_injury_collisions + sev.minor_injury_collisions - sev.total_fatal_injury_collisions)

    def chk_parts(name, parts, what):
        diff = sev.total_fatalities - parts
        gaps = {int(y): int(d) for y, d in zip(sev.year, diff) if d != 0}
        qlog.check(f"severity: {name} ≤ total_fatalities", bool((diff >= 0).all()),
                   f"unattributed fatalities {gaps} ({what})" if gaps else "exact match")

    chk_parts("fatalities by road user", sev[["fatality_bicyclist", "fatality_motorcyclist", "fatality_pedestrian",
                                              "fatality_vehicle_driver", "fatality_vehicle_passenger"]].sum(axis=1),
              "no road-user category recorded")
    chk_parts("intersection + midblock fatalities", sev.fatalities_intersection + sev.fatalities_midblock,
              "no location type recorded")


# ---------------------------------------------------------------------------
# Cross-dataset reconciliation
# ---------------------------------------------------------------------------

def reconcile(temporal: pd.DataFrame, sev: pd.DataFrame, qlog: QualityLog) -> None:
    by_year = temporal.groupby("year")["collisions"].sum()
    s = sev.set_index("year")
    years = sorted(set(by_year.index) & set(s.index))
    diff = {y: int(by_year[y] - s.loc[y, "total_collisions"]) for y in years}
    qlog.check("temporal total = severity total_collisions (each year)", all(v == 0 for v in diff.values()),
               f"years {years[0]}–{years[-1]}; differences {diff}")
    pivot = temporal.pivot_table(index="year", columns="severity", values="collisions", aggfunc="sum").fillna(0)
    mapping = {"Fatal": "fatal_collisions", "Serious": "serious_injury_collisions",
               "Minor": "minor_injury_collisions", "PDO": "pdo_collisions"}
    for sev_label, col in mapping.items():
        d = {y: int(pivot.loc[y, sev_label] - s.loc[y, col]) for y in years}
        qlog.check(f"temporal '{sev_label}' = severity {col} (each year)", all(v == 0 for v in d.values()),
                   "all years match" if all(v == 0 for v in d.values()) else f"differences {d}")


# ---------------------------------------------------------------------------
# Known-issue notes (numbers computed from data)
# ---------------------------------------------------------------------------

def build_notes(temporal: pd.DataFrame, top: pd.DataFrame, sev: pd.DataFrame, qlog: QualityLog) -> None:
    t = temporal
    yr = t.groupby("year")["collisions"].sum()
    minor = t[t.severity == "Minor"].groupby("year")["collisions"].sum()
    pdo = t[t.severity == "PDO"].groupby("year")["collisions"].sum()
    share = (minor / yr * 100).round(1)
    qlog.notes.append(
        "**Reporting-process break (Sept 2022).** Collision Reporting Centres replaced police reports. "
        f"Minor-injury collisions went from {minor.get(2021, 0):,} (2021) to {minor.get(2022, 0):,} (2022) "
        f"and {minor.get(2023, 0):,} (2023), i.e. {share.get(2021, 0)}% → {share.get(2023, 0)}% of all collisions, "
        f"while PDO fell from {pdo.get(2022, 0):,} to {pdo.get(2023, 0):,}. This is a classification change, not a "
        "safety trend. Handling: every row carries `reporting_period`; Minor/PDO trends are not compared across the break.")
    h24 = t[t.hour_code == 24].groupby("year")["collisions"].sum()
    h23 = t[t.hour_code == 23].groupby("year")["collisions"].sum()
    qlog.notes.append(
        "**Hour 24 spike.** Hour code 24 (23:01–24:00) has "
        f"{int(h24.sum()):,} collisions vs {int(h23.sum()):,} in the hour before it; its share drops to "
        f"{h24.get(2023, 0) / yr.get(2023, 1) * 100:.1f}% in 2023 from {h24.get(2021, 0) / yr.get(2021, 1) * 100:.1f}% "
        "in 2021. Consistent with unknown times being recorded as 00:00, but this cannot be proven from aggregates. "
        "Handling: kept, flagged `midnight_bucket`; late-night findings are not drawn from it.")
    h0 = int(t.loc[t.hour_code == 0, "collisions"].sum())
    qlog.notes.append(
        f"**Hour code 0.** {int((t.hour_code == 0).sum())} row ({h0} collision) has hour code 0, which the source "
        "itself labels hour group 'Invalid'. Handling: kept in totals so they reconcile; excluded from hour KPIs.")
    serious = t[t.severity_source == "SERIOUS"].year.unique().tolist()
    major = sorted(t[t.severity_source == "MAJOR"].year.unique().tolist())
    qlog.notes.append(
        f"**Severity label change.** 'SERIOUS' is used in {serious}; 'MAJOR' in {major[0]}–{major[-1]}. Both are "
        "mapped to `Serious`, justified by exact reconciliation with the Severity table's serious_injury_collisions.")
    src_ok = (top.assign(m=top["rank"] == top["rank_source"]).groupby(["year", "location_group"]).m.all())
    differ = [f"{y} {g}" for (y, g), ok in src_ok.items() if not ok]
    qlog.notes.append(
        "**Top-location ranking method varies by year.** The source rank equals standard competition ranking of "
        f"collision_count except in: {', '.join(differ)}. Handling: `rank` is recomputed consistently; "
        "`rank_source` is kept.")
    per_year = top.groupby(["year", "location_group"]).size().unstack()
    qlog.notes.append(
        "**Top-location lists are truncated (top ~50 per group per year, ties included).** Rows per year/group range "
        f"{int(per_year.min().min())}–{int(per_year.max().max())}. A site missing from a year's list is *unknown*, "
        "not zero, so location trends only use years where the site is listed.")
    labels = sorted(top.location_type_source.unique())
    qlog.notes.append(
        f"**Location type labels changed in 2022** ({', '.join(labels)}). Mapped to Intersection / Midblock; "
        "in the source, the 2022+ midblock labels share a single ranking.")
    ne = top[top.location_name.str.contains(r"\bBOULEVARD NE\b")].location_name.unique().tolist()
    if ne:
        qlog.notes.append(
            f"**Possible quadrant error left unchanged:** {ne}. Every other listing of this segment uses 'NW'; "
            "it is not corrected because the fix would be an inference, so this row may not match the NW listing.")
    gaps = sev.loc[sev.total_fatalities != sev.fatalities_intersection + sev.fatalities_midblock, "year"].tolist()
    road_user = sev[["fatality_bicyclist", "fatality_motorcyclist", "fatality_pedestrian", "fatality_vehicle_driver",
                     "fatality_vehicle_passenger"]].sum(axis=1)
    ru_gaps = {int(y): int(d) for y, d in zip(sev.year, sev.total_fatalities - road_user) if d}
    qlog.notes.append(
        f"**Fatality breakdowns don't always sum to the total.** Intersection + midblock is short by one in {gaps}; "
        f"road-user categories are short by {ru_gaps}. Handling: totals are used as published; breakdown "
        "KPIs note the unattributed remainder.")
    qlog.notes.append(
        f"**Coverage and currency.** Temporal and top locations cover {t.year.min()}–{t.year.max()}; severity covers "
        f"{sev.year.min()}–{sev.year.max()}. The tables are maintained manually and updated annually. The source is "
        "aggregated (no individual collisions, dates or coordinates), so day-level trends and coordinate checks are "
        "not possible.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

NULL_HANDLING = {
    "temporal": "row dropped if year/hour/collisions null; text dimensions flagged",
    "top_locations": "row dropped if year/rank/count/description null",
    "severity": "kept as null (no imputation)",
}


def summarize(key: str, raw: pd.DataFrame, clean: pd.DataFrame, qlog: QualityLog) -> DatasetSummary:
    m = latest_manifest(key)
    rules = qlog.rules_for(key)
    nulls = null_counts(standardize_columns(raw))
    s = DatasetSummary(
        dataset=key,
        source=f"{DATASETS[key].title} — {DATASETS[key].portal_url}",
        snapshot=m.snapshot_file if m else "n/a",
        rows_ingested=len(raw),
        nulls=nulls,
        null_handling={c: NULL_HANDLING[key] for c, n in nulls.items() if n},
        exact_duplicates_removed=sum(r.rows_affected for r in rules if r.rule == "Exact duplicate rows"),
        rows_dropped_invalid=sum(r.rows_affected for r in rules if r.action == "dropped" and r.rule != "Exact duplicate rows"),
        rows_flagged=int(clean["dq_flagged"].sum()),
        final_rows=len(clean),
    )
    qlog.summaries[key] = s
    return s


def run(out_dir: Path = DATA_PROCESSED, report_path: Path = REPORTS_DIR / "data_quality.md") -> dict[str, pd.DataFrame]:
    qlog = QualityLog()
    raws = {k: load_raw(latest_snapshot(k)) for k in DATASETS}
    cleaned = {
        "temporal": clean_temporal(raws["temporal"], qlog),
        "top_locations": clean_top_locations(raws["top_locations"], qlog),
        "severity": clean_severity(raws["severity"], qlog),
    }
    severity_internal_checks(cleaned["severity"], qlog)
    reconcile(cleaned["temporal"], cleaned["severity"], qlog)
    build_notes(cleaned["temporal"], cleaned["top_locations"], cleaned["severity"], qlog)
    for k in DATASETS:
        summarize(k, raws[k], cleaned[k], qlog)

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, df in cleaned.items():
        df.to_parquet(out_dir / f"{k}.parquet", index=False)
        df.to_csv(out_dir / f"{k}.csv", index=False)
    write_report(qlog, report_path)
    failed = [c.name for c in qlog.checks if not c.passed]
    if failed:
        log.warning("Validation checks failed: %s", failed)
    return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for k, df in run().items():
        print(f"{k:14s} {len(df):>7,d} rows")
