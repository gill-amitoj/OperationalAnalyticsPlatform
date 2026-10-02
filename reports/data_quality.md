# Data Quality Report

_Generated 2026-10-02 02:15 UTC by `traffic_safety.clean`. All numbers are computed by the pipeline from the raw snapshots listed below._

## Summary

| Dataset | Rows ingested | Exact duplicates removed | Invalid rows dropped | Rows flagged (kept) | Final rows | % retained |
|---|---:|---:|---:|---:|---:|---:|
| temporal | 18,545 | 0 | 0 | 796 | 18,545 | 100.00% |
| top_locations | 561 | 0 | 0 | 0 | 561 | 100.00% |
| severity | 14 | 0 | 0 | 0 | 14 | 100.00% |

**Validation checks:** 9 of 9 passed.

## `temporal`

- Source: Annual Collision Report: Temporal — https://data.edmonton.ca/d/jduq-w5pj
- Raw snapshot: `temporal_20261002T020529Z.csv`

### Nulls per column (raw)

| Column | Nulls | Handling |
|---|---:|---|
| `collision_report_year` | 0 | n/a — no nulls |
| `season` | 0 | n/a — no nulls |
| `collision_report_month_name` | 0 | n/a — no nulls |
| `collision_report_hour` | 0 | n/a — no nulls |
| `hour_group_name` | 0 | n/a — no nulls |
| `is_weekend` | 0 | n/a — no nulls |
| `day_of_week_name` | 0 | n/a — no nulls |
| `collision_classification` | 0 | n/a — no nulls |
| `collisions` | 0 | n/a — no nulls |

### Cleaning rules applied

| Rule | Rows affected | Action | Detail |
|---|---:|---|---|
| Exact duplicate rows | 0 | dropped | identical across every column |
| Missing or non-integer value in year, hour_code, collisions | 0 | dropped |  |
| Non-positive collision count | 0 | dropped | a count row must represent at least one collision |
| Future or implausible year | 0 | dropped | outside 1990–2026 |
| Unknown month name | 0 | dropped |  |
| Hour code outside 0–24 | 0 | dropped |  |
| Hour code 0 (source hour group = 'Invalid') | 1 | flagged | kept in totals; excluded from hour-of-day KPIs |
| Hour code 24 (23:01–24:00), possible default-midnight times | 795 | flagged | kept as hour 23:00–24:00; caveated in hour KPIs |
| Unknown day-of-week name | 0 | flagged |  |
| Unknown severity label | 0 | flagged |  |
| Hour group inconsistent with hour code | 0 | flagged | per field definition |
| Season inconsistent with month | 0 | flagged |  |
| Weekday/weekend label inconsistent with day name | 0 | flagged |  |
| Same dimension combination appears more than once | 0 | flagged | would double count; investigate if non-zero |
| Severity label 'MAJOR' harmonized to 'Serious' | 953 | fixed | label used 2020+; 2019 uses 'SERIOUS' |

## `top_locations`

- Source: Annual Collision Report: Top Collision Locations — https://data.edmonton.ca/d/mf6n-s5ts
- Raw snapshot: `top_locations_20261002T020530Z.csv`

### Nulls per column (raw)

| Column | Nulls | Handling |
|---|---:|---|
| `year` | 0 | n/a — no nulls |
| `location_type` | 0 | n/a — no nulls |
| `rank` | 0 | n/a — no nulls |
| `location_description` | 0 | n/a — no nulls |
| `collision_count` | 0 | n/a — no nulls |

### Cleaning rules applied

| Rule | Rows affected | Action | Detail |
|---|---:|---|---|
| Exact duplicate rows | 0 | dropped | identical across every column |
| Missing or non-integer value in year, rank_source, collision_count | 0 | dropped |  |
| Non-positive collision count | 0 | dropped |  |
| Non-positive rank | 0 | dropped |  |
| Future or implausible year | 0 | dropped |  |
| Missing location description | 0 | dropped |  |
| Unrecognized location type | 0 | flagged |  |
| Location type 'MID AVENUE' / 'MID STREET' / 'SOUTH OF INTERSECTION' grouped as 'Midblock' | 103 | fixed | 2022+ labels; these share one ranking in the source |
| Location name normalized (typos, abbreviations, 'AND'→'&', commas) | 12 | fixed | explicit lookup tables in clean.py |
| Rank recomputed from collision_count (standard competition ranking within year × group) | 114 | fixed | source ranking method differs by year; original kept as rank_source |
| Same location listed twice in one year | 0 | flagged |  |

## `severity`

- Source: Annual Collision Report: Collision Severity — https://data.edmonton.ca/d/77sf-j5rj
- Raw snapshot: `severity_20261002T020531Z.csv`

### Nulls per column (raw)

| Column | Nulls | Handling |
|---|---:|---|
| `year` | 0 | n/a — no nulls |
| `population` | 0 | n/a — no nulls |
| `private_passenger_vehicles` | 0 | n/a — no nulls |
| `private_motorcycles` | 0 | n/a — no nulls |
| `bicycle_collisions` | 0 | n/a — no nulls |
| `motorcycle_collisions` | 0 | n/a — no nulls |
| `pedestrian_collisions` | 0 | n/a — no nulls |
| `total_collisions` | 0 | n/a — no nulls |
| `collisions_per_1_000` | 0 | n/a — no nulls |
| `property_damage_only_pdo` | 0 | n/a — no nulls |
| `injuries_fatalities_per_1` | 0 | n/a — no nulls |
| `injuries_per_1_000_population` | 0 | n/a — no nulls |
| `fatalities_per_1_000` | 0 | n/a — no nulls |
| `fatality_bicyclist` | 0 | n/a — no nulls |
| `fatality_motorcyclist` | 0 | n/a — no nulls |
| `fatality_pedestrian` | 0 | n/a — no nulls |
| `fatality_vehicle_driver` | 0 | n/a — no nulls |
| `fatality_vehicle_passenger` | 0 | n/a — no nulls |
| `total_fatalities` | 0 | n/a — no nulls |
| `serious_injuries_bicyclist` | 0 | n/a — no nulls |
| `minor_injuries_bicyclist` | 0 | n/a — no nulls |
| `serious_injury_motorcyclist` | 0 | n/a — no nulls |
| `minor_injuries_motorcyclist` | 0 | n/a — no nulls |
| `serious_injuries_pedestrian` | 0 | n/a — no nulls |
| `minor_injuries_pedestrian` | 0 | n/a — no nulls |
| `total_serious_injuries` | 0 | n/a — no nulls |
| `total_minor_injuries` | 0 | n/a — no nulls |
| `total_serious_minor_injuries` | 0 | n/a — no nulls |
| `total_fatalities_injuries` | 0 | n/a — no nulls |
| `fatalities_intersection` | 0 | n/a — no nulls |
| `fatalities_intersection_1` | 0 | n/a — no nulls |
| `fatalities_midblock` | 0 | n/a — no nulls |
| `fatalities_midblock_percent` | 0 | n/a — no nulls |
| `injuries_intersection` | 0 | n/a — no nulls |
| `injuries_intersection_percent` | 0 | n/a — no nulls |
| `injuries_midblock` | 0 | n/a — no nulls |
| `injuries_midblock_percent` | 0 | n/a — no nulls |
| `fatal_collisions` | 0 | n/a — no nulls |
| `serious_injury_collisions` | 0 | n/a — no nulls |
| `minor_injury_collisions` | 0 | n/a — no nulls |
| `total_serious_minor_injury` | 0 | n/a — no nulls |
| `total_fatal_injury_collisions` | 0 | n/a — no nulls |

### Cleaning rules applied

| Rule | Rows affected | Action | Detail |
|---|---:|---|---|
| Exact duplicate rows | 0 | dropped | identical across every column |
| Missing or non-integer value in year | 0 | dropped |  |
| Future or implausible year | 0 | dropped |  |
| Negative count or rate | 0 | dropped |  |
| Year appears more than once | 0 | flagged |  |
| Rates recomputed per 100,000 population at full precision | 14 | fixed | source rates are rounded to 2 dp per 1,000 |

## Validation checks

| Check | Result | Detail |
|---|---|---|
| severity: PDO + fatal/injury collisions = total_collisions | PASS | all years match |
| severity: fatal + serious + minor collisions = total fatal/injury collisions | PASS | all years match |
| severity: fatalities by road user ≤ total_fatalities | PASS | unattributed fatalities {2023: 1} (no road-user category recorded) |
| severity: intersection + midblock fatalities ≤ total_fatalities | PASS | unattributed fatalities {2011: 1, 2015: 1, 2016: 1, 2017: 1, 2021: 1} (no location type recorded) |
| temporal total = severity total_collisions (each year) | PASS | years 2019–2023; differences {2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0} |
| temporal 'Fatal' = severity fatal_collisions (each year) | PASS | all years match |
| temporal 'Serious' = severity serious_injury_collisions (each year) | PASS | all years match |
| temporal 'Minor' = severity minor_injury_collisions (each year) | PASS | all years match |
| temporal 'PDO' = severity pdo_collisions (each year) | PASS | all years match |

## Known data issues and how they are handled

1. **Reporting-process break (Sept 2022).** Collision Reporting Centres replaced police reports. Minor-injury collisions went from 1,444 (2021) to 3,283 (2022) and 7,581 (2023), i.e. 8.3% → 37.8% of all collisions, while PDO fell from 16,565 to 12,133. This is a classification change, not a safety trend. Handling: every row carries `reporting_period`; Minor/PDO trends are not compared across the break.
2. **Hour 24 spike.** Hour code 24 (23:01–24:00) has 2,777 collisions vs 1,605 in the hour before it; its share drops to 1.6% in 2023 from 3.7% in 2021. Consistent with unknown times being recorded as 00:00, but this cannot be proven from aggregates. Handling: kept, flagged `midnight_bucket`; late-night findings are not drawn from it.
3. **Hour code 0.** 1 row (1 collision) has hour code 0, which the source itself labels hour group 'Invalid'. Handling: kept in totals so they reconcile; excluded from hour KPIs.
4. **Severity label change.** 'SERIOUS' is used in [2019]; 'MAJOR' in 2020–2023. Both are mapped to `Serious`, justified by exact reconciliation with the Severity table's serious_injury_collisions.
5. **Top-location ranking method varies by year.** The source rank equals standard competition ranking of collision_count except in: 2019 Intersection, 2019 Midblock, 2020 Intersection. Handling: `rank` is recomputed consistently; `rank_source` is kept.
6. **Top-location lists are truncated (top ~50 per group per year, ties included).** Rows per year/group range 50–77. A site missing from a year's list is *unknown*, not zero, so location trends only use years where the site is listed.
7. **Location type labels changed in 2022** (INTERSECTION, MID AVENUE, MID STREET, MIDBLOCK, SOUTH OF INTERSECTION). Mapped to Intersection / Midblock; in the source, the 2022+ midblock labels share a single ranking.
8. **Possible quadrant error left unchanged:** ['GATEWAY BOULEVARD NE BETWEEN 34-39A AVENUE NW']. Every other listing of this segment uses 'NW'; it is not corrected because the fix would be an inference, so this row may not match the NW listing.
9. **Fatality breakdowns don't always sum to the total.** Intersection + midblock is short by one in [2011, 2015, 2016, 2017, 2021]; road-user categories are short by {2023: 1}. Handling: totals are used as published; breakdown KPIs note the unattributed remainder.
10. **Coverage and currency.** Temporal and top locations cover 2019–2023; severity covers 2010–2023. The tables are maintained manually and updated annually. The source is aggregated (no individual collisions, dates or coordinates), so day-level trends and coordinate checks are not possible.
