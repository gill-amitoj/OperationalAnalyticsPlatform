# Findings evidence

_Generated 2026-10-02 02:27 UTC by `python -m traffic_safety.findings` from `sql/analysis/findings.sql`. Each result below is the live output of the query above it._

## `f1_night_severity`

```sql
-- Fatal+serious collisions per 1,000 collisions: 00:00-05:00 vs 07:00-18:00, all years.
-- Hour 23 (contains the flagged hour-24 bucket) is excluded from both windows.
SELECT
  SUM(CASE WHEN hour_of_day BETWEEN 0 AND 4 THEN collisions ELSE 0 END) AS night_collisions,
  SUM(CASE WHEN hour_of_day BETWEEN 0 AND 4 AND is_fatal_or_serious THEN collisions ELSE 0 END) AS night_fatal_serious,
  ROUND(1000.0 * SUM(CASE WHEN hour_of_day BETWEEN 0 AND 4 AND is_fatal_or_serious THEN collisions ELSE 0 END)
        / SUM(CASE WHEN hour_of_day BETWEEN 0 AND 4 THEN collisions ELSE 0 END), 1) AS night_per_1000,
  SUM(CASE WHEN hour_of_day BETWEEN 7 AND 17 THEN collisions ELSE 0 END) AS day_collisions,
  SUM(CASE WHEN hour_of_day BETWEEN 7 AND 17 AND is_fatal_or_serious THEN collisions ELSE 0 END) AS day_fatal_serious,
  ROUND(1000.0 * SUM(CASE WHEN hour_of_day BETWEEN 7 AND 17 AND is_fatal_or_serious THEN collisions ELSE 0 END)
        / SUM(CASE WHEN hour_of_day BETWEEN 7 AND 17 THEN collisions ELSE 0 END), 1) AS day_per_1000,
  ROUND(100.0 * SUM(CASE WHEN hour_of_day BETWEEN 0 AND 4 THEN collisions ELSE 0 END) / SUM(collisions), 1)
    AS night_pct_of_all_collisions,
  SUM(CASE WHEN hour_of_day BETWEEN 0 AND 4 AND severity = 'Fatal' THEN collisions ELSE 0 END) AS night_fatal,
  SUM(CASE WHEN severity = 'Fatal' THEN collisions ELSE 0 END) AS all_fatal
FROM temporal;
```

|   night_collisions |   night_fatal_serious |   night_per_1000 |   day_collisions |   day_fatal_serious |   day_per_1000 |   night_pct_of_all_collisions |   night_fatal |   all_fatal |
|-------------------:|----------------------:|-----------------:|-----------------:|--------------------:|---------------:|------------------------------:|--------------:|------------:|
|               3132 |                   127 |             40.5 |            71339 |                 775 |           10.9 |                           3.3 |            20 |          72 |

## `f1_night_severity_by_year`

```sql
-- Robustness: does the night/day gap hold in every year, including across the 2022 reporting change?
SELECT
  year,
  ROUND(1000.0 * SUM(CASE WHEN hour_of_day BETWEEN 0 AND 4 AND is_fatal_or_serious THEN collisions ELSE 0 END)
        / SUM(CASE WHEN hour_of_day BETWEEN 0 AND 4 THEN collisions ELSE 0 END), 1) AS night_per_1000,
  ROUND(1000.0 * SUM(CASE WHEN hour_of_day BETWEEN 7 AND 17 AND is_fatal_or_serious THEN collisions ELSE 0 END)
        / SUM(CASE WHEN hour_of_day BETWEEN 7 AND 17 THEN collisions ELSE 0 END), 1) AS day_per_1000
FROM temporal
GROUP BY year
ORDER BY year;
```

|   year |   night_per_1000 |   day_per_1000 |
|-------:|-----------------:|---------------:|
|   2019 |             39.1 |            9.5 |
|   2020 |             39.5 |           10   |
|   2021 |             47.3 |           10.6 |
|   2022 |             36   |           12.4 |
|   2023 |             42.1 |           11.9 |

## `f2_afternoon_peak`

```sql
-- Share of collisions (with a valid hour) occurring 15:00-18:00, overall and by year.
SELECT
  COALESCE(CAST(year AS VARCHAR), 'All years') AS year,
  ROUND(100.0 * SUM(CASE WHEN hour_of_day BETWEEN 15 AND 17 THEN collisions ELSE 0 END) / SUM(collisions), 1)
    AS pct_15_to_18,
  SUM(CASE WHEN hour_of_day BETWEEN 15 AND 17 THEN collisions ELSE 0 END) AS collisions_15_to_18
FROM temporal
WHERE hour_of_day IS NOT NULL
GROUP BY ROLLUP (year)
ORDER BY year;
```

| year      |   pct_15_to_18 |   collisions_15_to_18 |
|:----------|---------------:|----------------------:|
| 2019      |           28.5 |                  6245 |
| 2020      |           27.6 |                  4363 |
| 2021      |           27.6 |                  4791 |
| 2022      |           27.4 |                  5526 |
| 2023      |           28.7 |                  5755 |
| All years |           28   |                 26680 |

## `f2_busiest_cells`

```sql
-- The five busiest day x hour cells, 2019-2023.
SELECT day_of_week, hour_label, SUM(collisions) AS collisions
FROM v_kpi_day_hour
GROUP BY day_of_week, hour_label
ORDER BY collisions DESC
LIMIT 5;
```

| day_of_week   | hour_label   |   collisions |
|:--------------|:-------------|-------------:|
| FRIDAY        | 16:00–17:00  |         1763 |
| FRIDAY        | 15:00–16:00  |         1715 |
| WEDNESDAY     | 16:00–17:00  |         1708 |
| THURSDAY      | 16:00–17:00  |         1670 |
| TUESDAY       | 16:00–17:00  |         1654 |

## `f3_chronic_hotspots`

```sql
-- Locations on the City's published top list in all five years 2019-2023.
SELECT COALESCE(location_group, 'All') AS location_group, COUNT(*) AS locations,
       SUM(collisions_in_listed_years) AS collisions_2019_2023
FROM v_kpi_location_persistence
WHERE years_listed = 5
GROUP BY ROLLUP (location_group)
ORDER BY location_group;
```

| location_group   |   locations |   collisions_2019_2023 |
|:-----------------|------------:|-----------------------:|
| All              |          26 |                   5503 |
| Intersection     |          23 |                   5239 |
| Midblock         |           3 |                    264 |

## `f3_top_hotspot`

```sql
-- Year-by-year record of the highest-volume chronic hotspot.
SELECT year, location_name, collision_count, rank
FROM v_kpi_top_locations
WHERE location_key = (
  SELECT location_key FROM v_kpi_location_persistence
  WHERE years_listed = 5 ORDER BY collisions_in_listed_years DESC LIMIT 1)
ORDER BY year;
```

|   year | location_name                 |   collision_count |   rank |
|-------:|:------------------------------|------------------:|-------:|
|   2019 | 107 AVENUE NW & 142 STREET NW |                92 |      2 |
|   2020 | 107 AVENUE NW & 142 STREET NW |                71 |      1 |
|   2021 | 107 AVENUE NW & 142 STREET NW |                42 |     12 |
|   2022 | 107 AVENUE NW & 142 STREET NW |                75 |      2 |
|   2023 | 107 AVENUE NW & 142 STREET NW |               169 |      1 |

## `f3_max_single_year_location`

```sql
-- Largest single-location, single-year collision count in the published lists.
SELECT year, location_group, location_name, collision_count
FROM v_kpi_top_locations
ORDER BY collision_count DESC
LIMIT 3;
```

|   year | location_group   | location_name                       |   collision_count |
|-------:|:-----------------|:------------------------------------|------------------:|
|   2023 | Intersection     | 107 AVENUE NW & 142 STREET NW       |               169 |
|   2019 | Intersection     | YELLOWHEAD TRAIL NW & 149 STREET NW |                98 |
|   2019 | Intersection     | 107 AVENUE NW & 142 STREET NW       |                92 |

## `f4_fatal_serious_trend`

```sql
-- Fatal + serious-injury collisions per 100k residents: 2010, the 2020 low, and 2023.
SELECT year, reporting_period, fatal_serious_collisions, fatal_serious_collisions_per_100k, total_fatalities,
       ksi_persons, ksi_per_100k
FROM v_kpi_annual_summary
WHERE year IN (2010, 2019, 2020, 2021, 2022, 2023)
ORDER BY year;
```

|   year | reporting_period               |   fatal_serious_collisions |   fatal_serious_collisions_per_100k |   total_fatalities |   ksi_persons |   ksi_per_100k |
|-------:|:-------------------------------|---------------------------:|------------------------------------:|-------------------:|--------------:|---------------:|
|   2010 | Pre-CRC (police reports)       |                        447 |                               56.37 |                 27 |           518 |          65.32 |
|   2019 | Pre-CRC (police reports)       |                        258 |                               25.98 |                 14 |           282 |          28.4  |
|   2020 | Pre-CRC (police reports)       |                        201 |                               19.19 |                 12 |           243 |          23.2  |
|   2021 | Pre-CRC (police reports)       |                        245 |                               23.16 |                 16 |           275 |          26    |
|   2022 | Transition (CRC from Sep 2022) |                        314 |                               28.87 |                 14 |           351 |          32.27 |
|   2023 | CRC                            |                        325 |                               28.79 |                 24 |           369 |          32.69 |

## `f4_fatalities_rank`

```sql
-- How 2023 fatalities compare with every year since 2010.
SELECT year, total_fatalities,
       RANK() OVER (ORDER BY total_fatalities DESC) AS rank_highest
FROM v_kpi_annual_summary
ORDER BY year;
```

|   year |   total_fatalities |   rank_highest |
|-------:|-------------------:|---------------:|
|   2010 |                 27 |              2 |
|   2011 |                 22 |              8 |
|   2012 |                 27 |              2 |
|   2013 |                 23 |              6 |
|   2014 |                 23 |              6 |
|   2015 |                 32 |              1 |
|   2016 |                 22 |              8 |
|   2017 |                 27 |              2 |
|   2018 |                 19 |             10 |
|   2019 |                 14 |             12 |
|   2020 |                 12 |             14 |
|   2021 |                 16 |             11 |
|   2022 |                 14 |             12 |
|   2023 |                 24 |              5 |

## `f5_pedestrian_ksi`

```sql
-- Pedestrians killed or seriously injured per year.
SELECT year, reporting_period, fatalities, serious_injuries, fatalities + serious_injuries AS ksi
FROM v_kpi_vulnerable_road_users
WHERE road_user = 'Pedestrian'
ORDER BY year;
```

|   year | reporting_period               |   fatalities |   serious_injuries |   ksi |
|-------:|:-------------------------------|-------------:|-------------------:|------:|
|   2010 | Pre-CRC (police reports)       |            4 |                 81 |    85 |
|   2011 | Pre-CRC (police reports)       |            8 |                 85 |    93 |
|   2012 | Pre-CRC (police reports)       |            8 |                 86 |    94 |
|   2013 | Pre-CRC (police reports)       |            6 |                 77 |    83 |
|   2014 | Pre-CRC (police reports)       |            9 |                 83 |    92 |
|   2015 | Pre-CRC (police reports)       |           12 |                 58 |    70 |
|   2016 | Pre-CRC (police reports)       |           10 |                 55 |    65 |
|   2017 | Pre-CRC (police reports)       |            9 |                 60 |    69 |
|   2018 | Pre-CRC (police reports)       |            6 |                 63 |    69 |
|   2019 | Pre-CRC (police reports)       |            3 |                 60 |    63 |
|   2020 | Pre-CRC (police reports)       |            2 |                 30 |    32 |
|   2021 | Pre-CRC (police reports)       |            5 |                 46 |    51 |
|   2022 | Transition (CRC from Sep 2022) |            4 |                 52 |    56 |
|   2023 | CRC                            |            5 |                 73 |    78 |

## `f6_outlier_months`

```sql
-- Months more than 1.5x the average of the same calendar month in other years.
SELECT year, month_name, collisions, same_month_other_years_avg, ratio_to_other_years, fatal_serious_collisions
FROM v_kpi_monthly
WHERE is_outlier_month
ORDER BY ratio_to_other_years DESC;
```

|   year | month_name   |   collisions |   same_month_other_years_avg |   ratio_to_other_years |   fatal_serious_collisions |
|-------:|:-------------|-------------:|-----------------------------:|-----------------------:|---------------------------:|
|   2019 | FEBRUARY     |         3721 |                       1695.8 |                   2.19 |                         18 |
|   2020 | JANUARY      |         2883 |                       1906   |                   1.51 |                         16 |

## `f6_feb_2019_severity`

```sql
-- Was the Feb 2019 surge in severe collisions or minor ones? Fatal+serious share vs other Februaries.
SELECT
  CASE WHEN year = 2019 THEN 'Feb 2019' ELSE 'Other Februaries' END AS period,
  SUM(collisions) AS collisions,
  SUM(fatal_serious_collisions) AS fatal_serious,
  ROUND(1000.0 * SUM(fatal_serious_collisions) / SUM(collisions), 1) AS fatal_serious_per_1000
FROM v_kpi_monthly
WHERE month = 2
GROUP BY 1
ORDER BY 1;
```

| period           |   collisions |   fatal_serious |   fatal_serious_per_1000 |
|:-----------------|-------------:|----------------:|-------------------------:|
| Feb 2019         |         3721 |              18 |                      4.8 |
| Other Februaries |         6783 |              49 |                      7.2 |
