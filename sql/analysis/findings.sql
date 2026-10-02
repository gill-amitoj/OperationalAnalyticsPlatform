-- Queries behind reports/findings_brief.md. Run with: python -m traffic_safety.findings
-- Each block starts with "-- name: <id>" and runs against the DuckDB tables/views.

-- name: f1_night_severity
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

-- name: f1_night_severity_by_year
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

-- name: f2_afternoon_peak
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

-- name: f2_busiest_cells
-- The five busiest day x hour cells, 2019-2023.
SELECT day_of_week, hour_label, SUM(collisions) AS collisions
FROM v_kpi_day_hour
GROUP BY day_of_week, hour_label
ORDER BY collisions DESC
LIMIT 5;

-- name: f3_chronic_hotspots
-- Locations on the City's published top list in all five years 2019-2023.
SELECT COALESCE(location_group, 'All') AS location_group, COUNT(*) AS locations,
       SUM(collisions_in_listed_years) AS collisions_2019_2023
FROM v_kpi_location_persistence
WHERE years_listed = 5
GROUP BY ROLLUP (location_group)
ORDER BY location_group;

-- name: f3_top_hotspot
-- Year-by-year record of the highest-volume chronic hotspot.
SELECT year, location_name, collision_count, rank
FROM v_kpi_top_locations
WHERE location_key = (
  SELECT location_key FROM v_kpi_location_persistence
  WHERE years_listed = 5 ORDER BY collisions_in_listed_years DESC LIMIT 1)
ORDER BY year;

-- name: f3_max_single_year_location
-- Largest single-location, single-year collision count in the published lists.
SELECT year, location_group, location_name, collision_count
FROM v_kpi_top_locations
ORDER BY collision_count DESC
LIMIT 3;

-- name: f4_fatal_serious_trend
-- Fatal + serious-injury collisions per 100k residents: 2010, the 2020 low, and 2023.
SELECT year, reporting_period, fatal_serious_collisions, fatal_serious_collisions_per_100k, total_fatalities,
       ksi_persons, ksi_per_100k
FROM v_kpi_annual_summary
WHERE year IN (2010, 2019, 2020, 2021, 2022, 2023)
ORDER BY year;

-- name: f4_fatalities_rank
-- How 2023 fatalities compare with every year since 2010.
SELECT year, total_fatalities,
       RANK() OVER (ORDER BY total_fatalities DESC) AS rank_highest
FROM v_kpi_annual_summary
ORDER BY year;

-- name: f5_pedestrian_ksi
-- Pedestrians killed or seriously injured per year.
SELECT year, reporting_period, fatalities, serious_injuries, fatalities + serious_injuries AS ksi
FROM v_kpi_vulnerable_road_users
WHERE road_user = 'Pedestrian'
ORDER BY year;

-- name: f6_outlier_months
-- Months more than 1.5x the average of the same calendar month in other years.
SELECT year, month_name, collisions, same_month_other_years_avg, ratio_to_other_years, fatal_serious_collisions
FROM v_kpi_monthly
WHERE is_outlier_month
ORDER BY ratio_to_other_years DESC;

-- name: f6_feb_2019_severity
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
