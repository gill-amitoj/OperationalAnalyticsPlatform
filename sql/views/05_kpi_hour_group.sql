-- KPI: Collisions by the City's time-of-day periods (source: temporal).
-- Periods have different lengths, so collisions_per_hour divides by the period's clock hours
-- (fixed by the source definition: Evening 18:00-06:00 = 12 h; AM Peak, AM Non-Peak, PM Non-Peak,
-- PM Peak = 3 h each), not by the hours that happen to have data.
CREATE OR REPLACE VIEW {v_kpi_hour_group} AS
WITH g AS (
  SELECT
    hour_group,
    CASE WHEN hour_group = 'Evening' THEN 12 ELSE 3 END AS hours_in_period,
    SUM(collisions) AS collisions,
    SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) AS fatal_serious_collisions
  FROM {temporal}
  WHERE hour_of_day IS NOT NULL
  GROUP BY hour_group
)
SELECT
  hour_group,
  hours_in_period,
  collisions,
  ROUND(1.0 * collisions / hours_in_period, 1) AS collisions_per_hour,
  fatal_serious_collisions,
  ROUND(1.0 * fatal_serious_collisions / hours_in_period, 2) AS fatal_serious_per_hour
FROM g;
