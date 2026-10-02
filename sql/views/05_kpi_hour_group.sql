-- KPI: Collisions by the City's time-of-day periods (source: temporal).
-- Periods have different lengths (Evening = 12 h, others = 3 h), so collisions_per_hour normalizes
-- by the number of clock hours in the period; this is the fair comparison.
CREATE OR REPLACE VIEW {v_kpi_hour_group} AS
SELECT
  hour_group,
  COUNT(DISTINCT hour_of_day) AS hours_in_period,
  SUM(collisions) AS collisions,
  ROUND(1.0 * SUM(collisions) / COUNT(DISTINCT hour_of_day), 1) AS collisions_per_hour,
  SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) AS fatal_serious_collisions,
  ROUND(1.0 * SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) / COUNT(DISTINCT hour_of_day), 2)
    AS fatal_serious_per_hour
FROM {temporal}
WHERE hour_of_day IS NOT NULL
GROUP BY hour_group;
