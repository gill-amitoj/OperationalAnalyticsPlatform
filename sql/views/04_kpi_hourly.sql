-- KPI: Hour-of-day profile across all years (source: temporal).
-- pct_of_collisions = share of all collisions with a valid hour.
-- fatal_serious_per_1000 = fatal + serious-injury collisions per 1,000 collisions in that hour (severity mix).
CREATE OR REPLACE VIEW {v_kpi_hourly} AS
WITH h AS (
  SELECT
    hour_of_day,
    hour_label,
    hour_group,
    SUM(collisions) AS collisions,
    SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) AS fatal_serious_collisions,
    SUM(CASE WHEN severity = 'Fatal' THEN collisions ELSE 0 END) AS fatal_collisions
  FROM {temporal}
  WHERE hour_of_day IS NOT NULL
  GROUP BY hour_of_day, hour_label, hour_group
)
SELECT
  h.*,
  CASE WHEN hour_of_day = 23 THEN TRUE ELSE FALSE END AS is_midnight_bucket,
  ROUND(100.0 * collisions / SUM(collisions) OVER (), 2) AS pct_of_collisions,
  ROUND(1000.0 * fatal_serious_collisions / NULLIF(collisions, 0), 1) AS fatal_serious_per_1000
FROM h;
