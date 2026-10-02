-- KPI: Collisions by day of week, all years and per-year average (source: temporal).
-- The per-year average divides by ALL years in the data, so a day missing from a year counts as zero.
CREATE OR REPLACE VIEW {v_kpi_day_of_week} AS
WITH n AS (SELECT COUNT(DISTINCT year) AS n_years FROM {temporal})
SELECT
  t.day_of_week_num,
  t.day_of_week,
  t.is_weekend,
  SUM(t.collisions) AS collisions,
  ROUND(1.0 * SUM(t.collisions) / MAX(n.n_years), 1) AS avg_collisions_per_year,
  SUM(CASE WHEN t.is_fatal_or_serious THEN t.collisions ELSE 0 END) AS fatal_serious_collisions,
  ROUND(1000.0 * SUM(CASE WHEN t.is_fatal_or_serious THEN t.collisions ELSE 0 END) / NULLIF(SUM(t.collisions), 0), 1)
    AS fatal_serious_per_1000
FROM {temporal} t
CROSS JOIN n
GROUP BY t.day_of_week_num, t.day_of_week, t.is_weekend;
