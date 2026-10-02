-- KPI: Collisions by day of week, all years and per-year average (source: temporal).
CREATE OR REPLACE VIEW {v_kpi_day_of_week} AS
SELECT
  day_of_week_num,
  day_of_week,
  is_weekend,
  SUM(collisions) AS collisions,
  ROUND(1.0 * SUM(collisions) / COUNT(DISTINCT year), 1) AS avg_collisions_per_year,
  SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) AS fatal_serious_collisions,
  ROUND(1000.0 * SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) / NULLIF(SUM(collisions), 0), 1)
    AS fatal_serious_per_1000
FROM {temporal}
GROUP BY day_of_week_num, day_of_week, is_weekend;
