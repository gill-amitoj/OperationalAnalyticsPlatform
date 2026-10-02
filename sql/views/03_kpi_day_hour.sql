-- KPI: Day-of-week x hour-of-day matrix (heatmap), per year (source: temporal).
-- hour_of_day h covers h:01 to (h+1):00 per the source definition (source hour code = h + 1).
-- Excludes hour code 0 (source-labelled 'Invalid'). Hour 23 includes the flagged hour-24 bucket.
CREATE OR REPLACE VIEW {v_kpi_day_hour} AS
SELECT
  year,
  reporting_period,
  day_of_week_num,
  day_of_week,
  hour_of_day,
  hour_label,
  MAX(CASE WHEN hour_code = 24 THEN TRUE ELSE FALSE END) AS is_midnight_bucket,
  SUM(collisions) AS collisions,
  SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) AS fatal_serious_collisions
FROM {temporal}
WHERE hour_of_day IS NOT NULL
GROUP BY year, reporting_period, day_of_week_num, day_of_week, hour_of_day, hour_label;
