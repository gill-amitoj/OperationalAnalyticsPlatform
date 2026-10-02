-- KPI: Collisions by severity and year, with share of the year's total (source: temporal).
-- Minor vs PDO shares are NOT comparable across the Sept 2022 reporting change (see reporting_period).
CREATE OR REPLACE VIEW {v_kpi_severity_trend} AS
SELECT
  year,
  reporting_period,
  severity,
  MIN(severity_rank) AS severity_rank,
  SUM(collisions) AS collisions,
  ROUND(100.0 * SUM(collisions) / SUM(SUM(collisions)) OVER (PARTITION BY year), 2) AS pct_of_year
FROM {temporal}
GROUP BY year, reporting_period, severity;
