-- KPI: Collisions per calendar month, 2019 onward (source: temporal).
-- Source rows for Mar/Jun/Sep/Dec are split across two seasons; summing per month recombines them.
-- same_month_other_years_avg = average of the same calendar month in all OTHER years (leave-one-out),
-- so a month's own value does not inflate its baseline. ratio_to_other_years > 1.5 is flagged as an outlier.
CREATE OR REPLACE VIEW {v_kpi_monthly} AS
WITH m AS (
  SELECT
    year,
    month,
    month_name,
    MIN(reporting_period) AS reporting_period,
    SUM(collisions) AS collisions,
    SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) AS fatal_serious_collisions
  FROM {temporal}
  GROUP BY year, month, month_name
),
b AS (
  SELECT
    m.*,
    (SUM(collisions) OVER (PARTITION BY month) - collisions)
      / NULLIF(COUNT(*) OVER (PARTITION BY month) - 1, 0) AS same_month_other_years_avg
  FROM m
)
SELECT
  year,
  month,
  month_name,
  CAST(CAST(year AS STRING) || '-' || LPAD(CAST(month AS STRING), 2, '0') || '-01' AS DATE) AS month_start,
  reporting_period,
  collisions,
  fatal_serious_collisions,
  ROUND(same_month_other_years_avg, 1) AS same_month_other_years_avg,
  ROUND(collisions / NULLIF(same_month_other_years_avg, 0), 2) AS ratio_to_other_years,
  CASE WHEN collisions / NULLIF(same_month_other_years_avg, 0) > 1.5 THEN TRUE ELSE FALSE END AS is_outlier_month
FROM b;
