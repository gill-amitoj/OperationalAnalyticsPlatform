-- KPI drill-down: hourly profile of each month vs the same calendar month in all OTHER years
-- (leave-one-out baseline), used to explain outlier months flagged in v_kpi_monthly.
CREATE OR REPLACE VIEW {v_kpi_month_hour_drilldown} AS
WITH mh AS (
  SELECT year, month, month_name, hour_of_day, SUM(collisions) AS collisions,
         SUM(CASE WHEN is_fatal_or_serious THEN collisions ELSE 0 END) AS fatal_serious_collisions
  FROM {temporal}
  WHERE hour_of_day IS NOT NULL
  GROUP BY year, month, month_name, hour_of_day
),
yrs AS (
  SELECT month, COUNT(DISTINCT year) AS n_years FROM {temporal} GROUP BY month
),
tot AS (
  SELECT month, hour_of_day, SUM(collisions) AS all_years_collisions
  FROM mh GROUP BY month, hour_of_day
)
SELECT
  mh.year,
  mh.month,
  mh.month_name,
  mh.hour_of_day,
  mh.collisions,
  mh.fatal_serious_collisions,
  ROUND(1.0 * (tot.all_years_collisions - mh.collisions) / NULLIF(yrs.n_years - 1, 0), 1)
    AS same_month_other_years_avg
FROM mh
JOIN tot ON tot.month = mh.month AND tot.hour_of_day = mh.hour_of_day
JOIN yrs ON yrs.month = mh.month;
