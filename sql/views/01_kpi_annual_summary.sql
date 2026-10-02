-- KPI: Annual summary, 2010 onward (source: severity table, one row per year).
-- Rates are per 100,000 residents using the City's population figure for that year.
-- YoY % change = (this year - prior year) / prior year * 100.
CREATE OR REPLACE VIEW {v_kpi_annual_summary} AS
SELECT
  year,
  reporting_period,
  population,
  total_collisions,
  collisions_per_100k,
  fatal_collisions,
  serious_injury_collisions,
  fatal_serious_collisions,
  fatal_serious_collisions_per_100k,
  total_fatalities,
  total_serious_injuries,
  ksi_persons,
  ksi_per_100k,
  fatalities_per_100k,
  pedestrian_collisions,
  bicycle_collisions,
  motorcycle_collisions,
  LAG(total_collisions) OVER (ORDER BY year) AS prior_year_collisions,
  ROUND(100.0 * (total_collisions - LAG(total_collisions) OVER (ORDER BY year))
        / NULLIF(LAG(total_collisions) OVER (ORDER BY year), 0), 1) AS collisions_yoy_pct,
  ROUND(100.0 * (fatal_serious_collisions - LAG(fatal_serious_collisions) OVER (ORDER BY year))
        / NULLIF(LAG(fatal_serious_collisions) OVER (ORDER BY year), 0), 1) AS fatal_serious_yoy_pct
FROM {severity};
