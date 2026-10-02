-- KPI: Chronic hotspots = locations that appear on the City's top list in multiple years.
-- A year when a location is not listed is UNKNOWN (below the cutoff), not zero, so averages
-- use only the years listed.
CREATE OR REPLACE VIEW {v_kpi_location_persistence} AS
SELECT
  location_key,
  location_group,
  MIN(location_name) AS location_name,
  COUNT(DISTINCT year) AS years_listed,
  MIN(year) AS first_year_listed,
  MAX(year) AS last_year_listed,
  SUM(collision_count) AS collisions_in_listed_years,
  ROUND(1.0 * SUM(collision_count) / COUNT(DISTINCT year), 1) AS avg_collisions_per_listed_year,
  MIN(rank) AS best_rank,
  SUM(CASE WHEN rank <= 10 THEN 1 ELSE 0 END) AS years_in_top_10
FROM {top_locations}
GROUP BY location_key, location_group;
