-- KPI: Highest-collision locations per year (source: top_locations; City publishes ~top 50 per group).
-- rank = standard competition rank of collision_count within year x location_group (ties share a rank).
CREATE OR REPLACE VIEW {v_kpi_top_locations} AS
SELECT
  year,
  location_group,
  rank,
  location_name,
  location_key,
  collision_count,
  reporting_period
FROM {top_locations};
