-- KPI: Where people are killed or injured: intersection vs midblock, by year (source: severity).
-- Fatality parts may sum to one less than total_fatalities in some years (no location type recorded).
CREATE OR REPLACE VIEW {v_kpi_intersection_midblock} AS
SELECT
  year,
  reporting_period,
  total_fatalities,
  fatalities_intersection,
  fatalities_midblock,
  total_fatalities - fatalities_intersection - fatalities_midblock AS fatalities_unattributed,
  injuries_intersection,
  injuries_midblock,
  ROUND(100.0 * injuries_intersection / NULLIF(injuries_intersection + injuries_midblock, 0), 1)
    AS pct_injuries_at_intersections
FROM {severity};
