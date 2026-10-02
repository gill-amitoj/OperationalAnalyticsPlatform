-- KPI: Vulnerable road users (pedestrians, cyclists, motorcyclists) by year (source: severity).
-- serious_injuries and fatalities count PEOPLE; collisions counts COLLISIONS involving that road user.
CREATE OR REPLACE VIEW {v_kpi_vulnerable_road_users} AS
SELECT year, reporting_period, 'Pedestrian' AS road_user,
       pedestrian_collisions AS collisions, fatality_pedestrian AS fatalities,
       serious_injuries_pedestrian AS serious_injuries, minor_injuries_pedestrian AS minor_injuries
FROM {severity}
UNION ALL
SELECT year, reporting_period, 'Bicyclist',
       bicycle_collisions, fatality_bicyclist, serious_injuries_bicyclist, minor_injuries_bicyclist
FROM {severity}
UNION ALL
SELECT year, reporting_period, 'Motorcyclist',
       motorcycle_collisions, fatality_motorcyclist, serious_injury_motorcyclist, minor_injuries_motorcyclist
FROM {severity};
