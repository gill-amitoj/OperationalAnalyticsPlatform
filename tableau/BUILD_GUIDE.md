# Tableau Public Build Guide

How to rebuild the Edmonton Traffic Safety dashboard in **Tableau Public** from the CSV extracts in this folder.
The extracts are produced by `python -m traffic_safety.pipeline` from the SQL views in `sql/views/`, so the Tableau
numbers match the Plotly dashboard and the SQL exactly.

> Tableau Public works with file-based data. It cannot keep a live connection to BigQuery, so the workflow is:
> pipeline refreshes the CSVs, then you re-open the workbook and republish.

---

## 1. Setup (about 10 minutes)

1. Install **Tableau Public** (free desktop app) from <https://public.tableau.com/app/discover> and sign in with a Tableau Public account.
2. Pull the latest repo so `tableau/*.csv` is current (the weekly GitHub Action refreshes them).
3. Open Tableau Public, then **Connect → To a File → Text file**, and select `tableau/kpi_annual_summary.csv`.
4. Add each further extract as its **own data source** (Data menu → New Data Source → Text file). Don't join them;
   each CSV is already at the grain its chart needs.
5. For each data source, check the field types in the Data Source tab:
   - `year`, `month`, `hour_of_day`, `day_of_week_num`, `rank`: **Number (whole)**. Drag `year` to Dimensions
     (right-click, then Convert to Dimension), so it isn't summed.
   - `month_start` in `kpi_monthly.csv`: **Date**.
   - `is_*` fields: **Boolean**.

## 2. Which extract feeds which chart

| # | Sheet name | Extract | Chart type | Columns / Rows / Marks |
|---|---|---|---|---|
| 1 | KPI tiles | `kpi_annual_summary.csv` | Text (BANs) | Filter `year` = 2023. Text: `total_collisions`, `fatal_serious_collisions`, `total_fatalities`, `collisions_yoy_pct` |
| 2 | Long-run trend | `kpi_annual_summary.csv` | Line | Columns: `year` (continuous). Rows: `fatal_serious_collisions_per_100k`. Duplicate the sheet for `collisions_per_100k`. **Do not** put both on a dual axis; use two sheets |
| 3 | Monthly trend + outliers | `kpi_monthly.csv` | Line + circle | Columns: `month_start` (continuous month). Rows: `collisions`. Then a second mark layer with `collisions` and Color = `is_outlier_month` (True = orange, False = transparent). Tooltip: `ratio_to_other_years`, `same_month_other_years_avg` |
| 4 | Outlier drill-down | `kpi_month_hour_drilldown.csv` | Bar + line | Columns: `hour_of_day`. Rows: `collisions` (bars) and `same_month_other_years_avg` (line) on a **synchronized** axis. Filters: `year`, `month_name` (single-value dropdown) |
| 5 | Day × hour heatmap | `kpi_day_hour.csv` | Highlight table / heatmap | Columns: `hour_of_day` (discrete). Rows: `day_of_week` sorted by `day_of_week_num`. Color: SUM(`collisions`), sequential blue. Filter: `year` (multi-select, show as dropdown) |
| 6 | Severity by hour | `kpi_hourly.csv` | Bar | Columns: `hour_of_day` (discrete). Rows: `fatal_serious_per_1000`. Color: calculated field `Night` (see below) |
| 7 | Collisions per hour by period | `kpi_hour_group.csv` | Bar | Columns: `hour_group` (manual sort: AM Peak, AM Non-Peak, PM Non-Peak, PM Peak, Evening). Rows: `collisions_per_hour` |
| 8 | Day of week | `kpi_day_of_week.csv` | Bar | Columns: `day_of_week` sorted by `day_of_week_num`. Rows: `avg_collisions_per_year`. Tooltip: `fatal_serious_per_1000` |
| 9 | Chronic hotspots | `kpi_location_persistence.csv` | Horizontal bar | Filter `years_listed` = 5. Rows: `location_name` sorted by `avg_collisions_per_listed_year` (descending). Columns: `avg_collisions_per_listed_year` |
| 10 | Top locations by year | `kpi_top_locations.csv` | Horizontal bar | Filters: `year` (single-value dropdown), `location_group`, `rank` ≤ 10. Rows: `location_name`. Columns: `collision_count` |
| 11 | Vulnerable road users | `kpi_vulnerable_road_users.csv` | Line (3 series) | Columns: `year`. Rows: calculated `KSI` = `[fatalities] + [serious_injuries]`. Color: `road_user`. Label: line ends only |
| 12 | Severity mix | `kpi_severity_trend.csv` | 100% stacked bar | Columns: `year` (discrete). Rows: `pct_of_year`. Color: `severity`, sorted by `severity_rank` |
| 13 | Intersection vs midblock | `kpi_intersection_midblock.csv` | Line or bar | Columns: `year`. Rows: `pct_injuries_at_intersections` |

### Calculated fields

```text
// Night (sheet 6): highlights the 00:00–05:00 hours
IF [hour_of_day] <= 4 THEN "00:00–05:00" ELSE "Other hours" END

// KSI (sheet 11): people killed or seriously injured
[fatalities] + [serious_injuries]

// Reporting period reference line (sheets 2, 3, 11)
// Add a reference line at year 2022 (or month_start = 2022-09-01) labelled
// "Sep 2022: new reporting process (CRC)"
```

## 3. Colours (match the Plotly dashboard)

| Use | Hex |
|---|---|
| Main measure (single series) | `#2a78d6` |
| Highlight / outlier | `#eb6834` |
| Context (de-emphasized bars) | `#b5b3ab` |
| Road users: Pedestrian / Motorcyclist / Bicyclist | `#2a78d6` / `#eb6834` / `#1baf7a` |
| Severity (ordered light to dark): PDO / Minor / Serious / Fatal | `#b7d3f6` / `#6da7ec` / `#256abf` / `#0d366b` |
| Heatmap | Tableau's built-in **Blue** sequential palette |

Keep gridlines light, remove borders, and avoid dual axes (sheet 4 uses one synchronized axis, which is fine
because both measures are collisions per hour).

## 4. Dashboard layout (1200 × 1600, "Automatic" for phone)

1. **Header**: title "Edmonton Traffic Safety: When, where and how severely people are hurt" and a one-line description.
2. **Row 1**: KPI tiles (sheet 1), four across.
3. **Row 2**: long-run trend (sheet 2, both versions side by side).
4. **Row 3**: monthly trend (sheet 3). Add a **dashboard action** (Filter, on Select) from sheet 3 to sheet 4 on `year` + `month_name`,
   so clicking an outlier month drills into its hourly profile.
5. **Row 4**: day × hour heatmap (sheet 5) with its year filter.
6. **Row 5**: severity by hour (sheet 6) beside collisions per period (sheet 7).
7. **Row 6**: chronic hotspots (sheet 9) beside top locations by year (sheet 10).
8. **Row 7**: vulnerable road users (sheet 11) beside severity mix (sheet 12).
9. **Footer text box** (required by the licence; copy exactly):

   > Contains information licensed under the Open Government Licence – City of Edmonton.
   > Source: City of Edmonton Open Data, Annual Collision Report tables (data.edmonton.ca). Independent analysis;
   > not produced or endorsed by the City of Edmonton. The Sept 2022 change to Collision Reporting Centres affects
   > severity classification; data is aggregated (no individual collisions or coordinates).

Don't use the City of Edmonton logo or crest; the licence does not permit it.

## 5. Publish

1. **File → Save to Tableau Public As…**, named "Edmonton Traffic Safety (Vision Zero analytics)".
2. On your Tableau Public profile, open the viz → **Edit Details**: add the GitHub repo link and the attribution line.
3. Copy the share link into the repo README (section "Dashboards").

## 6. Refresh after new data

1. `python -m traffic_safety.pipeline` (or pull the weekly Action's commit).
2. In Tableau Public: **Data → Refresh All Extracts** (or re-open the workbook), check the KPI tiles' year filter, and save to Tableau Public again.
