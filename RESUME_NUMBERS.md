# Resume Numbers (verified)

Every number below was produced by the pipeline on the City of Edmonton snapshots ingested 2026-10-02 and re-checked
against the data. If a refresh changes the data, re-run the pipeline and update these numbers.

## Data
| Metric | Value | Source |
|---|---|---|
| Datasets ingested | 3 (City of Edmonton Annual Collision Report tables) | `src/traffic_safety/config.py` |
| Rows ingested | **19,120** (18,545 + 561 + 14) | `reports/data_quality.md` |
| Final row count | **19,120** | `reports/data_quality.md` |
| Percent retained | **100.00%** | `reports/data_quality.md` |
| Exact duplicates removed | **0** | `reports/data_quality.md` |
| Invalid rows dropped | **0** | `reports/data_quality.md` |
| Rows flagged (kept, with reason) | **796** (795 hour-24 bucket + 1 invalid hour) | `reports/data_quality.md` |
| Validation and reconciliation checks | **9 of 9 passing** (Temporal totals match the City's published yearly totals exactly, by year and severity) | `reports/data_quality.md` |
| Location names normalized | 313 raw names → **310** distinct locations (12 rows corrected) | `reports/data_quality.md` |
| Date range | **2010–2023** yearly totals; **2019–2023** time-of-day and location detail | `v_kpi_annual_summary`, `v_kpi_monthly` |

## Build
| Metric | Value |
|---|---|
| KPI SQL views | **12** (portable across DuckDB and BigQuery) |
| Tableau-ready extracts | **12** CSVs (one per view) |
| Dashboards | **1** interactive Plotly dashboard (9 charts + 4 KPI tiles); Tableau Public build guide (Tableau dashboard not yet published) |
| Named analysis queries | **12** in `sql/analysis/findings.sql` |
| pytest tests | **90, all passing**, offline, about 2 s |
| CI | GitHub Actions on every push, Python 3.11 / 3.12 / 3.13 |
| Refresh schedule | **Weekly**, Mondays 13:00 UTC (07:00 Edmonton), plus on demand |

## Top findings
1. **Late-night collisions are 3.7× more likely to be fatal or serious**: 40.5 vs 10.9 per 1,000 collisions
   (00:00–05:00 vs 07:00–18:00, 2019–2023; 95% CI for the ratio 3.10–4.49). These hours have 3.3% of collisions but 20 of 72 fatal collisions.
2. **28.0% of collisions happen 3–6 p.m.** (26,680 collisions), between 27.4% and 28.7% in every year 2019–2023.
3. **26 locations (23 intersections) were on the City's top-collision list all 5 years**, totalling 5,503 collisions;
   107 Ave & 142 St NW recorded **169 collisions in 2023**, the highest single-site count in the data.

## Suggested resume bullets (each number above is verified)
- Built an automated Python/SQL pipeline that ingests City of Edmonton open collision data (19,120 rows, 3 datasets) through the
  Socrata API, with pagination, retries, row-count verification and immutable raw snapshots; refreshes weekly through GitHub Actions.
- Designed data-quality rules and reconciliation checks (9/9 passing; 100% of rows retained, 796 flagged), and documented a 2022
  reporting-process break that would otherwise read as a 5× jump in minor-injury collisions.
- Wrote 12 KPI views in portable SQL (DuckDB and Google BigQuery) modelled on the City's Annual Collision Report, feeding a Plotly
  dashboard and 12 Tableau-ready extracts.
- Found that late-night collisions are 3.7× more likely to be fatal or serious, and that 26 locations stayed on the high-collision list
  for five straight years; summarized both for a non-technical audience with recommendations.
- Covered ingestion, every cleaning rule and every KPI with 90 offline pytest tests; the tests caught 4 real bugs, including a crash on new source labels.
