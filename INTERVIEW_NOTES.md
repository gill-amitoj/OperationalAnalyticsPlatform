# Interview Notes

## The 3 hardest data-quality problems

### 1. A reporting-process change that looks like a safety trend
**What I saw:** minor-injury collisions went from 1,444 (2021) to 3,283 (2022) to 7,581 (2023), 8% to 38% of all collisions, while
property-damage-only (PDO) collisions fell from 16,565 to 12,133. Injured people in the Fatalities & Injuries table jumped from 2,091 to 11,039.

**Why it's a trap:** read naively, injuries quintupled. The dataset description explains it: in September 2022 Collision Reporting
Centres replaced police reports, and the City notes the new reports "differ in consistency and accuracy."

**What I did:**
- Added a `reporting_period` column (Pre-CRC / Transition / CRC) to every table.
- Treated Minor vs PDO as **not comparable across the break**.
- Anchored trend claims on fatal and serious collisions and fatalities, which are least sensitive to the change.
- Put a severity-mix chart on the dashboard so the break is visible rather than hidden.
- Recommended in the brief that the City keep one severity definition.

**Talking point:** "The most important cleaning decision wasn't a line of code. It was knowing which comparisons the data can't support."

### 2. A severity label that changed name, proven equivalent by reconciliation
**What I saw:** 2019 uses `Serious`; 2020–2023 use `Major`. They might be the same category, or not.

**What I did:** I didn't assume. I reconciled against a second, independent table: Temporal "Major" totals equal the Severity table's
`serious_injury_collisions` **exactly** in every year (192, 230, 301, 304), and 2019's "Serious" (244) matches too. Only then did I map both to `Serious`.

I made reconciliation a permanent check. The Temporal totals match the City's published yearly totals exactly for every year and every severity class
(5 checks), plus 4 internal-consistency checks on the Severity table. All 9 pass. The checks also surfaced gaps in the City's own breakdowns
(1 unattributed fatality in 2023 by road user; 1 per year by location type in 5 years). Those are reported, not "fixed."

### 3. Tracking locations across years when names, labels and rankings all change
**What I saw:**
- Location types changed in 2022 (`Midblock` became `MID AVENUE` / `MID STREET` / `SOUTH OF INTERSECTION`).
- There are typos (`STREEET`, `BETWEN`, `ANTONY HENDAY`).
- Intersections appear in either order ("Yellowhead Trail & Fort Road" vs "Fort Road AND Yellowhead Trail").
- The ranking method changes by year (unique ranks in 2019; tied ranks later).
- Each list is truncated at about the top 50.

**What I did:**
- Built an explicit, reviewable correction table instead of fuzzy matching, which can silently merge different places.
- Expanded `ST`→`STREET` but left `ST.` (Saint, as in St. Albert Trail) alone.
- Made intersection keys order-independent and recomputed ranks with one method.
- Result: 313 raw names → 310 locations.
- Deliberately did **not** "fix" `GATEWAY BOULEVARD NE` to NW. It's probably a typo, but correcting it would be an inference.

**Key modelling decision:** a site missing from a year's list is *unknown* (below the cutoff), not zero. So the hotspot KPI averages over
**listed years only**. Treating missing as zero would make chronic hotspots look like they were improving.

**Honourable mentions:**
- Hour code 24 holds 2,777 collisions vs 1,605 in the hour before, consistent with unknown times recorded as midnight. I flagged it and excluded it from night findings.
- The API once returned a header-only page, which would have silently truncated a download. Ingestion now verifies the row count against `count(*)` and retries.

## Why each KPI is defined the way it is

| KPI | Reasoning |
|---|---|
| Fatal + serious collisions, and KSI per 100k | Vision Zero counts people killed or seriously injured, not collisions. Per-capita rates matter because the population grew 42% from 2010 to 2023 (793,000 → 1,128,811). |
| Year-over-year % | Mirrors the City's annual report; shown next to the reporting-period flag so 2022–23 changes aren't over-read. |
| Monthly outlier (leave-one-out baseline, >1.5×) | Comparing a month with its own average inflates the baseline; leave-one-out compares Feb 2019 with *other* Februaries. The 1.5× threshold is a simple, explainable cut. Only 2 of 60 months exceed it. |
| Severity per 1,000 collisions by hour | Counts alone point at rush hour, which is where *volume* is. A rate shows where *severity* is, and those are different answers. |
| Collisions per clock hour by period | The City's "Evening" period is 12 hours and the others are 3, so raw totals mislead. A test caught my first version dividing by hours that had data instead of the period length. |
| Day-of-week average over all years | Dividing by the years a weekday appears in would inflate sparse days (another test catch). |
| Hotspot persistence | Years listed and average per *listed* year (see problem 3). Persistence is a stronger case for engineering review than one bad year. |
| Vulnerable road users (people) | Pedestrians, cyclists and motorcyclists carry most of the severe outcomes, and Vision Zero programs target them. |
| Intersection vs midblock | About 71–78% of injuries happen at intersections every year, which tells you where engineering effort applies. |

**Skipped because the data can't support them:**
- Daily trends: no day-of-month field.
- Maps and clustering: no coordinates.
- Rates per traffic volume: counts don't join to collision sites.
- Severity by location: not published.
- Weather effects: not in the data.

## Limitations of the data
- **Aggregated only.** No individual collisions, so no causes, exact locations, or multivariate modelling.
- **Reporting break in Sept 2022** affects severity classification; minor and PDO numbers are not comparable across it.
- **No exposure measure.** Hotspots reflect traffic volume as much as risk; a busy intersection can top the list while being relatively safe per vehicle.
- **Truncated top lists** (about 50 per year per group).
- **2020** is a pandemic year and a poor baseline.
- **Data ends at 2023**, and the tables are updated manually once a year, so the weekly refresh mostly confirms nothing changed.
- **The night-severity finding is descriptive.** The rate ratio is 3.73 (95% CI 3.10–4.49), but the data can't say *why* (speed, impairment, lighting, fewer but faster vehicles).

## What I would build next
1. **Collision-level data.** Request it from the City or Edmonton Police Service. With dates and coordinates I could do spatial hotspot detection
   (kernel density or Getis-Ord), cause analysis, and before/after evaluation of interventions.
2. **Exposure-adjusted risk.** Join the City's Average Annual Weekday Traffic Volumes (2011–2022) to rank intersections by
   collisions per million entering vehicles. The blocker is matching location names; I'd build a geocoded crosswalk.
3. **More ACR tables.** Fatalities & Injuries (age, road user, traffic control) and Collision Cause would support pedestrian and
   left-turn analyses.
4. **Weather.** Join Environment Canada daily data to test the February 2019 spike (3,721 collisions, 2.2× baseline, but *lower* severity:
   4.8 vs 7.2 fatal/serious per 1,000), which looks like a winter-conditions event.
5. **Production hardening.** dbt models for the SQL layer, schema-drift alerts when the City changes column names or labels,
   Workload Identity Federation instead of a service-account key, and GitHub Pages hosting for the dashboard.
6. **Publish the Tableau Public version** and add a "what changed since last refresh" panel.
