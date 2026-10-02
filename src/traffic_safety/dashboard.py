"""
dashboard.py
============
Builds dashboard/index.html: a single-page Plotly dashboard driven entirely by
the SQL KPI views (DuckDB). Every number on the page comes from a view.

Design notes: one hue per job (blue = the measure, gray = context, orange =
emphasis/outlier), no dual axes, recessive grid, legends for multi-series
charts, light/dark chrome that follows the viewer's OS setting.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import ATTRIBUTION, DASHBOARD_DIR, DATASETS, LICENCE_URL
from .load import load_duckdb, query_views

log = logging.getLogger(__name__)

# Reference palette (validated: first three slots pass all-pairs CVD checks on light surface).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#b5b3ab"
# INK_2 is a mid-gray legible on both the light and dark surface (Plotly text cannot follow the CSS theme).
INK, INK_2, MUTED = "#0b0b0b", "#75736e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEVERITY_RAMP = {"PDO": "#b7d3f6", "Minor": "#6da7ec", "Serious": "#256abf", "Fatal": "#0d366b"}
DAY_ORDER = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def base_layout(fig: go.Figure, title: str, subtitle: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{INK_2}'>{subtitle}</span>"
                                             if subtitle else ""), x=0, xanchor="left", font=dict(size=15)),
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=56, r=24, t=80 if subtitle else 52, b=72),
        height=height,
        hoverlabel=dict(font=dict(family=FONT)),
        legend=dict(orientation="h", y=-0.14, yanchor="top", x=0, xanchor="left", bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False, rangemode="tozero")
    return fig


def display_name(name: str) -> str:
    """Title-case a location name but keep quadrants (NW/SW/NE/SE) upper-case."""
    return " ".join(w if w in {"NW", "SW", "NE", "SE"} else w.title() for w in name.split())


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_annual(v: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Fatal + serious-injury collisions per 100k residents", "All collisions per 100k residents"))
    for col, y in [(1, "fatal_serious_collisions_per_100k"), (2, "collisions_per_100k")]:
        fig.add_trace(go.Scatter(
            x=v.year, y=v[y], mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
            customdata=v[["fatal_serious_collisions", "total_collisions", "population"]],
            hovertemplate="<b>%{x}</b><br>%{y:,.1f} per 100k<br>Fatal+serious: %{customdata[0]:,}"
                          "<br>All collisions: %{customdata[1]:,}<br>Population: %{customdata[2]:,}<extra></extra>",
            showlegend=False), row=1, col=col)
        fig.add_vline(x=2022.67, line_width=1, line_color=MUTED, row=1, col=col)
    fig.add_annotation(x=2022.67, y=1, xref="x", yref="paper", text="CRC reporting change", showarrow=False,
                       xanchor="right", yanchor="top", font=dict(size=11, color=INK_2))
    base_layout(fig, "Long-run trend, 2010–2023",
                "Source: Annual Collision Report — Collision Severity. Rates use the City's population for each year.")
    fig.update_annotations(font=dict(size=12, color=INK_2))
    fig.update_layout(margin=dict(t=110))
    return fig


def chart_monthly(v: pd.DataFrame) -> go.Figure:
    v = v.sort_values("month_start")
    fig = go.Figure(go.Scatter(
        x=v.month_start, y=v.collisions, mode="lines", line=dict(color=BLUE, width=2), name="Collisions",
        customdata=v[["same_month_other_years_avg", "ratio_to_other_years"]],
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} collisions<br>Same month, other years: %{customdata[0]:,.0f}"
                      "<br>Ratio: %{customdata[1]:.2f}×<extra></extra>"))
    out = v[v.is_outlier_month]
    fig.add_trace(go.Scatter(
        x=out.month_start, y=out.collisions, mode="markers+text", name="Outlier month (>1.5× baseline)",
        marker=dict(color=ORANGE, size=10, line=dict(color=SURFACE, width=2)),
        text=[f"{d:%b %Y}: {r:.1f}×" for d, r in zip(pd.to_datetime(out.month_start), out.ratio_to_other_years)],
        textposition="top center", textfont=dict(color=INK_2, size=11), hoverinfo="skip"))
    fig.add_vline(x=pd.Timestamp("2022-09-01").timestamp() * 1000, line_width=1, line_color=MUTED)
    fig.add_annotation(x=pd.Timestamp("2022-09-01"), y=1, yref="paper", text="CRC reporting change",
                       showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color=INK_2))
    return base_layout(fig, "Collisions per month, 2019–2023",
                       "Baseline = average of the same calendar month in all other years. "
                       "Outliers: more than 1.5× baseline.")


def chart_drilldown(v: pd.DataFrame, monthly: pd.DataFrame, n: int = 6) -> go.Figure:
    picks = monthly.sort_values("ratio_to_other_years", ascending=False).head(n)
    fig = go.Figure()
    buttons = []
    for i, (_, m) in enumerate(picks.iterrows()):
        d = (v[(v.year == m.year) & (v.month == m.month)].set_index("hour_of_day")
               .reindex(range(24)).reset_index())
        d["collisions"] = d["collisions"].fillna(0)
        base = v[v.month == m.month].groupby("hour_of_day").same_month_other_years_avg.max().reindex(range(24))
        label = f"{m.month_name.title()} {m.year} ({m.ratio_to_other_years:.2f}×)"
        vis = i == 0
        fig.add_trace(go.Bar(x=d.hour_of_day, y=d.collisions, marker_color=BLUE, name="This month", visible=vis,
                             hovertemplate="%{x}:00–%{x}:59 · %{y:,} collisions<extra></extra>"))
        fig.add_trace(go.Scatter(x=base.index, y=base.values, mode="lines", line=dict(color=INK_2, width=2),
                                 name="Same month, other years (avg)", visible=vis,
                                 hovertemplate="Baseline %{y:,.1f}<extra></extra>"))
        mask = [False] * (2 * len(picks))
        mask[2 * i] = mask[2 * i + 1] = True
        buttons.append(dict(label=label, method="update", args=[{"visible": mask}]))
    base_layout(fig, "Drill-down: what happened in the outlier months?",
                "Hourly collisions in the selected month vs the same month's average in other years. "
                "Pick a month from the menu.")
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.0, yanchor="bottom",
                                        bgcolor=SURFACE, bordercolor=AXIS, font=dict(size=12))],
                      margin=dict(t=110))
    fig.update_xaxes(title_text="Hour of day", dtick=2)
    return fig


def chart_heatmap(v: pd.DataFrame) -> go.Figure:
    def matrix(d):
        p = d.pivot_table(index="day_of_week", columns="hour_of_day", values="collisions", aggfunc="sum")
        return p.reindex(index=DAY_ORDER, columns=range(24)).fillna(0)

    years = ["All years"] + sorted(v.year.unique().tolist())
    fig = go.Figure()
    for i, y in enumerate(years):
        m = matrix(v if y == "All years" else v[v.year == y])
        fig.add_trace(go.Heatmap(
            z=m.values, x=[f"{h:02d}" for h in m.columns], y=[d.title() for d in m.index],
            colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
            xgap=2, ygap=2, visible=i == 0, colorbar=dict(title=dict(text="Collisions", side="right"),
                                                          outlinewidth=0, thickness=12),
            hovertemplate="%{y} %{x}:00 · %{z:,} collisions<extra></extra>"))
    buttons = [dict(label=str(y), method="update",
                    args=[{"visible": [j == i for j in range(len(years))]}]) for i, y in enumerate(years)]
    base_layout(fig, "When collisions happen: day of week × hour of day",
                "Hour 23 includes the source's hour-24 bucket, which likely contains default-midnight times.", 400)
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.0, yanchor="bottom",
                                        bgcolor=SURFACE, bordercolor=AXIS)], margin=dict(t=110))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Hour of day")
    return fig


def chart_severity_by_hour(v: pd.DataFrame) -> go.Figure:
    v = v.sort_values("hour_of_day")
    night = v.hour_of_day.between(0, 4)
    fig = go.Figure(go.Bar(
        x=v.hour_of_day, y=v.fatal_serious_per_1000,
        marker=dict(color=[BLUE if n else GRAY for n in night], cornerradius=4),
        customdata=v[["collisions", "fatal_serious_collisions"]],
        hovertemplate="<b>%{x}:00–%{x}:59</b><br>%{y:.1f} fatal/serious per 1,000 collisions"
                      "<br>%{customdata[1]:,} of %{customdata[0]:,} collisions<extra></extra>"))
    base_layout(fig, "How severe are collisions at each hour?",
                "Fatal + serious-injury collisions per 1,000 collisions, 2019–2023. Blue = 00:00–05:00.")
    fig.update_xaxes(title_text="Hour of day", dtick=2)
    return fig


def chart_hour_group(v: pd.DataFrame) -> go.Figure:
    order = ["AM Peak", "AM Non-Peak", "PM Non-Peak", "PM Peak", "Evening"]
    v = v.set_index("hour_group").reindex(order).reset_index()
    fig = go.Figure(go.Bar(
        x=v.hour_group, y=v.collisions_per_hour, marker=dict(color=BLUE, cornerradius=4),
        customdata=v[["collisions", "hours_in_period"]],
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} collisions per clock hour"
                      "<br>%{customdata[0]:,} total over %{customdata[1]} hours<extra></extra>"))
    base_layout(fig, "Collisions per hour, by City time period",
                "Normalized by period length (Evening spans 12 hours; the others 3). 2019–2023 total.")
    return fig


def chart_locations(persist: pd.DataFrame, top: pd.DataFrame) -> go.Figure:
    chronic = (persist[persist.years_listed == persist.years_listed.max()]
               .sort_values("avg_collisions_per_listed_year").tail(15))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=chronic.location_name.map(display_name), x=chronic.avg_collisions_per_listed_year, orientation="h",
        marker=dict(color=BLUE, cornerradius=4), visible=True,
        customdata=chronic[["location_group", "best_rank", "years_in_top_10"]],
        hovertemplate="<b>%{y}</b><br>%{x:.1f} collisions/yr on average<br>%{customdata[0]} · best rank "
                      "%{customdata[1]} · top-10 in %{customdata[2]} of 5 years<extra></extra>"))
    years = sorted(top.year.unique())
    for y in years:
        d = top[(top.year == y) & (top.location_group == "Intersection") & (top["rank"] <= 10)]
        d = d.sort_values(["collision_count", "location_name"])
        fig.add_trace(go.Bar(y=d.location_name.map(display_name), x=d.collision_count, orientation="h",
                             marker=dict(color=BLUE, cornerradius=4), visible=False,
                             customdata=d[["rank"]],
                             hovertemplate="<b>%{y}</b><br>%{x} collisions · rank %{customdata[0]}<extra></extra>"))
    n = 1 + len(years)
    buttons = [dict(label="Chronic hotspots (listed all 5 years)", method="update",
                    args=[{"visible": [i == 0 for i in range(n)]}])]
    buttons += [dict(label=f"Top-10 intersections {y}", method="update",
                     args=[{"visible": [i == k + 1 for i in range(n)]}]) for k, y in enumerate(years)]
    base_layout(fig, "Highest-collision locations",
                "Chronic = on the City's top list in every year 2019–2023; average uses listed years only.", 520)
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.0, yanchor="bottom",
                                        bgcolor=SURFACE, bordercolor=AXIS)],
                      margin=dict(l=330, t=110))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK_2, size=11))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title_text="Collisions")
    return fig


def chart_vru(v: pd.DataFrame) -> go.Figure:
    v = v.assign(ksi=v.fatalities + v.serious_injuries).sort_values("year")
    fig = go.Figure()
    for color, user in [(BLUE, "Pedestrian"), (ORANGE, "Motorcyclist"), (AQUA, "Bicyclist")]:
        d = v[v.road_user == user]
        fig.add_trace(go.Scatter(
            x=d.year, y=d.ksi, mode="lines", name=user, line=dict(color=color, width=2),
            customdata=d[["fatalities", "serious_injuries"]],
            hovertemplate=f"<b>{user}s, %{{x}}</b><br>%{{y}} killed or seriously injured"
                          "<br>(%{customdata[0]} killed, %{customdata[1]} seriously injured)<extra></extra>"))
        last = d.iloc[-1]
        fig.add_annotation(x=last.year, y=last.ksi, text=f"{user}s {int(last.ksi)}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=INK_2, size=11))
    fig.add_vline(x=2022.67, line_width=1, line_color=MUTED)
    base_layout(fig, "Vulnerable road users killed or seriously injured",
                "People, not collisions, 2010–2023. Vertical line = Sep 2022 reporting change.")
    fig.update_layout(margin=dict(r=120))
    fig.update_xaxes(range=[2009.6, 2023.4], dtick=2)
    return fig


def chart_severity_mix(v: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for sev in ["PDO", "Minor", "Serious", "Fatal"]:
        d = v[v.severity == sev].sort_values("year")
        fig.add_trace(go.Bar(x=d.year.astype(str), y=d.pct_of_year, name=sev, marker_color=SEVERITY_RAMP[sev],
                             customdata=d[["collisions"]],
                             hovertemplate=f"<b>{sev} %{{x}}</b><br>%{{y:.1f}}% · %{{customdata[0]:,}} collisions"
                                           "<extra></extra>"))
    base_layout(fig, "Severity mix by year: the reporting change is visible",
                "Share of each year's collisions. Minor vs PDO is not comparable<br>"
                "across the Sep 2022 switch to Collision Reporting Centres.")
    fig.update_layout(margin=dict(t=96))
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_traces(marker_line=dict(color=SURFACE, width=2))
    fig.update_yaxes(ticksuffix="%", range=[0, 100])
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def kpi_tiles(views: dict[str, pd.DataFrame]) -> list[tuple[str, str, str]]:
    a = views["v_kpi_annual_summary"].set_index("year")
    last = int(a.index.max())
    hourly = views["v_kpi_hourly"]
    night = hourly[hourly.hour_of_day.between(0, 4)]
    day = hourly[hourly.hour_of_day.between(7, 17)]
    night_rate = 1000 * night.fatal_serious_collisions.sum() / night.collisions.sum()
    day_rate = 1000 * day.fatal_serious_collisions.sum() / day.collisions.sum()
    persist = views["v_kpi_location_persistence"]
    return [
        (f"{int(a.loc[last, 'total_collisions']):,}", f"Collisions in {last}",
         f"{a.loc[last, 'collisions_yoy_pct']:+.1f}% vs {last - 1}"),
        (f"{int(a.loc[last, 'fatal_serious_collisions']):,}", f"Fatal + serious-injury collisions, {last}",
         f"{int(a.loc[last, 'total_fatalities'])} people killed"),
        (f"{night_rate / day_rate:.1f}×", "Severity at night vs daytime",
         f"{night_rate:.1f} vs {day_rate:.1f} fatal/serious per 1,000 (00–05h vs 07–18h)"),
        (f"{int((persist.years_listed == 5).sum())}", "Chronic hotspots",
         "Locations on the City's top list all 5 years"),
    ]


PAGE_CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --border:rgba(11,11,11,0.10); color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --border:rgba(255,255,255,0.10); color-scheme: dark; } }
:root[data-theme="dark"] { --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
  --border:rgba(255,255,255,0.10); color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; }
.sub { color: var(--ink2); margin: 0 0 20px; line-height: 1.5; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile, .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
.tile { padding: 16px; }
.tile .v { font-size: 30px; font-weight: 600; }
.tile .l { font-size: 13px; color: var(--ink2); margin-top: 2px; }
.tile .d { font-size: 12px; color: var(--muted); margin-top: 6px; }
.grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.card { padding: 8px; margin-bottom: 12px; overflow: hidden; min-width: 0; }
.full { grid-column: 1 / -1; }
@media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
footer { color: var(--muted); font-size: 12px; line-height: 1.6; margin-top: 16px; }
footer a { color: var(--ink2); }
"""

# Plotly figures carry their own colors; swap chart chrome when the page is dark.
THEME_JS = """
(function(){
  const dark = {paper_bgcolor:'#1a1a19', plot_bgcolor:'#1a1a19', 'font.color':'#ffffff'};
  const light = {paper_bgcolor:'#fcfcfb', plot_bgcolor:'#fcfcfb', 'font.color':'#0b0b0b'};
  function isDark(){
    const t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function apply(){
    const d = isDark();
    document.querySelectorAll('.js-plotly-plot').forEach(function(el){
      const upd = Object.assign({}, d ? dark : light);
      Object.keys(el.layout || {}).forEach(function(k){
        if (/^[xy]axis\\d*$/.test(k)) {
          upd[k + '.gridcolor'] = d ? '#2c2c2a' : '#e1e0d9';
          upd[k + '.linecolor'] = d ? '#383835' : '#c3c2b7';
        }
      });
      upd['updatemenus[0].bgcolor'] = d ? '#1a1a19' : '#fcfcfb';
      try { Plotly.relayout(el, upd); } catch(e) {}
    });
  }
  window.addEventListener('load', apply);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply);
})();
"""


def build(views: dict[str, pd.DataFrame], out_dir: Path = DASHBOARD_DIR) -> Path:
    figs = [
        ("full", chart_annual(views["v_kpi_annual_summary"])),
        ("full", chart_monthly(views["v_kpi_monthly"])),
        ("full", chart_drilldown(views["v_kpi_month_hour_drilldown"], views["v_kpi_monthly"])),
        ("full", chart_heatmap(views["v_kpi_day_hour"])),
        ("half", chart_severity_by_hour(views["v_kpi_hourly"])),
        ("half", chart_hour_group(views["v_kpi_hour_group"])),
        ("full", chart_locations(views["v_kpi_location_persistence"], views["v_kpi_top_locations"])),
        ("half", chart_vru(views["v_kpi_vulnerable_road_users"])),
        ("half", chart_severity_mix(views["v_kpi_severity_trend"])),
    ]
    cards = []
    for i, (size, fig) in enumerate(figs):
        div = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False,
                          config={"displaylogo": False, "responsive": True})
        cards.append(f'<div class="card {"full" if size == "full" else ""}">{div}</div>')
    tiles = "".join(f'<div class="tile"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div>'
                    f'<div class="d">{html.escape(d)}</div></div>' for v, l, d in kpi_tiles(views))
    sources = " · ".join(f'<a href="{d.portal_url}">{html.escape(d.title)}</a>' for d in DATASETS.values())
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edmonton Traffic Safety</title><style>{PAGE_CSS}</style></head>
<body><main>
<h1>Edmonton Traffic Safety Dashboard</h1>
<p class="sub">Independent analysis of City of Edmonton open collision data (Annual Collision Report tables), in
support of Vision Zero questions: when, where, and how severely people are hurt on Edmonton roads.
Data covers 2010–2023 (yearly totals) and 2019–2023 (time-of-day and locations).</p>
<div class="tiles">{tiles}</div>
<div class="grid">{''.join(cards)}</div>
<footer>
<p>{html.escape(ATTRIBUTION)} <a href="{LICENCE_URL}">Licence</a>. Sources: {sources}.
This is an independent project and is not produced or endorsed by the City of Edmonton.</p>
<p>Limitations: the source is aggregated (no individual collisions, dates, or coordinates). A reporting-process change in
Sept 2022 (Collision Reporting Centres) affects severity classification. Generated {generated}.</p>
</footer>
</main><script>{THEME_JS}</script></body></html>"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(page)
    return out


def main() -> Path:
    con = load_duckdb()
    out = build(query_views(con))
    log.info("Dashboard written to %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(main())
