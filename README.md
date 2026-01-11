# System Usage & Reliability Analytics

Analyzing synthetic system log data to find traffic patterns, performance issues, and error trends.

**Note:** All data is synthetically generated - no real user data.

## What This Project Does

I built this to simulate what monitoring a real production system looks like. Take raw logs, clean them up, and pull out insights.

**Questions I wanted to answer:**

When do we get the most traffic?
How does performance change under load?
Which endpoints are slow?
When do errors spike?

## Project Structure

```
system-analytics/
├── data/
│   ├── raw/system_logs.csv
│   └── processed/clean_logs.csv
├── scripts/
│   ├── generate_data.py
│   ├── clean_data.py
│   ├── analysis.py
│   └── create_dashboard.py
├── visualizations/
│   ├── dashboard.html        # Interactive dashboard
│   ├── dashboard.png
│   ├── *.png                 # Static charts
│   └── tables/
├── README.md
└── requirements.txt
```

## Setup

```bash
cd system-analytics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running It

```bash
python scripts/generate_data.py
python scripts/clean_data.py
python scripts/analysis.py
python scripts/create_dashboard.py

# Open the interactive dashboard
open visualizations/dashboard.html
```

## Data Schema

Each row = one API request

| Column           | Type     | What it is                  |
|------------------|----------|-----------------------------|
| timestamp        | datetime | When it happened            |
| service          | string   | auth, events, or payments   |
| endpoint         | string   | /login, /checkout, etc      |
| response_time_ms | int      | Latency                     |
| status_code      | int      | 200, 400, 500, etc          |
| error            | bool     | Failed or not               |

## How I Generated the Data

Wanted it to feel real, so I added some patterns:

Evening rush (6-10 PM) gets 2.5x traffic
Weekends are quieter
Late night is basically dead
More traffic means slower responses and more errors
Payment endpoints are slower than auth (makes sense for real systems)
Added a degradation event on Jan 5th to make things interesting
Threw in some bad data for the cleaning script to handle

## Results

85k requests over 10 days:

**Traffic:** Peak hours have ~2.8x more requests. Auth handles most of it.

**Performance:** 
Avg 153ms, P95 380ms
Slowest endpoint is /checkout at 400ms
Peak hours are ~50% slower

**Errors:** 4.5% overall. Payments has highest rate. Errors go up with traffic.

**Takeaway:** Scale up during evenings, look into payments service.

## Tech

Python
Pandas
NumPy
Matplotlib/Seaborn
Plotly for interactive dashboard
