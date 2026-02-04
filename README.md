# 📊 Operational System Analytics Platform

> **End-to-end data analytics pipeline** simulating enterprise-grade system monitoring — from synthetic log generation through SQL analysis, interactive dashboards, and Excel business review layers.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-orange.svg)](https://plotly.com)
[![Excel](https://img.shields.io/badge/Excel-Business%20Review-darkgreen.svg)](https://www.microsoft.com/excel)

---

## 🎯 Project Overview

This project demonstrates a **production-grade analytics workflow** used in enterprise environments like banking, fintech, and SaaS platforms. I built this to showcase my ability to:

- **Generate realistic data** with temporal patterns (peak hours, weekends, degradation events)
- **Clean and transform** messy data using industry best practices
- **Analyze performance** with SQL-style aggregations and statistical metrics
- **Visualize insights** via interactive Plotly dashboards
- **Support business review** with Excel exports featuring VLOOKUP, IF formulas, and pivot-ready structures

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OPERATIONAL SYSTEM ANALYTICS PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  GENERATE    │───▶│    CLEAN     │───▶│   ANALYZE    │───▶│ DASHBOARD │ │
│  │   DATA       │    │    DATA      │    │    (SQL)     │    │  (HTML)   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│        │                   │                   │                    │       │
│        ▼                   ▼                   ▼                    ▼       │
│   raw_logs.csv       clean_logs.csv      metrics/*.csv        dashboard    │
│   (85K+ records)     (validated)         (aggregated)         (interactive)│
│                                                │                            │
│                                                ▼                            │
│                                    ┌─────────────────────┐                  │
│                                    │   EXCEL BUSINESS    │                  │
│                                    │      REVIEW         │                  │
│                                    │  (VLOOKUP, IF,      │                  │
│                                    │   Pivot Tables)     │                  │
│                                    └─────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Key Results & Impact

| Metric | Value | Insight |
|--------|-------|---------|
| **Total Requests Analyzed** | 85,000+ | 10 days of simulated production traffic |
| **Peak vs Off-Peak Ratio** | 2.8x | Evening hours (6-10 PM) see nearly 3x traffic |
| **Avg Response Time** | 153ms | P95 at 380ms indicates tail latency issues |
| **Slowest Endpoint** | `/checkout` (400ms) | Payment service needs optimization |
| **Overall Error Rate** | 4.5% | Payments service has highest error rate |
| **Actionable Insight** | Scale up during evenings | Optimize payments service for reliability |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Business Intelligence** | Excel (xlsxwriter), VLOOKUP, Pivot Tables |
| **Dashboard** | Interactive HTML/CSS/JS |
| **Version Control** | Git, GitHub |

---

## 📁 Project Structure

```
OperationalSystemAnalytics/
├── 📂 data/
│   ├── raw/                    # Raw synthetic logs (85K+ records)
│   │   └── system_logs.csv
│   └── processed/              # Cleaned, validated data
│       └── clean_logs.csv
│
├── 📂 scripts/
│   ├── generate_data.py        # Synthetic data generation with realistic patterns
│   ├── clean_data.py           # Data validation & transformation
│   ├── analysis.py             # SQL-style aggregations & metrics
│   ├── create_dashboard.py     # Interactive Plotly dashboard
│   └── excel_business_review.py # Excel export with business formulas
│
├── 📂 visualizations/
│   ├── dashboard.html          # 🌐 Interactive dashboard (open in browser)
│   ├── business_review.xlsx    # 📊 Excel business review workbook
│   ├── tables/                 # Exported metrics tables (CSV)
│   └── *.png                   # Static chart images
│
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd OperationalSystemAnalytics
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
# Generate synthetic log data
python scripts/generate_data.py

# Clean and validate data
python scripts/clean_data.py

# Run analysis and generate visualizations
python scripts/analysis.py

# Create interactive dashboard
python scripts/create_dashboard.py

# Export to Excel for business review
python scripts/excel_business_review.py
```

### 3. View Results
```bash
# Open interactive dashboard
open visualizations/dashboard.html

# Open Excel business review
open visualizations/business_review.xlsx
```

---

## 📊 Data Schema

Each record represents one API request to the system:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Request timestamp |
| `service` | string | Service name: `auth`, `events`, `payments` |
| `endpoint` | string | API endpoint: `/login`, `/checkout`, etc. |
| `response_time_ms` | int | Response latency in milliseconds |
| `status_code` | int | HTTP status code (200, 400, 500, etc.) |
| `error` | bool | Whether request resulted in error |

### Derived Columns (after cleaning):
| Column | Type | Description |
|--------|------|-------------|
| `hour` | int | Hour of day (0-23) |
| `day_of_week` | string | Day name (Monday-Sunday) |
| `is_peak_hour` | bool | True if 6-10 PM |
| `is_weekend` | bool | True for Saturday/Sunday |

---

## 🎲 Realistic Data Generation

The synthetic data mimics real production systems:

| Pattern | Implementation | Why It Matters |
|---------|----------------|----------------|
| **Evening Rush** | 6-10 PM gets 2.5x traffic | Simulates user behavior patterns |
| **Weekend Slowdown** | 30% less traffic on weekends | Reflects business cycle |
| **Late Night Lull** | 2-6 AM at 30% capacity | Mirrors real usage patterns |
| **Load-Dependent Latency** | High traffic → slower responses | Realistic system behavior |
| **Service-Specific Performance** | Payments slower than auth | Domain-realistic modeling |
| **Degradation Event** | Jan 5th incident | Simulates real outage scenarios |
| **Bad Data Injection** | Invalid records for cleaning | Tests data quality pipeline |

---

## 📊 Excel Business Review Layer

The Excel integration follows enterprise banking practices:

| Sheet | Purpose | Formulas Used |
|-------|---------|---------------|
| **Metrics** | Endpoint performance data | Imported from SQL analysis |
| **ServiceLookup** | Service → Owner/Priority mapping | Reference table |
| **BusinessReview** | Executive summary with enrichment | VLOOKUP, IF, conditional logic |

### Example Formulas:
```excel
# VLOOKUP to get service owner
=VLOOKUP(A2, ServiceLookup!$A$2:$C$10, 2, FALSE)

# IF for SLA status
=IF(B2>200, "SLA Breach", "Within SLA")

# Conditional priority flagging
=IF(C2>5%, "Critical", IF(C2>3%, "Warning", "OK"))
```

---

## 📄 Resume Bullet Points

Use these for data analytics / data engineering internship applications:

> • **Built end-to-end operational analytics pipeline** processing 85K+ synthetic API logs, implementing data generation, cleaning, SQL-style analysis, and interactive Plotly dashboards
>
> • **Engineered realistic synthetic data** with temporal patterns (2.8x peak-hour traffic, weekend cycles, degradation events) to simulate production system monitoring scenarios
>
> • **Developed interactive HTML dashboard** with Plotly visualizing traffic patterns, response times, and error rates across 12 API endpoints and 3 microservices
>
> • **Integrated Excel business review layer** with VLOOKUP, IF formulas, and pivot-ready structures following enterprise banking BI practices for executive reporting
>
> • **Identified performance bottlenecks** including `/checkout` endpoint at 400ms avg response time and 4.5% error rate in payments service, providing actionable scaling recommendations

---

## 🔮 Future Enhancements

- [ ] Real-time streaming with Apache Kafka
- [ ] PostgreSQL integration for SQL queries
- [ ] Machine learning anomaly detection
- [ ] Automated alerting with Slack/PagerDuty
- [ ] Cloud deployment (AWS/GCP)

---

## 📞 Contact

**Amitoj Singh Gill**  
[GitHub](https://github.com/gill-amitoj) | [LinkedIn](#) | [Email](#)

---

*Built as a portfolio project demonstrating enterprise data analytics skills for internship applications.*
