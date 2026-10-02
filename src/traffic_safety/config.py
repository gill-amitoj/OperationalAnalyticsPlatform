"""
config.py
=========
Dataset registry and project paths.

All three datasets are tables from the City of Edmonton's Annual Collision
Report, published on data.edmonton.ca (Socrata) under the
Open Government Licence – City of Edmonton.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DUCKDB_PATH = ROOT / "data" / "traffic_safety.duckdb"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
TABLEAU_DIR = ROOT / "tableau"
DASHBOARD_DIR = ROOT / "dashboard"

SOCRATA_DOMAIN = "data.edmonton.ca"

ATTRIBUTION = (
    "Contains information licensed under the Open Government Licence – City of Edmonton."
)
LICENCE_URL = "https://data.edmonton.ca/stories/s/City-of-Edmonton-Open-Data-Terms-of-Use/msh8-if28"


@dataclass(frozen=True)
class Dataset:
    key: str          # short name used for files and tables
    socrata_id: str   # four-by-four id on data.edmonton.ca
    title: str

    @property
    def resource_url(self) -> str:
        return f"https://{SOCRATA_DOMAIN}/resource/{self.socrata_id}.csv"

    @property
    def metadata_url(self) -> str:
        return f"https://{SOCRATA_DOMAIN}/api/views/{self.socrata_id}.json"

    @property
    def portal_url(self) -> str:
        return f"https://{SOCRATA_DOMAIN}/d/{self.socrata_id}"


DATASETS = {
    "temporal": Dataset("temporal", "jduq-w5pj", "Annual Collision Report: Temporal"),
    "top_locations": Dataset("top_locations", "mf6n-s5ts", "Annual Collision Report: Top Collision Locations"),
    "severity": Dataset("severity", "77sf-j5rj", "Annual Collision Report: Collision Severity"),
}
