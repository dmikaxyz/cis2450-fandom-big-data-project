from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "duckdb" / "fandom.duckdb"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"


@st.cache_resource(show_spinner=False)
def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


@st.cache_data(show_spinner=False, ttl=300)
def query(sql: str) -> pl.DataFrame:
    return get_connection().sql(sql).pl()


def list_views() -> list[str]:
    df = query(
        """
        select table_name
        from information_schema.tables
        where table_schema = 'main'
        order by table_name
        """
    )
    return df["table_name"].to_list()


def has_view(name: str) -> bool:
    return name in list_views()


def warn_if_missing(views: list[str]) -> list[str]:
    missing = [view for view in views if not has_view(view)]
    if missing:
        st.warning(
            "Missing DuckDB views: "
            + ", ".join(f"`{view}`" for view in missing)
            + ". Run `uv run python src/load_duckdb.py` after building the relevant Parquet tables."
        )
    return missing
