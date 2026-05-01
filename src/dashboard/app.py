from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st

from lib import DB_PATH, has_view, list_views, query


st.set_page_config(page_title="Fandom Project Dashboard", layout="wide")

st.title("Cross-Platform Fandom Analysis")
st.caption("YouTube comments · Bluesky posts · Reddit threads · Collaborative filtering")

if not DB_PATH.exists():
    st.error(
        f"DuckDB file not found at `{DB_PATH}`.\n\n"
        "Build processed tables and run `uv run python src/load_duckdb.py` first."
    )
    st.stop()

views = list_views()
st.sidebar.header("Status")
st.sidebar.write(f"Database: `{DB_PATH.name}`")
st.sidebar.write(f"Views available: **{len(views)}**")
with st.sidebar.expander("Show all views"):
    for view in views:
        st.write(view)

st.markdown(
    """
Use the sidebar to navigate between platforms:

- **Overview** — totals across IPs and platforms
- **YouTube** — comments, top users, IP overlap
- **Bluesky** — posts, top users, IP overlap
- **Collaborative filtering** — model evaluation results
- **Cross-platform** — YouTube vs Bluesky overlap side-by-side
"""
)

st.subheader("Quick stats")
metric_cols = st.columns(4)

if has_view("youtube_comments"):
    yt = query("select count(*) as n_comments, count(distinct commenter_id) as n_users, count(distinct ip_name) as n_ips from youtube_comments")
    metric_cols[0].metric("YouTube comments", f"{yt[0, 'n_comments']:,}")
    metric_cols[1].metric("YouTube commenters", f"{yt[0, 'n_users']:,}")

if has_view("bluesky_posts"):
    bs = query("select count(*) as n_posts, count(distinct author_did) as n_authors from bluesky_posts")
    metric_cols[2].metric("Bluesky posts", f"{bs[0, 'n_posts']:,}")
    metric_cols[3].metric("Bluesky authors", f"{bs[0, 'n_authors']:,}")
