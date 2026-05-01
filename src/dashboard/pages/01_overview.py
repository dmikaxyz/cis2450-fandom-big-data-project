from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from lib import has_view, query, warn_if_missing


st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")
st.caption("IP catalog and per-platform totals.")

warn_if_missing(["youtube_comments", "bluesky_posts"])

st.subheader("Posts/comments per IP per platform")
parts: list[pl.DataFrame] = []
if has_view("youtube_comments"):
    parts.append(
        query(
            """
            select ip_name, count(*) as n_items, 'youtube' as platform
            from youtube_comments
            group by ip_name
            """
        )
    )
if has_view("bluesky_posts"):
    parts.append(
        query(
            """
            select ip_name, count(*) as n_items, 'bluesky' as platform
            from bluesky_posts
            group by ip_name
            """
        )
    )

if not parts:
    st.info("No platform tables found yet.")
    st.stop()

totals = pl.concat(parts).sort(["ip_name", "platform"])
pivot = (
    totals
    .pivot(values="n_items", index="ip_name", on="platform")
    .fill_null(0)
    .sort("ip_name")
)
st.dataframe(pivot.to_pandas(), use_container_width=True)

st.subheader("Bar chart")
totals_pd = totals.to_pandas()
fig, ax = plt.subplots(figsize=(10, 5))
for platform, group in totals_pd.groupby("platform"):
    ax.bar(group["ip_name"], group["n_items"], label=platform, alpha=0.8)
ax.set_ylabel("Items collected")
ax.set_xlabel("IP")
ax.set_title("Items per IP per platform")
ax.legend()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Unique users per IP per platform")
unique_parts: list[pl.DataFrame] = []
if has_view("youtube_user_ip_activity"):
    unique_parts.append(
        query(
            """
            select ip_name, count(distinct commenter_id) as n_users, 'youtube' as platform
            from youtube_user_ip_activity
            where commenter_id is not null and commenter_id <> ''
            group by ip_name
            """
        )
    )
if has_view("bluesky_user_ip_activity"):
    unique_parts.append(
        query(
            """
            select ip_name, count(distinct author_did) as n_users, 'bluesky' as platform
            from bluesky_user_ip_activity
            where author_did is not null and author_did <> ''
            group by ip_name
            """
        )
    )

if unique_parts:
    unique_totals = (
        pl.concat(unique_parts)
        .pivot(values="n_users", index="ip_name", on="platform")
        .fill_null(0)
        .sort("ip_name")
    )
    st.dataframe(unique_totals.to_pandas(), use_container_width=True)
