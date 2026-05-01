from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import streamlit as st

from lib import has_view, query, warn_if_missing


st.set_page_config(page_title="Bluesky", layout="wide")
st.title("Bluesky")
st.caption("Posts, users, and audience overlap by IP.")

warn_if_missing(["bluesky_posts", "bluesky_user_ip_activity"])

if not has_view("bluesky_posts") or not has_view("bluesky_user_ip_activity"):
    st.stop()

ip_options = (
    query("select distinct ip_name from bluesky_posts order by ip_name")
    ["ip_name"].to_list()
)
selected_ips = st.multiselect("Filter IPs", ip_options, default=ip_options)
if not selected_ips:
    st.info("Select at least one IP to view data.")
    st.stop()

ip_filter_sql = ", ".join([f"'{ip}'" for ip in selected_ips])

st.subheader("IP-breadth distribution of authors")
breadth = query(
    """
    with per_user as (
        select author_did, count(distinct ip_name) as unique_ips
        from bluesky_user_ip_activity
        where author_did is not null and author_did <> ''
        group by author_did
    )
    select unique_ips, count(*) as n_authors
    from per_user
    group by unique_ips
    order by unique_ips
    """
)
breadth_pd = breadth.to_pandas()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(breadth_pd["unique_ips"], breadth_pd["n_authors"], color="#4c78a8")
ax.set_xlabel("Distinct IPs the author posted about")
ax.set_ylabel("Number of authors")
ax.set_yscale("log")
for x, y in zip(breadth_pd["unique_ips"], breadth_pd["n_authors"]):
    ax.text(x, y, str(y), ha="center", va="bottom", fontsize=9)
ax.set_title("Bluesky authors by IP breadth")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Top multi-IP authors")
multi_ip = query(
    """
    with per_user as (
        select
            author_did,
            any_value(author_handle) as author_handle,
            any_value(author_display_name) as author_display_name,
            count(distinct ip_name) as unique_ips,
            sum(total_posts) as total_posts,
            list(distinct ip_name order by ip_name) as ips
        from bluesky_user_ip_activity
        where author_did is not null and author_did <> ''
        group by author_did
    )
    select * from per_user
    where unique_ips > 1
    order by unique_ips desc, total_posts desc
    limit 100
    """
)
st.dataframe(multi_ip.to_pandas(), use_container_width=True)

st.subheader("Top authors per IP")
per_ip = query(
    f"""
    select ip_name, author_handle, author_display_name, total_posts
    from bluesky_user_ip_activity
    where ip_name in ({ip_filter_sql})
      and author_did is not null and author_did <> ''
    order by total_posts desc
    limit 100
    """
)
st.dataframe(per_ip.to_pandas(), use_container_width=True)

st.subheader("Audience Jaccard overlap between IPs")
user_ip_df = query(
    """
    select author_did, ip_name
    from bluesky_user_ip_activity
    where author_did is not null and author_did <> ''
    """
).unique()

ip_order = sorted(user_ip_df["ip_name"].unique().to_list())
pair_overlap = (
    user_ip_df.rename({"ip_name": "source_ip"})
    .join(user_ip_df.rename({"ip_name": "target_ip"}), on="author_did")
    .filter(pl.col("source_ip") < pl.col("target_ip"))
    .group_by(["source_ip", "target_ip"])
    .agg(pl.len().alias("shared_users"))
)
ip_size = user_ip_df.group_by("ip_name").agg(pl.col("author_did").n_unique().alias("n_users"))
size_lookup = dict(zip(ip_size["ip_name"].to_list(), ip_size["n_users"].to_list()))
index_by_ip = {name: i for i, name in enumerate(ip_order)}

matrix = np.full((len(ip_order), len(ip_order)), np.nan)
for row in pair_overlap.iter_rows(named=True):
    i = index_by_ip[row["source_ip"]]
    j = index_by_ip[row["target_ip"]]
    a = size_lookup[row["source_ip"]]
    b = size_lookup[row["target_ip"]]
    union = a + b - row["shared_users"]
    if union:
        jaccard = row["shared_users"] / union
        matrix[i, j] = jaccard
        matrix[j, i] = jaccard

mask = np.eye(len(ip_order), dtype=bool)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    mask=mask,
    xticklabels=ip_order,
    yticklabels=ip_order,
    cbar_kws={"label": "Jaccard"},
    ax=ax,
)
ax.set_title("Bluesky audience Jaccard overlap")
ax.set_xlabel("Target IP")
ax.set_ylabel("Source IP")
st.pyplot(fig)
