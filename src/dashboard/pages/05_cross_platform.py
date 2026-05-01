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

from lib import has_view, query


st.set_page_config(page_title="Cross-Platform", layout="wide")
st.title("Cross-Platform Overlap")
st.caption("Side-by-side YouTube vs Bluesky audience overlap.")


def jaccard_matrix_for(view: str, user_id_col: str) -> tuple[list[str], np.ndarray]:
    user_ip_df = query(
        f"""
        select {user_id_col} as user_id, ip_name
        from {view}
        where {user_id_col} is not null and {user_id_col} <> ''
        """
    ).unique()

    ip_order = sorted(user_ip_df["ip_name"].unique().to_list())
    if not ip_order:
        return ip_order, np.full((0, 0), np.nan)

    pair_overlap = (
        user_ip_df.rename({"ip_name": "source_ip"})
        .join(user_ip_df.rename({"ip_name": "target_ip"}), on="user_id")
        .filter(pl.col("source_ip") < pl.col("target_ip"))
        .group_by(["source_ip", "target_ip"])
        .agg(pl.len().alias("shared_users"))
    )
    ip_size = user_ip_df.group_by("ip_name").agg(pl.col("user_id").n_unique().alias("n_users"))
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

    return ip_order, matrix


platforms = []
if has_view("youtube_user_ip_activity"):
    platforms.append(("YouTube", "youtube_user_ip_activity", "commenter_id"))
if has_view("bluesky_user_ip_activity"):
    platforms.append(("Bluesky", "bluesky_user_ip_activity", "author_did"))

if not platforms:
    st.info("Build the YouTube and Bluesky tables first, then re-run `load_duckdb.py`.")
    st.stop()

cols = st.columns(len(platforms))
for col, (label, view, user_col) in zip(cols, platforms):
    ip_order, matrix = jaccard_matrix_for(view, user_col)
    if matrix.size == 0:
        col.warning(f"No data in `{view}`.")
        continue
    mask = np.eye(len(ip_order), dtype=bool)
    fig, ax = plt.subplots(figsize=(6, 5))
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
    ax.set_title(f"{label} Jaccard overlap")
    ax.set_xlabel("Target IP")
    ax.set_ylabel("Source IP")
    col.pyplot(fig)

if len(platforms) == 2:
    st.subheader("Pair-by-pair comparison")
    rows: list[dict] = []
    matrices = {label: jaccard_matrix_for(view, col_name) for label, view, col_name in platforms}
    yt_ips, yt_matrix = matrices["YouTube"]
    bs_ips, bs_matrix = matrices["Bluesky"]
    common = sorted(set(yt_ips) & set(bs_ips))
    yt_idx = {name: i for i, name in enumerate(yt_ips)}
    bs_idx = {name: i for i, name in enumerate(bs_ips)}
    for i, ip_a in enumerate(common):
        for ip_b in common[i + 1 :]:
            rows.append(
                {
                    "ip_a": ip_a,
                    "ip_b": ip_b,
                    "youtube_jaccard": yt_matrix[yt_idx[ip_a], yt_idx[ip_b]],
                    "bluesky_jaccard": bs_matrix[bs_idx[ip_a], bs_idx[ip_b]],
                }
            )
    if rows:
        st.dataframe(
            pl.DataFrame(rows)
            .sort("youtube_jaccard", descending=True)
            .to_pandas(),
            use_container_width=True,
        )
