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


st.set_page_config(page_title="YouTube", layout="wide")
st.title("YouTube")
st.caption("Per-IP collection stats, audience overlap, and cross-IP videos.")

required_views = ["youtube_videos", "youtube_user_ip_activity", "youtube_user_video_map"]
warn_if_missing(required_views)
if not all(has_view(view) for view in required_views):
    st.stop()

st.subheader("Per-IP video stats")
ip_video_stats = query(
    """
    select
        ip_name,
        count(*) as videos,
        avg(comment_count) as avg_comment_count,
        max(comment_count) as max_comment_count
    from youtube_videos
    group by ip_name
    order by videos desc
    """
)
st.dataframe(ip_video_stats.to_pandas(), use_container_width=True)

st.subheader("Total collected comment count by IP")
total_comments = query(
    """
    select
        ip_name,
        sum(comment_count) as total_comment_count
    from youtube_videos
    group by ip_name
    order by total_comment_count desc
    """
)
total_comments_pd = total_comments.to_pandas()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(total_comments_pd["ip_name"], total_comments_pd["total_comment_count"], color="#4c78a8")
ax.set_ylabel("Total comment count")
ax.set_title("Total collected comment count by IP")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Shared commenter overlap heatmap")
user_ip_df = query(
    """
    select commenter_id, ip_name
    from youtube_user_ip_activity
    where commenter_id is not null and commenter_id <> ''
    """
).unique()

ip_order = sorted(user_ip_df["ip_name"].unique().to_list())
pair_overlap = (
    user_ip_df.rename({"ip_name": "source_ip"})
    .join(user_ip_df.rename({"ip_name": "target_ip"}), on="commenter_id")
    .filter(pl.col("source_ip") < pl.col("target_ip"))
    .group_by(["source_ip", "target_ip"])
    .agg(pl.len().alias("shared_commenters"))
)

heatmap_matrix = np.zeros((len(ip_order), len(ip_order)), dtype=int)
index_by_ip = {name: i for i, name in enumerate(ip_order)}
for row in pair_overlap.iter_rows(named=True):
    i = index_by_ip[row["source_ip"]]
    j = index_by_ip[row["target_ip"]]
    heatmap_matrix[i, j] = row["shared_commenters"]
    heatmap_matrix[j, i] = row["shared_commenters"]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    heatmap_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ip_order,
    yticklabels=ip_order,
    ax=ax,
)
ax.set_title("Shared commenter overlap between IPs")
ax.set_xlabel("Target IP")
ax.set_ylabel("Source IP")
st.pyplot(fig)

st.subheader("Top videos ranked by distinct commenters who also appear in multiple IPs")
cross_ip_videos = query(
    """
    with user_ip_span as (
        select commenter_id, count(distinct ip_name) as unique_ips
        from youtube_user_ip_activity
        where commenter_id is not null and commenter_id <> ''
        group by commenter_id
    ),
    user_video as (
        select distinct ip_name, video_id, commenter_id
        from youtube_user_video_map
        where commenter_id is not null and commenter_id <> ''
    ),
    video_size as (
        select ip_name, video_id, count(distinct commenter_id) as distinct_commenters_on_video
        from user_video
        group by ip_name, video_id
    )
    select
        uv.ip_name as "IP",
        v.title as "Video Title",
        count(distinct uv.commenter_id) as "Cross-IP Commenters",
        vs.distinct_commenters_on_video as "Distinct Commenters On Video",
        count(distinct uv.commenter_id) * 1.0 / vs.distinct_commenters_on_video as "Cross-IP Share"
    from user_video uv
    join user_ip_span uis
      on uis.commenter_id = uv.commenter_id and uis.unique_ips > 1
    left join youtube_videos v
      on v.ip_name = uv.ip_name and v.video_id = uv.video_id
    left join video_size vs
      on vs.ip_name = uv.ip_name and vs.video_id = uv.video_id
    group by uv.ip_name, uv.video_id, v.title, vs.distinct_commenters_on_video
    order by "Cross-IP Commenters" desc, "Cross-IP Share" desc
    limit 20
    """
)
st.dataframe(
    cross_ip_videos.to_pandas().style.format({
        "Cross-IP Commenters": "{:,.0f}",
        "Distinct Commenters On Video": "{:,.0f}",
        "Cross-IP Share": "{:.1%}",
    }),
    use_container_width=True,
)

st.subheader("Top 10 cross-IP video pairs ranked by shared commenters")
video_pair_overlap = query(
    """
    with user_video as (
        select distinct ip_name, video_id, commenter_id
        from youtube_user_video_map
        where commenter_id is not null and commenter_id <> ''
    ),
    video_size as (
        select ip_name, video_id, count(distinct commenter_id) as distinct_commenters
        from user_video
        group by ip_name, video_id
    )
    select
        a.ip_name as "Left IP",
        av.title as "Left Video",
        b.ip_name as "Right IP",
        bv.title as "Right Video",
        count(distinct a.commenter_id) as "Shared Commenters",
        count(distinct a.commenter_id) * 1.0 /
            (avs.distinct_commenters + bvs.distinct_commenters - count(distinct a.commenter_id))
            as "Jaccard Similarity"
    from user_video a
    join user_video b
      on a.commenter_id = b.commenter_id
     and a.ip_name < b.ip_name
    left join youtube_videos av
      on av.ip_name = a.ip_name and av.video_id = a.video_id
    left join youtube_videos bv
      on bv.ip_name = b.ip_name and bv.video_id = b.video_id
    left join video_size avs
      on avs.ip_name = a.ip_name and avs.video_id = a.video_id
    left join video_size bvs
      on bvs.ip_name = b.ip_name and bvs.video_id = b.video_id
    group by a.ip_name, a.video_id, av.title,
             b.ip_name, b.video_id, bv.title,
             avs.distinct_commenters, bvs.distinct_commenters
    order by "Shared Commenters" desc, "Jaccard Similarity" desc
    limit 10
    """
)
st.dataframe(
    video_pair_overlap.to_pandas().style.format({
        "Shared Commenters": "{:,.0f}",
        "Jaccard Similarity": "{:.1%}",
    }),
    use_container_width=True,
)
