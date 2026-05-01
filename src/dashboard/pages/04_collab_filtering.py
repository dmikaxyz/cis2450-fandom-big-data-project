from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import streamlit as st

from lib import DATA_PROCESSED_DIR


st.set_page_config(page_title="Collaborative Filtering", layout="wide")
st.title("Collaborative Filtering Model")
st.caption("SVD-based recommender on YouTube `(commenter, video)` interactions.")


VALIDATION_PATH = DATA_PROCESSED_DIR / "cf_validation_results.parquet"
TEST_PATH = DATA_PROCESSED_DIR / "cf_test_results.parquet"
TRAIN_STATS_PATH = DATA_PROCESSED_DIR / "cf_train_matrix_stats.parquet"
IP_AFFINITY_PATH = DATA_PROCESSED_DIR / "cf_ip_affinity.parquet"
USER_FACTORS_PATH = DATA_PROCESSED_DIR / "cf_user_factors.npy"
ITEM_FACTORS_PATH = DATA_PROCESSED_DIR / "cf_item_factors.npy"
USER_INDEX_PATH = DATA_PROCESSED_DIR / "cf_user_index.parquet"
VIDEO_INDEX_PATH = DATA_PROCESSED_DIR / "cf_video_index.parquet"
USER_HISTORY_PATH = DATA_PROCESSED_DIR / "cf_user_history.parquet"
VIDEO_METADATA_PATH = DATA_PROCESSED_DIR / "cf_video_metadata.parquet"


def load_optional(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> dict[str, Any] | None:
    required = [
        USER_FACTORS_PATH,
        ITEM_FACTORS_PATH,
        USER_INDEX_PATH,
        VIDEO_INDEX_PATH,
        USER_HISTORY_PATH,
        VIDEO_METADATA_PATH,
    ]
    if not all(path.exists() for path in required):
        return None

    user_factors = np.load(USER_FACTORS_PATH)
    item_factors = np.load(ITEM_FACTORS_PATH)
    user_index = pl.read_parquet(USER_INDEX_PATH)
    video_index = pl.read_parquet(VIDEO_INDEX_PATH)
    user_history = pl.read_parquet(USER_HISTORY_PATH)
    video_metadata = pl.read_parquet(VIDEO_METADATA_PATH)

    user_to_idx = dict(zip(user_index["commenter_id"].to_list(), user_index["idx"].to_list()))
    video_to_idx = dict(zip(video_index["video_id"].to_list(), video_index["idx"].to_list()))

    user_histories: dict[str, set[str]] = {}
    for row in user_history.iter_rows(named=True):
        user_histories.setdefault(row["commenter_id"], set()).add(row["video_id"])

    metadata_map: dict[str, dict[str, Any]] = {
        row["video_id"]: row for row in video_metadata.iter_rows(named=True)
    }

    return {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_to_idx": user_to_idx,
        "video_to_idx": video_to_idx,
        "user_ids": user_index["commenter_id"].to_list(),
        "video_ids": video_index["video_id"].to_list(),
        "user_histories": user_histories,
        "metadata_map": metadata_map,
    }


def score_user_items(
    artifacts: dict[str, Any],
    user_id: str,
    candidate_video_ids: list[str],
) -> np.ndarray:
    user_idx = artifacts["user_to_idx"][user_id]
    candidate_idxs = np.array([artifacts["video_to_idx"][v] for v in candidate_video_ids])
    user_vector = artifacts["user_factors"][user_idx]
    item_matrix = artifacts["item_factors"][candidate_idxs]
    return item_matrix @ user_vector


def recommend_videos_for_user(
    artifacts: dict[str, Any],
    user_id: str,
    top_k: int = 10,
    exclude_seen: bool = True,
) -> pl.DataFrame:
    if user_id not in artifacts["user_to_idx"]:
        return pl.DataFrame()

    candidate_video_ids = list(artifacts["video_to_idx"].keys())
    if exclude_seen:
        seen = artifacts["user_histories"].get(user_id, set())
        candidate_video_ids = [v for v in candidate_video_ids if v not in seen]
    if not candidate_video_ids:
        return pl.DataFrame()

    scores = score_user_items(artifacts, user_id, candidate_video_ids)
    rows = []
    for video_id, score in zip(candidate_video_ids, scores.tolist()):
        metadata = artifacts["metadata_map"].get(video_id, {})
        rows.append(
            {
                "video_id": video_id,
                "ip_name": metadata.get("ip_name"),
                "title": metadata.get("title"),
                "view_count": metadata.get("view_count"),
                "comment_count": metadata.get("comment_count"),
                "predicted_score": score,
            }
        )

    return pl.DataFrame(rows).sort("predicted_score", descending=True).head(top_k)


def score_ips_for_user(
    artifacts: dict[str, Any],
    user_id: str,
    per_ip_top_k: int = 5,
) -> pl.DataFrame:
    full_recs = recommend_videos_for_user(
        artifacts,
        user_id,
        top_k=len(artifacts["video_to_idx"]),
        exclude_seen=True,
    )
    if full_recs.is_empty():
        return pl.DataFrame()
    return (
        full_recs
        .with_columns(
            pl.col("predicted_score")
              .rank(method="ordinal", descending=True)
              .over("ip_name")
              .alias("ip_rank")
        )
        .filter(pl.col("ip_rank") <= per_ip_top_k)
        .group_by("ip_name")
        .agg(
            pl.len().alias("candidate_videos_considered"),
            pl.col("predicted_score").mean().alias("mean_topk_video_score"),
            pl.col("predicted_score").max().alias("best_video_score"),
        )
        .sort(["mean_topk_video_score", "best_video_score"], descending=[True, True])
    )


st.markdown(
    """
### Approach

- **Implicit feedback**: each `(commenter_id, video_id)` pair where the user commented is a positive interaction, weighted by `total_comments`.
- **Matrix factorization**: a sparse user-video matrix is approximated with truncated SVD, producing low-dimensional `user_factors` and `item_factors`.
- **Ranking task**: for each held-out user, score every unseen video and check whether their true held-out video appears in the top-K.

### Splits

- Users with at least 3 distinct videos
- Per-user split: one train, one validation, one test interaction (seeded)

### Metrics

- **HitRate@10**: did the held-out video appear in the top 10 recommendations
- **NDCG@10**: position-weighted score on the held-out video
- **MRR**: reciprocal rank of the held-out video
"""
)

train_stats = load_optional(TRAIN_STATS_PATH)
if train_stats is not None and not train_stats.is_empty():
    st.subheader("Training matrix stats")
    row = train_stats.row(0, named=True)
    cols = st.columns(4)
    cols[0].metric("Users", f"{int(row.get('num_users', 0)):,}")
    cols[1].metric("Videos", f"{int(row.get('num_videos', 0)):,}")
    cols[2].metric("Observed pairs", f"{int(row.get('observed_user_video_pairs', 0)):,}")
    density = row.get("matrix_density", 0.0) or 0.0
    cols[3].metric("Density", f"{density:.4%}")
    st.dataframe(train_stats.to_pandas(), use_container_width=True)

st.subheader("Validation metrics by latent dimension")
validation_df = load_optional(VALIDATION_PATH)
if validation_df is None or validation_df.is_empty():
    st.warning("`cf_validation_results.parquet` not found.")
else:
    val_sorted = (
        validation_df.sort("n_components")
        if "n_components" in validation_df.columns
        else validation_df
    )
    st.dataframe(val_sorted.to_pandas(), use_container_width=True)

    val_pd = val_sorted.to_pandas()
    if "n_components" in val_pd.columns:
        metric_columns = [c for c in ["hit_rate_at_k", "ndcg_at_k", "mrr"] if c in val_pd.columns]
        if metric_columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            for metric in metric_columns:
                ax.plot(val_pd["n_components"], val_pd[metric], marker="o", label=metric)
            ax.set_xlabel("Latent dimensions")
            ax.set_ylabel("Validation metric")
            ax.set_title("Validation performance by latent dimension")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            metric_cols = st.columns(len(metric_columns))
            for col, metric in zip(metric_cols, metric_columns):
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(val_pd["n_components"], val_pd[metric], marker="o", color="#4c78a8")
                ax.set_xlabel("Latent dimensions")
                ax.set_ylabel(metric)
                ax.set_title(metric)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                col.pyplot(fig)

st.subheader("Held-out test metrics")
test_df = load_optional(TEST_PATH)
if test_df is None or test_df.is_empty():
    st.warning("`cf_test_results.parquet` not found.")
else:
    row = test_df.row(0, named=True)
    cols = st.columns(4)
    if "best_n_components" in row:
        cols[0].metric("Best latent dim", int(row["best_n_components"]))
    if "n_eval" in row:
        cols[1].metric("Test users", f"{int(row['n_eval']):,}")
    if "hit_rate_at_k" in row:
        cols[2].metric("HitRate@10", f"{row['hit_rate_at_k']:.3f}")
    if "ndcg_at_k" in row:
        cols[3].metric("NDCG@10", f"{row['ndcg_at_k']:.3f}")
    if "mrr" in row:
        st.metric("MRR", f"{row['mrr']:.3f}")
    st.dataframe(test_df.to_pandas(), use_container_width=True)

st.subheader("Undirected IP-IP co-affinity from user preference profiles")
st.markdown(
    """
For every user we score every video using `user_factor · item_factor`, aggregate scores per IP and
normalize them into a probability distribution with **softmax** (so SVD's negative scores are handled
cleanly). The pair affinity for a user is `p(IP_A | user) * p(IP_B | user)`. Aggregating the
**average** across all users gives the IP-pair co-affinity heatmap below — the model-based
counterpart to the raw audience-overlap heatmap on the YouTube page.
"""
)
ip_affinity_df = load_optional(IP_AFFINITY_PATH)
if ip_affinity_df is None or ip_affinity_df.is_empty():
    st.info("`cf_ip_affinity.parquet` not found.")
else:
    sort_cols = [c for c in ["avg_pair_strength_per_user", "total_pair_strength"] if c in ip_affinity_df.columns]
    if sort_cols:
        ip_affinity_df = ip_affinity_df.sort(sort_cols, descending=[True] * len(sort_cols))
    st.dataframe(ip_affinity_df.to_pandas(), use_container_width=True)

    if {"ip_a", "ip_b", "avg_pair_strength_per_user"}.issubset(ip_affinity_df.columns):
        ips = sorted(set(ip_affinity_df["ip_a"].to_list()) | set(ip_affinity_df["ip_b"].to_list()))
        index_by_ip = {name: i for i, name in enumerate(ips)}
        matrix = np.full((len(ips), len(ips)), np.nan)
        for row in ip_affinity_df.iter_rows(named=True):
            i = index_by_ip[row["ip_a"]]
            j = index_by_ip[row["ip_b"]]
            matrix[i, j] = row["avg_pair_strength_per_user"]
            matrix[j, i] = row["avg_pair_strength_per_user"]

        mask = np.eye(len(ips), dtype=bool)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".4f",
            cmap="Blues",
            mask=mask,
            xticklabels=ips,
            yticklabels=ips,
            cbar_kws={"label": "Avg pair strength per user"},
            ax=ax,
        )
        ax.set_title("CF-derived IP-IP co-affinity")
        ax.set_xlabel("IP B")
        ax.set_ylabel("IP A")
        st.pyplot(fig)

st.subheader("Try the model on a single user")
artifacts = load_model_artifacts()

if artifacts is None:
    st.info(
        "Model artifacts not found. Export them from the collaborative filtering notebook "
        "(see snippet at the bottom of this section)."
    )
else:
    if "cf_selected_user" not in st.session_state:
        st.session_state.cf_selected_user = artifacts["user_ids"][0]

    def _randomize_user() -> None:
        st.session_state.cf_selected_user = random.choice(artifacts["user_ids"])

    col_input, col_random = st.columns([4, 1])
    with col_input:
        st.text_input("Commenter ID", key="cf_selected_user")
    with col_random:
        st.write("")
        st.write("")
        st.button("Random user", on_click=_randomize_user)

    selected_user = st.session_state.cf_selected_user
    if selected_user not in artifacts["user_to_idx"]:
        st.error(f"`{selected_user}` is not in the trained user index.")
    else:
        history = sorted(artifacts["user_histories"].get(selected_user, set()))
        st.caption(f"User has {len(history)} interactions in training history.")

        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Score IPs for this user"):
                ip_scores = score_ips_for_user(artifacts, selected_user, per_ip_top_k=5)
                if ip_scores.is_empty():
                    st.warning("No IPs to score (user has seen everything?).")
                else:
                    st.dataframe(ip_scores.to_pandas(), use_container_width=True)
        with action_cols[1]:
            if st.button("Recommend top 10 videos"):
                recs = recommend_videos_for_user(artifacts, selected_user, top_k=10)
                if recs.is_empty():
                    st.warning("No recommendations available.")
                else:
                    st.dataframe(recs.to_pandas(), use_container_width=True)
