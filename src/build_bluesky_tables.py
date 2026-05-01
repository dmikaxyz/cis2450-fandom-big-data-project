from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from config import DATA_PROCESSED_DIR, ensure_data_dirs
from utils import clean_text


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def int_or_zero(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def add_datetime_column(frame: pl.DataFrame, source_col: str, target_col: str) -> pl.DataFrame:
    if frame.is_empty() or source_col not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col(source_col)
        .str.to_datetime(time_zone="UTC", strict=False)
        .alias(target_col)
    )


def post_row(
    run_id: str,
    run_dir: Path,
    ip_name: str,
    matched_queries: list[str],
    post: dict[str, Any],
    is_thread_post: bool = False,
    parent_post_uri: str | None = None,
) -> dict[str, Any] | None:
    uri = post.get("uri")
    if not isinstance(uri, str) or not uri:
        return None

    author = post.get("author") or {}
    record = post.get("record") or {}
    raw_text = record.get("text", "") or ""

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "platform": "bluesky",
        "ip_name": ip_name,
        "post_uri": uri,
        "post_cid": post.get("cid"),
        "parent_post_uri": parent_post_uri,
        "is_thread_post": is_thread_post,
        "created_at": record.get("createdAt") or post.get("indexedAt"),
        "indexed_at": post.get("indexedAt"),
        "author_did": author.get("did"),
        "author_handle": author.get("handle"),
        "author_display_name": author.get("displayName"),
        "raw_text": raw_text,
        "clean_text": clean_text(raw_text),
        "text_length": len(clean_text(raw_text)),
        "like_count": int_or_zero(post.get("likeCount")),
        "repost_count": int_or_zero(post.get("repostCount")),
        "reply_count": int_or_zero(post.get("replyCount")),
        "quote_count": int_or_zero(post.get("quoteCount")),
        "bookmark_count": int_or_zero(post.get("bookmarkCount")),
        "matched_queries": " | ".join(matched_queries),
        "matched_query_count": len(matched_queries),
    }


def flatten_bluesky_posts(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = run_dir.name.removeprefix("bluesky_run_")

    for ip_dir in sorted(run_dir.iterdir()):
        if not ip_dir.is_dir():
            continue
        ip_payload_path = ip_dir / "ip_payload.json"
        if not ip_payload_path.exists():
            continue

        ip_payload = load_json(ip_payload_path)
        ip_name = ip_payload.get("ip_name", ip_dir.name)

        for entry in ip_payload.get("posts", []):
            matched_queries = entry.get("matched_queries", []) or []
            post = entry.get("post") or {}
            row = post_row(run_id, run_dir, ip_name, matched_queries, post, is_thread_post=False)
            if row is None:
                continue
            rows.append(row)

            parent_uri = row["post_uri"]
            for thread_post in entry.get("thread_posts", []) or []:
                thread_row = post_row(
                    run_id,
                    run_dir,
                    ip_name,
                    matched_queries,
                    thread_post,
                    is_thread_post=True,
                    parent_post_uri=parent_uri,
                )
                if thread_row is None:
                    continue
                if thread_row["post_uri"] == parent_uri:
                    continue
                rows.append(thread_row)

    return rows


def build_bluesky_tables(
    run_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, str]:
    ensure_data_dirs()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    target_dir = output_dir or DATA_PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    post_rows = flatten_bluesky_posts(run_dir)
    posts_frame = pl.DataFrame(post_rows) if post_rows else pl.DataFrame()
    posts_frame = add_datetime_column(posts_frame, "created_at", "created_at_ts")
    posts_frame = add_datetime_column(posts_frame, "indexed_at", "indexed_at_ts")

    if not posts_frame.is_empty():
        posts_frame = (
            posts_frame
            .filter(pl.col("author_did").is_not_null() & (pl.col("author_did") != ""))
            .filter(pl.col("clean_text").is_not_null())
            .filter(pl.col("clean_text") != "")
            .unique(subset=["run_id", "ip_name", "post_uri"], keep="first")
            .sort(["ip_name", "created_at_ts"], descending=[False, True])
        )

    users_frame = pl.DataFrame()
    user_ip_frame = pl.DataFrame()

    if not posts_frame.is_empty():
        users_frame = (
            posts_frame
            .group_by(["run_id", "author_did", "author_handle", "author_display_name"])
            .agg(
                pl.len().alias("total_posts"),
                pl.col("ip_name").n_unique().alias("unique_ips"),
                pl.col("created_at_ts").min().alias("first_post_at"),
                pl.col("created_at_ts").max().alias("last_post_at"),
                pl.col("like_count").sum().alias("total_like_count"),
                pl.col("repost_count").sum().alias("total_repost_count"),
                pl.col("reply_count").sum().alias("total_reply_count"),
            )
            .sort(["unique_ips", "total_posts"], descending=[True, True])
        )

        user_ip_frame = (
            posts_frame
            .group_by(["run_id", "ip_name", "author_did", "author_handle", "author_display_name"])
            .agg(
                pl.len().alias("total_posts"),
                pl.col("created_at_ts").min().alias("first_post_at"),
                pl.col("created_at_ts").max().alias("last_post_at"),
                pl.col("like_count").sum().alias("total_like_count"),
                pl.col("repost_count").sum().alias("total_repost_count"),
                pl.col("reply_count").sum().alias("total_reply_count"),
            )
            .sort(["ip_name", "total_posts"], descending=[False, True])
        )

    outputs = {
        "bluesky_posts": target_dir / "bluesky_posts.parquet",
        "bluesky_users": target_dir / "bluesky_users.parquet",
        "bluesky_user_ip_activity": target_dir / "bluesky_user_ip_activity.parquet",
    }

    if not posts_frame.is_empty():
        posts_frame.write_parquet(outputs["bluesky_posts"])
    if not users_frame.is_empty():
        users_frame.write_parquet(outputs["bluesky_users"])
    if not user_ip_frame.is_empty():
        user_ip_frame.write_parquet(outputs["bluesky_user_ip_activity"])

    return {key: str(value) for key, value in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cleaned Bluesky post / user / IP activity tables from a saved bluesky_run directory."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs = build_bluesky_tables(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
    )
    print(outputs)


if __name__ == "__main__":
    main()
