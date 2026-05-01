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


def add_datetime_columns(frame: pl.DataFrame, source_col: str, target_col: str) -> pl.DataFrame:
    if frame.is_empty() or source_col not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col(source_col).str.to_datetime(
            format="%Y-%m-%dT%H:%M:%SZ",
            time_zone="UTC",
            strict=False,
        ).alias(target_col)
    )


def flatten_youtube_comments(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = run_dir.name.removeprefix("youtube_run_")

    for ip_dir in sorted(run_dir.iterdir()):
        if not ip_dir.is_dir():
            continue
        ip_payload_path = ip_dir / "ip_payload.json"
        if not ip_payload_path.exists():
            continue

        ip_payload = load_json(ip_payload_path)
        ip_name = ip_payload.get("ip_name", ip_dir.name)
        for video in ip_payload.get("selected_videos", []):
            video_id = video.get("video_id")
            if not video_id:
                continue

            details = video.get("video_details", {})
            video_snippet = details.get("snippet", {})
            video_stats = details.get("statistics", {})
            matched_queries = video.get("matched_queries", [])
            for bundle in video.get("all_comment_threads", []):
                thread = bundle.get("thread", {})
                thread_snippet = thread.get("snippet", {})
                top_comment = thread_snippet.get("topLevelComment", {})
                top_snippet = top_comment.get("snippet", {})
                content_id = top_comment.get("id")
                if not content_id:
                    continue

                raw_text = top_snippet.get("textDisplay", "")
                commenter_id = top_snippet.get("authorChannelId", {}).get("value")
                rows.append(
                    {
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "platform": "youtube",
                        "ip_name": ip_name,
                        "video_id": video_id,
                        "thread_id": thread.get("id"),
                        "content_id": content_id,
                        "parent_comment_id": None,
                        "comment_level": "top_level",
                        "published_at": top_snippet.get("publishedAt"),
                        "commenter_id": commenter_id,
                        "commenter_name": top_snippet.get("authorDisplayName"),
                        "raw_text": raw_text,
                        "clean_text": clean_text(raw_text),
                        "text_length": len(clean_text(raw_text)),
                        "like_count": int(top_snippet.get("likeCount", 0) or 0),
                        "reply_count": int(thread_snippet.get("totalReplyCount", 0) or 0),
                        "video_title": video_snippet.get("title"),
                        "video_channel_id": video_snippet.get("channelId"),
                        "video_channel_title": video_snippet.get("channelTitle"),
                        "video_view_count": int(video_stats.get("viewCount", 0) or 0),
                        "video_like_count": int(video_stats.get("likeCount", 0) or 0),
                        "video_comment_count": int(video_stats.get("commentCount", 0) or 0),
                        "matched_queries": " | ".join(matched_queries),
                        "matched_query_count": len(matched_queries),
                    }
                )
    return rows


def build_youtube_comment_tables(
    run_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, str]:
    ensure_data_dirs()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    target_dir = output_dir or DATA_PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    comment_rows = flatten_youtube_comments(run_dir)
    comments_frame = pl.DataFrame(comment_rows) if comment_rows else pl.DataFrame()
    comments_frame = add_datetime_columns(comments_frame, "published_at", "published_at_ts")

    if not comments_frame.is_empty():
        comments_frame = (
            comments_frame
            .filter(pl.col("clean_text").is_not_null())
            .filter(pl.col("clean_text") != "")
            .filter(pl.col("clean_text") != "[deleted]")
            .filter(pl.col("clean_text") != "[removed]")
            .unique(subset=["run_id", "content_id"], keep="first")
            .sort(["ip_name", "video_id", "published_at_ts"])
        )

    users_frame = pl.DataFrame()
    user_video_frame = pl.DataFrame()
    user_ip_frame = pl.DataFrame()

    if not comments_frame.is_empty():
        users_frame = (
            comments_frame
            .filter(pl.col("commenter_id").is_not_null() & (pl.col("commenter_id") != ""))
            .group_by(["run_id", "commenter_id", "commenter_name"])
            .agg(
                pl.len().alias("total_comments"),
                pl.col("video_id").n_unique().alias("unique_videos"),
                pl.col("ip_name").n_unique().alias("unique_ips"),
                pl.col("published_at_ts").min().alias("first_comment_at"),
                pl.col("published_at_ts").max().alias("last_comment_at"),
            )
            .sort(["total_comments", "unique_videos"], descending=[True, True])
        )

        user_video_frame = (
            comments_frame
            .filter(pl.col("commenter_id").is_not_null() & (pl.col("commenter_id") != ""))
            .group_by(["run_id", "ip_name", "video_id", "commenter_id", "commenter_name"])
            .agg(
                pl.len().alias("total_comments"),
                pl.col("published_at_ts").min().alias("first_comment_at"),
                pl.col("published_at_ts").max().alias("last_comment_at"),
            )
            .sort(["ip_name", "video_id", "total_comments"], descending=[False, False, True])
        )

        user_ip_frame = (
            comments_frame
            .filter(pl.col("commenter_id").is_not_null() & (pl.col("commenter_id") != ""))
            .group_by(["run_id", "ip_name", "commenter_id", "commenter_name"])
            .agg(
                pl.len().alias("total_comments"),
                pl.col("video_id").n_unique().alias("unique_videos"),
                pl.col("published_at_ts").min().alias("first_comment_at"),
                pl.col("published_at_ts").max().alias("last_comment_at"),
            )
            .sort(["ip_name", "total_comments"], descending=[False, True])
        )

    outputs = {
        "youtube_comments": target_dir / "youtube_comments.parquet",
        "youtube_users": target_dir / "youtube_users.parquet",
        "youtube_user_video_map": target_dir / "youtube_user_video_map.parquet",
        "youtube_user_ip_activity": target_dir / "youtube_user_ip_activity.parquet",
    }

    if not comments_frame.is_empty():
        comments_frame.write_parquet(outputs["youtube_comments"])
    if not users_frame.is_empty():
        users_frame.write_parquet(outputs["youtube_users"])
    if not user_video_frame.is_empty():
        user_video_frame.write_parquet(outputs["youtube_user_video_map"])
    if not user_ip_frame.is_empty():
        user_ip_frame.write_parquet(outputs["youtube_user_ip_activity"])

    return {key: str(value) for key, value in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build YouTube comment/user mapping tables from a saved youtube_run directory."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs = build_youtube_comment_tables(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
    )
    print(outputs)


if __name__ == "__main__":
    main()
