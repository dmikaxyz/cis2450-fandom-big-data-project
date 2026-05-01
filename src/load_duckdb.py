from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from config import DATA_DUCKDB_DIR, DATA_PROCESSED_DIR, ensure_data_dirs


def parquet_path(name: str) -> Path:
    return DATA_PROCESSED_DIR / name


def load_database(database_path: Path | None = None) -> str:
    ensure_data_dirs()
    db_path = database_path or (DATA_DUCKDB_DIR / "fandom.duckdb")
    con = duckdb.connect(str(db_path))
    con.execute("drop view if exists ip_pair_day_features")

    tables = {
        "ip_alias_lookup": parquet_path("ip_alias_lookup.parquet"),
        "youtube_comments_clean": parquet_path("youtube_comments_clean.parquet"),
        "youtube_comments": parquet_path("youtube_comments.parquet"),
        "youtube_users": parquet_path("youtube_users.parquet"),
        "youtube_user_video_map": parquet_path("youtube_user_video_map.parquet"),
        "youtube_user_ip_activity": parquet_path("youtube_user_ip_activity.parquet"),
        "youtube_videos": parquet_path("youtube_videos.parquet"),
        "youtube_video_query_map": parquet_path("youtube_video_query_map.parquet"),
        "youtube_query_results": parquet_path("youtube_query_results.parquet"),
        "reddit_posts_comments_clean": parquet_path("reddit_posts_comments_clean.parquet"),
        "bluesky_posts": parquet_path("bluesky_posts.parquet"),
        "bluesky_users": parquet_path("bluesky_users.parquet"),
        "bluesky_user_ip_activity": parquet_path("bluesky_user_ip_activity.parquet"),
        "ip_day_activity": parquet_path("ip_day_activity.parquet"),
        "user_ip_activity": parquet_path("user_ip_activity.parquet"),
        "ip_user_overlap": parquet_path("ip_user_overlap.parquet"),
        "video_inventory": parquet_path("video_inventory.parquet"),
    }

    for table_name, path in tables.items():
        if path.exists():
            con.execute(
                f"""
                create or replace view {table_name} as
                select * from read_parquet('{path.as_posix()}')
                """
            )
    con.close()
    return str(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load processed Parquet tables into DuckDB views.")
    parser.add_argument("--database-path", type=Path, default=None)
    args = parser.parse_args()
    db_path = load_database(database_path=args.database_path)
    print({"database_path": db_path})


if __name__ == "__main__":
    main()
