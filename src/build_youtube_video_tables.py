from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from config import DATA_PROCESSED_DIR, ensure_data_dirs


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def int_or_zero(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def flatten_search_results(run_dir: Path, ip_dir: Path, ip_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for search_path in sorted(ip_dir.glob("search_*.json")):
        payload = load_json(search_path)
        query = payload.get("query")
        for rank, item in enumerate(payload.get("items", []), start=1):
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue
            rows.append(
                {
                    "run_id": run_dir.name.removeprefix("youtube_run_"),
                    "run_dir": str(run_dir),
                    "ip_name": ip_name,
                    "query": query,
                    "query_result_rank": rank,
                    "video_id": video_id,
                    "search_title": snippet.get("title"),
                    "search_description": snippet.get("description"),
                    "search_channel_id": snippet.get("channelId"),
                    "search_channel_title": snippet.get("channelTitle"),
                    "search_published_at": snippet.get("publishedAt") or snippet.get("publishTime"),
                }
            )
    return rows


def build_video_and_query_rows(run_dir: Path, ip_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ip_payload_path = ip_dir / "ip_payload.json"
    if not ip_payload_path.exists():
        raise FileNotFoundError(f"Missing ip_payload.json in {ip_dir}")

    ip_payload = load_json(ip_payload_path)
    ip_name = ip_payload.get("ip_name", ip_dir.name)
    run_id = run_dir.name.removeprefix("youtube_run_")

    query_result_rows = flatten_search_results(run_dir, ip_dir, ip_name)
    query_lookup: dict[str, list[dict[str, Any]]] = {}
    for row in query_result_rows:
        query_lookup.setdefault(row["video_id"], []).append(row)

    video_rows: list[dict[str, Any]] = []
    video_query_rows: list[dict[str, Any]] = []

    for rank, video in enumerate(ip_payload.get("selected_videos", []), start=1):
        video_id = video.get("video_id")
        if not video_id:
            continue

        details = video.get("video_details", {})
        snippet = details.get("snippet", {})
        stats = details.get("statistics", {})
        search_snippets = video.get("search_snippets", [])
        first_search_snippet = {}
        if search_snippets:
            first_search_snippet = search_snippets[0].get("snippet", {})
        elif query_lookup.get(video_id):
            first_search_snippet = {
                "title": query_lookup[video_id][0].get("search_title"),
                "description": query_lookup[video_id][0].get("search_description"),
                "channelId": query_lookup[video_id][0].get("search_channel_id"),
                "channelTitle": query_lookup[video_id][0].get("search_channel_title"),
                "publishedAt": query_lookup[video_id][0].get("search_published_at"),
            }

        matched_queries = video.get("matched_queries", [])
        if not matched_queries and query_lookup.get(video_id):
            matched_queries = sorted({row["query"] for row in query_lookup[video_id] if row.get("query")})

        title = snippet.get("title") or video.get("title") or first_search_snippet.get("title")
        description = snippet.get("description") or first_search_snippet.get("description")
        channel_id = snippet.get("channelId") or first_search_snippet.get("channelId")
        channel_title = snippet.get("channelTitle") or first_search_snippet.get("channelTitle")
        published_at = snippet.get("publishedAt") or first_search_snippet.get("publishedAt")

        row = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "ip_name": ip_name,
            "video_rank_within_ip": rank,
            "video_id": video_id,
            "title": title,
            "description": description,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "published_at": published_at,
            "view_count": int_or_zero(stats.get("viewCount") or video.get("view_count")),
            "like_count": int_or_zero(stats.get("likeCount")),
            "favorite_count": int_or_zero(stats.get("favoriteCount")),
            "comment_count": int_or_zero(stats.get("commentCount") or video.get("comment_count")),
            "matched_queries": " | ".join(matched_queries),
            "matched_query_count": len(matched_queries),
            "search_snippet_count": len(search_snippets) if search_snippets else len(query_lookup.get(video_id, [])),
            "fetched_thread_count": int_or_zero(video.get("fetched_thread_count")),
            "status": video.get("status", "selected"),
        }
        video_rows.append(row)

        for query in matched_queries:
            video_query_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "ip_name": ip_name,
                    "video_id": video_id,
                    "query": query,
                }
            )

    return video_rows, video_query_rows, query_result_rows


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


def build_youtube_video_tables(
    run_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, str]:
    ensure_data_dirs()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    target_dir = output_dir or DATA_PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    video_rows: list[dict[str, Any]] = []
    video_query_rows: list[dict[str, Any]] = []
    query_result_rows: list[dict[str, Any]] = []

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "ip_payload.json").exists():
            continue
        child_video_rows, child_query_map_rows, child_query_result_rows = build_video_and_query_rows(run_dir, child)
        video_rows.extend(child_video_rows)
        video_query_rows.extend(child_query_map_rows)
        query_result_rows.extend(child_query_result_rows)

    videos_frame = pl.DataFrame(video_rows) if video_rows else pl.DataFrame()
    videos_frame = add_datetime_columns(videos_frame, "published_at", "published_at_ts")
    if not videos_frame.is_empty():
        videos_frame = videos_frame.sort(
            ["ip_name", "video_rank_within_ip", "comment_count"],
            descending=[False, False, True],
        )

    video_query_frame = pl.DataFrame(video_query_rows) if video_query_rows else pl.DataFrame()
    if not video_query_frame.is_empty():
        video_query_frame = video_query_frame.unique(subset=["run_id", "ip_name", "video_id", "query"], keep="first")

    query_results_frame = pl.DataFrame(query_result_rows) if query_result_rows else pl.DataFrame()
    query_results_frame = add_datetime_columns(query_results_frame, "search_published_at", "search_published_at_ts")
    if not query_results_frame.is_empty():
        query_results_frame = query_results_frame.sort(["ip_name", "query", "query_result_rank"])

    outputs = {
        "youtube_videos": target_dir / "youtube_videos.parquet",
        "youtube_video_query_map": target_dir / "youtube_video_query_map.parquet",
        "youtube_query_results": target_dir / "youtube_query_results.parquet",
    }

    if not videos_frame.is_empty():
        videos_frame.write_parquet(outputs["youtube_videos"])
    if not video_query_frame.is_empty():
        video_query_frame.write_parquet(outputs["youtube_video_query_map"])
    if not query_results_frame.is_empty():
        query_results_frame.write_parquet(outputs["youtube_query_results"])

    return {key: str(value) for key, value in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DuckDB-friendly YouTube video and query tables from a saved youtube_run directory."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs = build_youtube_video_tables(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
    )
    print(outputs)


if __name__ == "__main__":
    main()
