from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from googleapiclient.errors import HttpError
from tqdm.auto import tqdm

from config import DATA_RAW_DIR, ensure_data_dirs, load_settings
from ip_config import PRIMARY_IPS
from utils import safe_filename, write_json
from youtube_collect_common import (
    LOGGER,
    SEARCH_RESULTS_PER_QUERY,
    QuotaExceededError,
    build_client,
    configure_logging,
    finalize_ip_payload,
    is_quota_exceeded,
    iso_days_ago,
    save_ip_payload,
    save_manifest,
    search_videos,
    select_videos_for_ip,
)


def collect_youtube_search_metadata(
    output_dir: Path,
    queries_per_ip: int | None = None,
    min_comment_count: int = 100,
    days_back: int | None = None,
) -> dict[str, Any]:
    configure_logging()
    settings = load_settings()
    ensure_data_dirs()
    youtube = build_client(settings.youtube_api_key)
    published_after = iso_days_ago(days_back) if days_back is not None else None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / "youtube_runs" / f"youtube_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_payload: dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "run_id": timestamp,
        "run_dir": str(run_dir),
        "stage": "search_metadata",
        "ips": [],
    }
    manifest: dict[str, Any] = {
        "run_id": timestamp,
        "started_at": run_payload["collected_at"],
        "stage": "search_metadata",
        "status": "running",
        "params": {
            "queries_per_ip": queries_per_ip,
            "search_results_per_query": SEARCH_RESULTS_PER_QUERY,
            "min_comment_count": min_comment_count,
            "days_back": days_back,
        },
        "ips": {},
    }
    save_manifest(run_dir, manifest)

    current_ip_name: str | None = None
    current_ip_dir: Path | None = None
    current_ip_payload: dict[str, Any] | None = None
    current_candidate_video_count = 0

    try:
        ip_iterator = tqdm(PRIMARY_IPS, desc="IPs", unit="ip")
        for ip_config in ip_iterator:
            ip_name = ip_config.canonical_name
            current_ip_name = ip_name
            ip_iterator.set_postfix_str(ip_name)
            LOGGER.info("Starting search stage for IP '%s'", ip_name)

            ip_dir = run_dir / safe_filename(ip_name)
            ip_dir.mkdir(parents=True, exist_ok=True)
            current_ip_dir = ip_dir

            ip_payload: dict[str, Any] = {
                "ip_name": ip_name,
                "queries": [],
                "selected_videos": [],
            }
            current_ip_payload = ip_payload
            active_queries = (
                ip_config.youtube_queries[:queries_per_ip]
                if queries_per_ip is not None
                else list(ip_config.youtube_queries)
            )
            ip_payload["queries_used"] = active_queries

            manifest["ips"][ip_name] = {
                "status": "running",
                "queries_used": active_queries,
                "candidate_video_count": 0,
                "selected_video_count": 0,
                "videos_completed": 0,
                "total_top_level_comments_fetched": 0,
                "total_replies_fetched": 0,
            }
            save_manifest(run_dir, manifest)

            candidates_by_id: dict[str, dict[str, Any]] = {}
            for query in active_queries:
                LOGGER.info("Searching query '%s' for '%s'", query, ip_name)
                try:
                    search_items = search_videos(
                        youtube=youtube,
                        query=query,
                        published_after=published_after,
                    )
                except HttpError as exc:
                    if is_quota_exceeded(exc):
                        raise QuotaExceededError(str(exc)) from exc
                    raise

                write_json(
                    ip_dir / f"search_{safe_filename(query)}.json",
                    {"query": query, "items": search_items},
                )

                query_payload: dict[str, Any] = {"query": query, "search_results": len(search_items)}
                for item in search_items:
                    video_id = item.get("id", {}).get("videoId")
                    if not video_id:
                        continue
                    if video_id not in candidates_by_id:
                        candidates_by_id[video_id] = {
                            "video_id": video_id,
                            "matched_queries": [],
                            "search_snippets": [],
                        }
                    if query not in candidates_by_id[video_id]["matched_queries"]:
                        candidates_by_id[video_id]["matched_queries"].append(query)
                    candidates_by_id[video_id]["search_snippets"].append(item)

                ip_payload["queries"].append(query_payload)
                save_ip_payload(ip_dir, ip_payload, len(candidates_by_id))

            manifest["ips"][ip_name]["candidate_video_count"] = len(candidates_by_id)
            current_candidate_video_count = len(candidates_by_id)
            save_manifest(run_dir, manifest)

            try:
                selected_videos = select_videos_for_ip(
                    youtube=youtube,
                    ip_name=ip_name,
                    query_candidates=list(candidates_by_id.values()),
                    min_comment_count=min_comment_count,
                )
            except HttpError as exc:
                if is_quota_exceeded(exc):
                    raise QuotaExceededError(str(exc)) from exc
                raise

            ip_payload["selected_videos"] = selected_videos
            save_ip_payload(ip_dir, ip_payload, current_candidate_video_count)

            write_json(
                ip_dir / "selected_videos.json",
                {
                    "ip_name": ip_name,
                    "candidate_video_count": len(candidates_by_id),
                    "selected_video_count": len(selected_videos),
                    "selected_videos": [
                        {
                            "video_id": video["video_id"],
                            "matched_queries": video["matched_queries"],
                            "comment_count": video["comment_count"],
                            "view_count": video["view_count"],
                            "title": video.get("video_details", {}).get("snippet", {}).get("title"),
                        }
                        for video in selected_videos
                    ],
                },
            )

            manifest["ips"][ip_name]["selected_video_count"] = len(selected_videos)
            manifest["ips"][ip_name]["status"] = "metadata_completed"
            save_manifest(run_dir, manifest)

            finalized_ip_payload = finalize_ip_payload(ip_payload, current_candidate_video_count)
            run_payload["ips"].append(finalized_ip_payload)
            current_ip_name = None
            current_ip_dir = None
            current_ip_payload = None
            current_candidate_video_count = 0

    except QuotaExceededError as exc:
        if current_ip_name and current_ip_payload and current_ip_dir:
            finalized_ip_payload = finalize_ip_payload(current_ip_payload, current_candidate_video_count)
            write_json(current_ip_dir / "ip_payload.json", finalized_ip_payload)
            manifest["ips"][current_ip_name]["status"] = "partial"
            if not any(ip.get("ip_name") == current_ip_name for ip in run_payload["ips"]):
                run_payload["ips"].append(finalized_ip_payload)
        manifest["status"] = "quota_exceeded"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = str(exc)
        save_manifest(run_dir, manifest)
        partial_path = run_dir / "partial_run_payload.json"
        write_json(partial_path, run_payload)
        LOGGER.error("Quota exceeded during search stage. Partial payload saved to %s", partial_path)
        raise

    output_path = output_dir / f"youtube_search_{timestamp}.json"
    write_json(output_path, run_payload)
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["final_output_path"] = str(output_path)
    save_manifest(run_dir, manifest)
    LOGGER.info("Finished search stage. Metadata saved to %s", output_path)
    return {
        "output_path": str(output_path),
        "run_dir": str(run_dir),
        "ips_collected": len(run_payload["ips"]),
        "queries_per_ip": queries_per_ip,
        "search_results_per_query": SEARCH_RESULTS_PER_QUERY,
        "min_comment_count": min_comment_count,
        "days_back": days_back,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search YouTube fandom queries, cache the top 50 results per query, and save all qualifying video metadata."
    )
    parser.add_argument(
        "--queries-per-ip",
        type=int,
        default=None,
        help="Optional cap on queries per IP. If omitted, all configured queries for each IP are used.",
    )
    parser.add_argument("--min-comment-count", type=int, default=100)
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Optional published-after filter in days. If omitted, search results are not date-filtered.",
    )
    args = parser.parse_args()

    result = collect_youtube_search_metadata(
        output_dir=DATA_RAW_DIR,
        queries_per_ip=args.queries_per_ip,
        min_comment_count=args.min_comment_count,
        days_back=args.days_back,
    )
    print(result)


if __name__ == "__main__":
    main()
