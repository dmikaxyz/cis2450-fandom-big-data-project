from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm.auto import tqdm

from utils import write_json


LOGGER = logging.getLogger("collect_youtube")
SEARCH_RESULTS_PER_QUERY = 50


class QuotaExceededError(RuntimeError):
    """Raised when the YouTube Data API daily quota has been exhausted."""


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_client(api_key: str):
    if not api_key:
        raise ValueError("Missing YOUTUBE_API_KEY in .env")
    return build("youtube", "v3", developerKey=api_key)


def iso_days_ago(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(run_dir / "manifest.json", manifest)


def is_quota_exceeded(exc: HttpError) -> bool:
    try:
        payload = json.loads(exc.content.decode("utf-8"))
        errors = payload.get("error", {}).get("errors", [])
        return any(error.get("reason") == "quotaExceeded" for error in errors)
    except Exception:
        return "quotaExceeded" in str(exc)


def finalize_ip_payload(
    ip_payload: dict[str, Any],
    candidate_video_count: int,
) -> dict[str, Any]:
    ip_payload["candidate_video_count"] = candidate_video_count
    ip_payload["selected_video_count"] = len(ip_payload.get("selected_videos", []))
    ip_payload["total_top_level_comments_fetched"] = sum(
        video.get("fetched_thread_count", 0) for video in ip_payload.get("selected_videos", [])
    )
    ip_payload["total_replies_fetched"] = sum(
        video.get("fetched_reply_count", 0) for video in ip_payload.get("selected_videos", [])
    )
    return ip_payload


def save_ip_payload(ip_dir: Path, ip_payload: dict[str, Any], candidate_video_count: int) -> None:
    write_json(ip_dir / "ip_payload.json", finalize_ip_payload(ip_payload, candidate_video_count))


def summarize_comment_threads(comment_threads: list[dict[str, Any]]) -> tuple[int, int]:
    return (
        len(comment_threads),
        sum(len(bundle.get("all_replies", [])) for bundle in comment_threads),
    )


def search_videos(
    youtube: Any,
    query: str,
    published_after: str | None = None,
) -> list[dict[str, Any]]:
    request_kwargs: dict[str, Any] = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "order": "relevance",
        "maxResults": SEARCH_RESULTS_PER_QUERY,
    }
    if published_after:
        request_kwargs["publishedAfter"] = published_after

    response = youtube.search().list(**request_kwargs).execute()
    return response.get("items", [])


def fetch_video_details(youtube: Any, video_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not video_ids:
        return {}
    details: dict[str, dict[str, Any]] = {}
    for batch in chunks(video_ids, size=50):
        response = (
            youtube.videos()
            .list(part="snippet,statistics", id=",".join(batch))
            .execute()
        )
        for item in response.get("items", []):
            details[item["id"]] = item
    return details


def fetch_comment_threads(
    youtube: Any,
    video_id: str,
    estimated_total_comments: int = 0,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
    max_pagination_steps: int | None = None,
) -> list[dict[str, Any]]:
    enriched_threads: list[dict[str, Any]] = []
    next_page_token: str | None = None
    total_hint = estimated_total_comments if estimated_total_comments > 0 else None
    pages_fetched = 0

    with tqdm(
        total=total_hint,
        desc=f"comments:{video_id[:8]}",
        unit="comment",
        leave=False,
    ) as progress:
        while True:
            response = (
                youtube.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText",
                    order="time",
                )
                .execute()
            )
            batch_threads = response.get("items", [])
            for thread in batch_threads:
                enriched_threads.append(
                    {
                        "thread": thread,
                        "all_replies": [],
                    }
                )
            progress.update(len(batch_threads))
            progress.set_postfix_str(f"threads={len(enriched_threads)}")
            if checkpoint is not None:
                checkpoint(enriched_threads)
            next_page_token = response.get("nextPageToken")
            pages_fetched += 1
            if max_pagination_steps is not None and pages_fetched >= max_pagination_steps:
                break
            if not next_page_token:
                break

        progress.set_postfix_str(f"threads={len(enriched_threads)} pages={pages_fetched} top_level_only")
    return enriched_threads


def select_videos_for_ip(
    youtube: Any,
    ip_name: str,
    query_candidates: list[dict[str, Any]],
    min_comment_count: int,
) -> list[dict[str, Any]]:
    candidate_ids = [candidate["video_id"] for candidate in query_candidates]
    details_lookup = fetch_video_details(youtube, candidate_ids)
    ranked_candidates: list[dict[str, Any]] = []

    for candidate in query_candidates:
        details = details_lookup.get(candidate["video_id"], {})
        stats = details.get("statistics", {})
        comment_count = int(stats.get("commentCount", 0)) if stats.get("commentCount") else 0
        if comment_count < min_comment_count:
            continue
        ranked_candidates.append(
            {
                "video_id": candidate["video_id"],
                "matched_queries": candidate["matched_queries"],
                "search_snippets": candidate["search_snippets"],
                "video_details": details,
                "comment_count": comment_count,
                "view_count": int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
                "ip_name": ip_name,
                "all_comment_threads": [],
                "fetched_thread_count": 0,
                "fetched_reply_count": 0,
                "error": None,
                "status": "selected",
            }
        )

    ranked_candidates.sort(
        key=lambda row: (
            row["comment_count"],
            len(row["matched_queries"]),
            row["view_count"],
        ),
        reverse=True,
    )
    return ranked_candidates
