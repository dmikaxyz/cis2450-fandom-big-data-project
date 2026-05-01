from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from googleapiclient.errors import HttpError

from config import DATA_RAW_DIR, ensure_data_dirs, load_settings
from utils import safe_filename, write_json
from youtube_collect_common import (
    LOGGER,
    QuotaExceededError,
    build_client,
    configure_logging,
    fetch_comment_threads,
    finalize_ip_payload,
    is_quota_exceeded,
    save_manifest,
    summarize_comment_threads,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_round_robin_top_level_comments(
    run_dir: Path,
    output_dir: Path,
    max_videos: int,
    max_pagination_steps_per_video: int | None = None,
) -> dict[str, Any]:
    configure_logging()
    settings = load_settings()
    ensure_data_dirs()
    youtube = build_client(settings.youtube_api_key)

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find manifest: {manifest_path}")

    manifest = load_json(manifest_path)
    run_id = manifest.get("run_id", run_dir.name.removeprefix("youtube_run_"))
    manifest["stage"] = "top_level_comments_round_robin"
    manifest["status"] = "running"
    manifest["comments_started_at"] = datetime.now(timezone.utc).isoformat()
    manifest["round_robin_max_videos"] = max_videos
    manifest["max_pagination_steps_per_video"] = max_pagination_steps_per_video
    save_manifest(run_dir, manifest)

    run_payload: dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage": "top_level_comments_round_robin",
        "max_videos": max_videos,
        "max_pagination_steps_per_video": max_pagination_steps_per_video,
        "ips": [],
    }

    ip_payloads: dict[str, dict[str, Any]] = {}
    ip_dirs: dict[str, Path] = {}
    ip_candidate_counts: dict[str, int] = {}
    queues: dict[str, deque[int]] = {}

    for ip_name in manifest.get("ips", {}):
        ip_dir = run_dir / safe_filename(ip_name)
        ip_payload_path = ip_dir / "ip_payload.json"
        if not ip_payload_path.exists():
            continue

        ip_payload = load_json(ip_payload_path)
        ip_payloads[ip_name] = ip_payload
        ip_dirs[ip_name] = ip_dir
        ip_candidate_counts[ip_name] = int(ip_payload.get("candidate_video_count", 0))
        queues[ip_name] = deque(
            index
            for index, video in enumerate(ip_payload.get("selected_videos", []))
            if video.get("status") != "completed"
        )

        ip_status = manifest["ips"][ip_name]
        ip_status["status"] = "queued_for_round_robin_comments"
        ip_status["target_video_count"] = len(queues[ip_name])
        ip_status["videos_completed"] = sum(
            1 for video in ip_payload.get("selected_videos", []) if video.get("status") == "completed"
        )
        ip_status["total_top_level_comments_fetched"] = sum(
            video.get("fetched_thread_count", 0) for video in ip_payload.get("selected_videos", [])
        )
        ip_status["total_replies_fetched"] = 0

    save_manifest(run_dir, manifest)

    current_ip_name: str | None = None
    videos_collected = 0

    try:
        active_ip_names = [ip_name for ip_name, queue in queues.items() if queue]
        LOGGER.info(
            "Starting round-robin top-level comment collection for %s IPs with max_videos=%s and max_pagination_steps_per_video=%s",
            len(active_ip_names),
            max_videos,
            max_pagination_steps_per_video,
        )
        while active_ip_names and videos_collected < max_videos:
            progressed = False
            LOGGER.info(
                "Starting round-robin pass with %s active IPs and %s/%s videos collected so far",
                len(active_ip_names),
                videos_collected,
                max_videos,
            )
            for ip_name in list(active_ip_names):
                if videos_collected >= max_videos:
                    break
                queue = queues[ip_name]
                if not queue:
                    continue

                current_ip_name = ip_name
                ip_payload = ip_payloads[ip_name]
                ip_dir = ip_dirs[ip_name]
                ip_payload_path = ip_dir / "ip_payload.json"
                index = queue.popleft()
                video_payload = ip_payload["selected_videos"][index]
                if video_payload.get("status") == "completed":
                    continue

                manifest["ips"][ip_name]["status"] = "running_comments"
                save_manifest(run_dir, manifest)

                video_id = video_payload["video_id"]
                estimated_total_comments = int(video_payload.get("comment_count", 0) or 0)
                video_output_path = ip_dir / f"video_{safe_filename(video_id)}.json"
                LOGGER.info(
                    "Round-robin fetching top-level comments for IP '%s', video %s (queue_index=%s, estimated comment count=%s, pages_limit=%s)",
                    ip_name,
                    video_id,
                    index,
                    estimated_total_comments,
                    max_pagination_steps_per_video,
                )
                latest_checkpointed_video: dict[str, Any] | None = None

                def checkpoint_video(comment_threads: list[dict[str, Any]]) -> None:
                    nonlocal latest_checkpointed_video
                    fetched_thread_count, fetched_reply_count = summarize_comment_threads(comment_threads)
                    latest_checkpointed_video = {
                        "video_id": video_id,
                        "matched_queries": video_payload["matched_queries"],
                        "search_snippets": video_payload["search_snippets"],
                        "video_details": video_payload["video_details"],
                        "comment_count": video_payload.get("comment_count", 0),
                        "view_count": video_payload.get("view_count", 0),
                        "ip_name": video_payload.get("ip_name", ip_name),
                        "all_comment_threads": comment_threads,
                        "fetched_thread_count": fetched_thread_count,
                        "fetched_reply_count": fetched_reply_count,
                        "error": None,
                        "status": "partial",
                    }
                    write_json(video_output_path, latest_checkpointed_video)

                try:
                    comment_threads = fetch_comment_threads(
                        youtube=youtube,
                        video_id=video_id,
                        estimated_total_comments=estimated_total_comments,
                        checkpoint=checkpoint_video,
                        max_pagination_steps=max_pagination_steps_per_video,
                    )
                    error_message = None
                except HttpError as exc:
                    if is_quota_exceeded(exc):
                        if latest_checkpointed_video is not None:
                            ip_payload["selected_videos"][index] = latest_checkpointed_video
                            finalized_ip_payload = finalize_ip_payload(ip_payload, ip_candidate_counts[ip_name])
                            write_json(ip_payload_path, finalized_ip_payload)
                            manifest["ips"][ip_name]["videos_completed"] = sum(
                                1 for video in ip_payload["selected_videos"] if video.get("status") == "completed"
                            )
                            manifest["ips"][ip_name]["total_top_level_comments_fetched"] = sum(
                                video.get("fetched_thread_count", 0) for video in ip_payload["selected_videos"]
                            )
                            save_manifest(run_dir, manifest)
                            LOGGER.warning(
                                "Quota exceeded while fetching %s for %s. Latest partial checkpoint was saved.",
                                video_id,
                                ip_name,
                            )
                        raise QuotaExceededError(str(exc)) from exc
                    comment_threads = []
                    error_message = str(exc)
                    LOGGER.warning("Video '%s' failed: %s", video_id, error_message)

                assembled_video = {
                    "video_id": video_id,
                    "matched_queries": video_payload["matched_queries"],
                    "search_snippets": video_payload["search_snippets"],
                    "video_details": video_payload["video_details"],
                    "comment_count": video_payload.get("comment_count", 0),
                    "view_count": video_payload.get("view_count", 0),
                    "ip_name": video_payload.get("ip_name", ip_name),
                    "all_comment_threads": comment_threads,
                    "fetched_thread_count": len(comment_threads),
                    "fetched_reply_count": 0,
                    "error": error_message,
                    "status": "completed" if error_message is None else "error",
                }
                write_json(video_output_path, assembled_video)
                ip_payload["selected_videos"][index] = assembled_video
                finalized_ip_payload = finalize_ip_payload(ip_payload, ip_candidate_counts[ip_name])
                write_json(ip_payload_path, finalized_ip_payload)

                manifest["ips"][ip_name]["videos_completed"] = sum(
                    1 for video in ip_payload["selected_videos"] if video.get("status") == "completed"
                )
                manifest["ips"][ip_name]["total_top_level_comments_fetched"] = sum(
                    video.get("fetched_thread_count", 0) for video in ip_payload["selected_videos"]
                )
                manifest["ips"][ip_name]["total_replies_fetched"] = 0
                save_manifest(run_dir, manifest)
                LOGGER.info(
                    "Saved completed video %s for %s with %s top-level comments fetched. Round-robin progress: %s/%s videos.",
                    video_id,
                    ip_name,
                    assembled_video["fetched_thread_count"],
                    videos_collected + 1,
                    max_videos,
                )

                videos_collected += 1
                progressed = True

            active_ip_names = [ip_name for ip_name in active_ip_names if queues[ip_name]]
            if not progressed:
                LOGGER.info("No further progress in round-robin loop; stopping early.")
                break

        for ip_name, ip_payload in ip_payloads.items():
            finalized_ip_payload = finalize_ip_payload(ip_payload, ip_candidate_counts[ip_name])
            write_json(ip_dirs[ip_name] / "ip_payload.json", finalized_ip_payload)
            manifest["ips"][ip_name]["status"] = (
                "completed"
                if not queues[ip_name]
                else "paused_after_round_robin_limit"
            )
            run_payload["ips"].append(finalized_ip_payload)

    except QuotaExceededError as exc:
        for ip_name, ip_payload in ip_payloads.items():
            finalized_ip_payload = finalize_ip_payload(
                ip_payload,
                ip_candidate_counts[ip_name],
            )
            write_json(ip_dirs[ip_name] / "ip_payload.json", finalized_ip_payload)
            if not any(existing_ip.get("ip_name") == ip_name for existing_ip in run_payload["ips"]):
                run_payload["ips"].append(finalized_ip_payload)
        if current_ip_name is not None:
            manifest["ips"][current_ip_name]["status"] = "partial"
        manifest["status"] = "quota_exceeded"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = str(exc)
        save_manifest(run_dir, manifest)
        partial_path = run_dir / "partial_run_payload.json"
        write_json(partial_path, run_payload)
        LOGGER.error("Quota exceeded during round-robin comments. Partial payload saved to %s", partial_path)
        raise

    output_path = output_dir / f"youtube_sample_{run_id}.json"
    write_json(output_path, run_payload)
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["comments_output_path"] = str(output_path)
    manifest["videos_collected_this_round"] = videos_collected
    save_manifest(run_dir, manifest)
    LOGGER.info("Finished round-robin comment stage. Final output saved to %s", output_path)
    return {
        "output_path": str(output_path),
        "run_dir": str(run_dir),
        "ips_collected": len(run_payload["ips"]),
        "videos_collected": videos_collected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch top-level comments in round-robin order across IPs and videos from a saved youtube_run directory."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--max-videos",
        type=int,
        required=True,
        help="Maximum number of videos to fetch comments for in this round-robin pass.",
    )
    parser.add_argument(
        "--max-pagination-steps-per-video",
        type=int,
        default=None,
        help="Optional cap on commentThreads.list pages per video. Each page returns up to 100 top-level comments and costs one API unit.",
    )
    args = parser.parse_args()

    result = collect_round_robin_top_level_comments(
        run_dir=args.run_dir,
        output_dir=DATA_RAW_DIR,
        max_videos=args.max_videos,
        max_pagination_steps_per_video=args.max_pagination_steps_per_video,
    )
    print(result)


if __name__ == "__main__":
    main()
