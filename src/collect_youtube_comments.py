from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from googleapiclient.errors import HttpError
from tqdm.auto import tqdm

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


def load_video_ids(video_id_args: list[str], video_ids_file: Path | None) -> set[str]:
    video_ids = {video_id.strip() for video_id in video_id_args if video_id.strip()}
    if video_ids_file is not None:
        lines = [line.strip() for line in video_ids_file.read_text(encoding="utf-8").splitlines()]
        video_ids.update(line for line in lines if line and not line.startswith("#"))
    if not video_ids:
        raise ValueError("Provide at least one --video-id or a --video-ids-file for comment collection.")
    return video_ids


def collect_youtube_top_level_comments(
    run_dir: Path,
    output_dir: Path,
    target_video_ids: set[str],
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
    manifest["stage"] = "top_level_comments"
    manifest["status"] = "running"
    manifest["comments_started_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(run_dir, manifest)

    run_payload: dict[str, Any] = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage": "top_level_comments",
        "target_video_ids": sorted(target_video_ids),
        "ips": [],
    }

    current_ip_name: str | None = None
    current_ip_dir: Path | None = None
    current_ip_payload: dict[str, Any] | None = None
    current_candidate_video_count = 0

    try:
        ip_items = list(manifest.get("ips", {}).items())
        ip_iterator = tqdm(ip_items, desc="IPs", unit="ip")
        for ip_name, ip_status in ip_iterator:
            current_ip_name = ip_name
            ip_iterator.set_postfix_str(ip_name)
            LOGGER.info("Starting comment stage for IP '%s'", ip_name)

            ip_dir = run_dir / safe_filename(ip_name)
            ip_payload_path = ip_dir / "ip_payload.json"
            if not ip_payload_path.exists():
                raise FileNotFoundError(f"Could not find IP payload: {ip_payload_path}")

            ip_payload = load_json(ip_payload_path)
            current_ip_payload = ip_payload
            current_ip_dir = ip_dir
            current_candidate_video_count = int(ip_payload.get("candidate_video_count", 0))

            ip_status["status"] = "running_comments"
            ip_status["target_video_count"] = sum(
                1 for video in ip_payload.get("selected_videos", []) if video.get("video_id") in target_video_ids
            )
            ip_status["videos_completed"] = sum(
                1
                for video in ip_payload.get("selected_videos", [])
                if video.get("video_id") in target_video_ids and video.get("status") == "completed"
            )
            ip_status["total_top_level_comments_fetched"] = sum(
                video.get("fetched_thread_count", 0)
                for video in ip_payload.get("selected_videos", [])
                if video.get("video_id") in target_video_ids
            )
            ip_status["total_replies_fetched"] = 0
            save_manifest(run_dir, manifest)

            video_iterator = tqdm(ip_payload.get("selected_videos", []), desc=f"{ip_name[:18]} videos", unit="video", leave=False)
            for index, video_payload in enumerate(video_iterator):
                if video_payload.get("video_id") not in target_video_ids:
                    continue
                if video_payload.get("status") == "completed":
                    continue

                video_id = video_payload["video_id"]
                estimated_total_comments = int(video_payload.get("comment_count", 0) or 0)
                video_output_path = ip_dir / f"video_{safe_filename(video_id)}.json"
                video_iterator.set_postfix_str(video_id[:8])
                LOGGER.info(
                    "Fetching top-level comments for video '%s' (%s) with estimated comment count %s",
                    video_id,
                    ip_name,
                    estimated_total_comments,
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
                    )
                    error_message = None
                except HttpError as exc:
                    if is_quota_exceeded(exc):
                        if latest_checkpointed_video is not None:
                            ip_payload["selected_videos"][index] = latest_checkpointed_video
                            finalized_ip_payload = finalize_ip_payload(ip_payload, current_candidate_video_count)
                            write_json(ip_payload_path, finalized_ip_payload)
                            ip_status["videos_completed"] = sum(
                                1
                                for video in ip_payload["selected_videos"]
                                if video.get("video_id") in target_video_ids and video.get("status") == "completed"
                            )
                            ip_status["total_top_level_comments_fetched"] = sum(
                                video.get("fetched_thread_count", 0)
                                for video in ip_payload["selected_videos"]
                                if video.get("video_id") in target_video_ids
                            )
                            save_manifest(run_dir, manifest)
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
                finalized_ip_payload = finalize_ip_payload(ip_payload, current_candidate_video_count)
                write_json(ip_payload_path, finalized_ip_payload)

                ip_status["videos_completed"] = sum(
                    1
                    for video in ip_payload["selected_videos"]
                    if video.get("video_id") in target_video_ids and video.get("status") == "completed"
                )
                ip_status["total_top_level_comments_fetched"] = sum(
                    video.get("fetched_thread_count", 0)
                    for video in ip_payload["selected_videos"]
                    if video.get("video_id") in target_video_ids
                )
                ip_status["total_replies_fetched"] = 0
                save_manifest(run_dir, manifest)

            finalized_ip_payload = finalize_ip_payload(ip_payload, current_candidate_video_count)
            write_json(ip_payload_path, finalized_ip_payload)
            ip_status["status"] = "completed"
            save_manifest(run_dir, manifest)
            run_payload["ips"].append(finalized_ip_payload)
            current_ip_name = None
            current_ip_dir = None
            current_ip_payload = None
            current_candidate_video_count = 0

    except QuotaExceededError as exc:
        if current_ip_name and current_ip_payload and current_ip_dir:
            finalized_ip_payload = finalize_ip_payload(current_ip_payload, current_candidate_video_count)
            write_json(current_ip_dir / "ip_payload.json", finalized_ip_payload)
            if not any(ip.get("ip_name") == current_ip_name for ip in run_payload["ips"]):
                run_payload["ips"].append(finalized_ip_payload)
            manifest["ips"][current_ip_name]["status"] = "partial"
        manifest["status"] = "quota_exceeded"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = str(exc)
        save_manifest(run_dir, manifest)
        partial_path = run_dir / "partial_run_payload.json"
        write_json(partial_path, run_payload)
        LOGGER.error("Quota exceeded during comment stage. Partial payload saved to %s", partial_path)
        raise

    output_path = output_dir / f"youtube_sample_{run_id}.json"
    write_json(output_path, run_payload)
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["comments_output_path"] = str(output_path)
    save_manifest(run_dir, manifest)
    LOGGER.info("Finished comment stage. Final output saved to %s", output_path)
    return {
        "output_path": str(output_path),
        "run_dir": str(run_dir),
        "ips_collected": len(run_payload["ips"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a saved YouTube search run directory and fetch top-level comments for explicitly chosen videos."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id", action="append", default=[], help="Video ID to fetch top-level comments for. Repeatable.")
    parser.add_argument(
        "--video-ids-file",
        type=Path,
        default=None,
        help="Optional text file with one video ID per line.",
    )
    args = parser.parse_args()
    target_video_ids = load_video_ids(args.video_id, args.video_ids_file)

    result = collect_youtube_top_level_comments(
        run_dir=args.run_dir,
        output_dir=DATA_RAW_DIR,
        target_video_ids=target_video_ids,
    )
    print(result)


if __name__ == "__main__":
    main()
