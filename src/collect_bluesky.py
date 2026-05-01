from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from tqdm.auto import tqdm

from config import DATA_RAW_DIR, ensure_data_dirs, load_settings
from ip_config import PRIMARY_IPS
from utils import safe_filename, write_json


LOGGER = logging.getLogger("collect_bluesky")
PDS = "https://bsky.social/xrpc"
APPVIEW = PDS
PAGE_SIZE = 100
SLEEP_SEC = 0.25
RETRY_STATUSES = {429, 500, 502, 503, 504}


class BlueskyClient:
    def __init__(self, handle: str, app_password: str) -> None:
        if not handle or not app_password:
            raise ValueError("Missing BLUESKY_HANDLE or BLUESKY_APP_PASSWORD in .env")
        self.handle = handle
        self.app_password = app_password
        self.session_payload: dict[str, str] = {}

    def login(self) -> None:
        response = requests.post(
            f"{PDS}/com.atproto.server.createSession",
            json={"identifier": self.handle, "password": self.app_password},
            timeout=20,
        )
        response.raise_for_status()
        self.session_payload = response.json()
        LOGGER.info("Authenticated to Bluesky as %s", self.handle)

    def refresh(self) -> None:
        refresh_jwt = self.session_payload.get("refreshJwt")
        if not refresh_jwt:
            self.login()
            return

        response = requests.post(
            f"{PDS}/com.atproto.server.refreshSession",
            headers={"Authorization": f"Bearer {refresh_jwt}"},
            timeout=20,
        )
        if response.status_code != 200:
            self.login()
            return
        self.session_payload = response.json()

    def get(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.session_payload:
            self.login()

        url = f"{APPVIEW}/{method}"
        for attempt in range(4):
            access_jwt = self.session_payload["accessJwt"]
            response = requests.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {access_jwt}"},
                timeout=20,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 401:
                self.refresh()
                continue
            if response.status_code in RETRY_STATUSES:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()

        raise RuntimeError(f"Bluesky {method} failed after retries: {response.status_code}")

    def search_posts(
        self,
        query: str,
        since: str | None,
        until: str | None,
        max_pages: int,
    ) -> Iterable[dict[str, Any]]:
        cursor = None
        for _ in range(max_pages):
            params: dict[str, Any] = {"q": query, "limit": PAGE_SIZE}
            if since:
                params["since"] = since
            if until:
                params["until"] = until
            if cursor:
                params["cursor"] = cursor

            data = self.get("app.bsky.feed.searchPosts", params)
            yield data

            cursor = data.get("cursor")
            if not cursor:
                return
            time.sleep(SLEEP_SEC)

    def get_thread_posts(self, uri: str, depth: int = 6) -> list[dict[str, Any]]:
        data = self.get("app.bsky.feed.getPostThread", {"uri": uri, "depth": depth})
        posts: list[dict[str, Any]] = []

        def walk(node: dict[str, Any]) -> None:
            post = node.get("post")
            if post:
                posts.append(post)
            for child in node.get("replies", []) or []:
                walk(child)

        walk(data.get("thread", {}) or {})
        return posts


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def iso_days_ago(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_client() -> BlueskyClient:
    settings = load_settings()
    return BlueskyClient(
        handle=settings.bluesky_handle,
        app_password=settings.bluesky_app_password,
    )


def post_uri(post: dict[str, Any]) -> str | None:
    uri = post.get("uri")
    return uri if isinstance(uri, str) and uri else None


def collect_bluesky_posts(
    output_dir: Path,
    queries_per_ip: int | None = None,
    days_back: int | None = 30,
    max_pages_per_query: int = 5,
    include_threads: bool = False,
    thread_reply_min: int = 5,
) -> dict[str, Any]:
    configure_logging()
    ensure_data_dirs()
    client = build_client()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    collected_at = datetime.now(timezone.utc).isoformat()
    run_dir = output_dir / "bluesky_runs" / f"bluesky_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    since = iso_days_ago(days_back) if days_back is not None else None
    until = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    run_payload: dict[str, Any] = {
        "collected_at": collected_at,
        "run_id": timestamp,
        "run_dir": str(run_dir),
        "stage": "search_posts",
        "params": {
            "queries_per_ip": queries_per_ip,
            "days_back": days_back,
            "since": since,
            "until": until,
            "max_pages_per_query": max_pages_per_query,
            "include_threads": include_threads,
            "thread_reply_min": thread_reply_min,
        },
        "ips": [],
    }
    manifest: dict[str, Any] = {
        "run_id": timestamp,
        "started_at": collected_at,
        "stage": "search_posts",
        "status": "running",
        "params": run_payload["params"],
        "ips": {},
    }
    write_json(run_dir / "manifest.json", manifest)

    seen_uris: set[str] = set()
    ip_iterator = tqdm(PRIMARY_IPS, desc="IPs", unit="ip")
    for ip_config in ip_iterator:
        ip_name = ip_config.canonical_name
        ip_iterator.set_postfix_str(ip_name)
        LOGGER.info("Starting Bluesky collection for IP '%s'", ip_name)

        ip_dir = run_dir / safe_filename(ip_name)
        ip_dir.mkdir(parents=True, exist_ok=True)

        active_queries = (
            ip_config.bluesky_queries[:queries_per_ip]
            if queries_per_ip is not None
            else list(ip_config.bluesky_queries)
        )
        ip_payload: dict[str, Any] = {
            "ip_name": ip_name,
            "queries_used": active_queries,
            "queries": [],
            "posts": [],
        }
        manifest["ips"][ip_name] = {
            "status": "running",
            "queries_used": active_queries,
            "post_count": 0,
        }
        write_json(run_dir / "manifest.json", manifest)

        ip_posts_by_uri: dict[str, dict[str, Any]] = {}
        for query in active_queries:
            LOGGER.info("Searching Bluesky query '%s' for '%s'", query, ip_name)
            query_posts: list[dict[str, Any]] = []
            pages_fetched = 0

            for page in client.search_posts(
                query=query,
                since=since,
                until=until,
                max_pages=max_pages_per_query,
            ):
                pages_fetched += 1
                for post in page.get("posts", []):
                    uri = post_uri(post)
                    if not uri:
                        continue
                    query_posts.append(post)
                    if uri not in ip_posts_by_uri:
                        ip_posts_by_uri[uri] = {
                            "ip_name": ip_name,
                            "matched_queries": [],
                            "post": post,
                            "thread_posts": [],
                        }
                    if query not in ip_posts_by_uri[uri]["matched_queries"]:
                        ip_posts_by_uri[uri]["matched_queries"].append(query)

                    if include_threads and post.get("replyCount", 0) >= thread_reply_min and uri not in seen_uris:
                        ip_posts_by_uri[uri]["thread_posts"] = client.get_thread_posts(uri)

                    seen_uris.add(uri)

            write_json(
                ip_dir / f"search_{safe_filename(query)}.json",
                {
                    "ip_name": ip_name,
                    "query": query,
                    "since": since,
                    "until": until,
                    "pages_fetched": pages_fetched,
                    "posts": query_posts,
                },
            )
            ip_payload["queries"].append(
                {
                    "query": query,
                    "pages_fetched": pages_fetched,
                    "post_count": len(query_posts),
                }
            )

        ip_payload["posts"] = sorted(
            ip_posts_by_uri.values(),
            key=lambda item: (item["post"].get("record", {}) or {}).get("createdAt", ""),
            reverse=True,
        )
        manifest["ips"][ip_name]["status"] = "completed"
        manifest["ips"][ip_name]["post_count"] = len(ip_payload["posts"])
        write_json(ip_dir / "ip_payload.json", ip_payload)
        write_json(run_dir / "manifest.json", manifest)
        run_payload["ips"].append(ip_payload)

    output_path = output_dir / f"bluesky_search_{timestamp}.json"
    write_json(output_path, run_payload)
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["final_output_path"] = str(output_path)
    write_json(run_dir / "manifest.json", manifest)

    LOGGER.info("Finished Bluesky collection. Raw output saved to %s", output_path)
    return {
        "output_path": str(output_path),
        "run_dir": str(run_dir),
        "ips_collected": len(run_payload["ips"]),
        "total_posts": sum(len(ip["posts"]) for ip in run_payload["ips"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Bluesky posts for configured fandom queries.")
    parser.add_argument(
        "--queries-per-ip",
        type=int,
        default=None,
        help="Optional cap on Bluesky queries per IP. If omitted, all configured queries are used.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Search posts from this many days back.",
    )
    parser.add_argument("--no-date-filter", action="store_true", help="Do not pass since/until filters to Bluesky search.")
    parser.add_argument("--max-pages-per-query", type=int, default=5)
    parser.add_argument("--include-threads", action="store_true")
    parser.add_argument("--thread-reply-min", type=int, default=5)
    args = parser.parse_args()

    result = collect_bluesky_posts(
        output_dir=DATA_RAW_DIR,
        queries_per_ip=args.queries_per_ip,
        days_back=None if args.no_date_filter else args.days_back,
        max_pages_per_query=args.max_pages_per_query,
        include_threads=args.include_threads,
        thread_reply_min=args.thread_reply_min,
    )
    print(result)


if __name__ == "__main__":
    main()
