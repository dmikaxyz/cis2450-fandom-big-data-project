from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
DATA_DUCKDB_DIR = ROOT / "data" / "duckdb"


@dataclass(frozen=True)
class Settings:
    youtube_api_key: str
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    bluesky_handle: str
    bluesky_app_password: str


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "").strip(),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", "").strip(),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "").strip(),
        bluesky_handle=os.getenv("BLUESKY_HANDLE", "").strip(),
        bluesky_app_password=os.getenv("BLUESKY_APP_PASSWORD", "").strip(),
    )


def ensure_data_dirs() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DUCKDB_DIR.mkdir(parents=True, exist_ok=True)
