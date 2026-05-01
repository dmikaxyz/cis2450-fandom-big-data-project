from __future__ import annotations

from config import load_settings


def main() -> None:
    settings = load_settings()
    checks = {
        "YOUTUBE_API_KEY": bool(settings.youtube_api_key),
        "REDDIT_CLIENT_ID": bool(settings.reddit_client_id),
        "REDDIT_CLIENT_SECRET": bool(settings.reddit_client_secret),
        "REDDIT_USER_AGENT": bool(settings.reddit_user_agent),
        "BLUESKY_HANDLE": bool(settings.bluesky_handle),
        "BLUESKY_APP_PASSWORD": bool(settings.bluesky_app_password),
    }
    print(checks)
    missing = [name for name, is_set in checks.items() if not is_set]
    if missing:
        raise SystemExit(f"Missing required values in .env: {', '.join(missing)}")


if __name__ == "__main__":
    main()
