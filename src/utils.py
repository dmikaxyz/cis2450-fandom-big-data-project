from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    text = URL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
