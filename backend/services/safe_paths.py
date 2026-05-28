from __future__ import annotations

import re


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip(".-")
    return cleaned[:80] or "item"
