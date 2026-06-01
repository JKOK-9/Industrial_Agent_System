from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REGISTRY_PATH, ensure_runtime_dirs


DEFAULT_REGISTRY = {
    "base_models": [],
    "fine_tuned_models": [],
    "training_jobs": [],
    "download_jobs": [],
    "knowledge_sources": [],
    "prompt_assets": [],
}


class Registry:
    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        ensure_runtime_dirs()
        self.path = path
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write(DEFAULT_REGISTRY)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return deepcopy(DEFAULT_REGISTRY)
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            data = deepcopy(DEFAULT_REGISTRY)

        for key, value in DEFAULT_REGISTRY.items():
            data.setdefault(key, deepcopy(value))
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._read()

    def list(self, collection: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._read().get(collection, [])

    def get(self, collection: str, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            for item in self._read().get(collection, []):
                if item.get("id") == item_id:
                    return item
        return None

    def add(self, collection: str, item: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        item.setdefault("created_at", now)
        item.setdefault("updated_at", now)
        with self._lock:
            data = self._read()
            data.setdefault(collection, []).append(item)
            self._write(data)
        return item

    def update(self, collection: str, item_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()
            items = data.setdefault(collection, [])
            for index, item in enumerate(items):
                if item.get("id") == item_id:
                    merged = {**item, **updates, "updated_at": utc_now()}
                    items[index] = merged
                    self._write(data)
                    return merged
        return None

    def delete(self, collection: str, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()
            items = data.setdefault(collection, [])
            for index, item in enumerate(items):
                if item.get("id") == item_id:
                    removed = items.pop(index)
                    self._write(data)
                    return removed
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
