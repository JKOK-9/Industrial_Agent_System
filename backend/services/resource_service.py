from __future__ import annotations

import csv
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Protocol

from .safe_paths import safe_name
from ..config import KNOWLEDGE_DIR, PROMPTS_DIR, is_path_inside
from ..schemas import KnowledgeSourceRequest, PromptAssetRequest
from ..storage import Registry, utc_now


class UploadedFile(Protocol):
    filename: str

    def read(self, size: int = -1) -> bytes:
        ...


KNOWLEDGE_SUFFIXES = {
    "text": {".txt"},
    "markdown": {".md", ".txt"},
    "json": {".json", ".jsonl"},
    "table": {".csv", ".tsv"},
}
PROMPT_SUFFIXES = {".txt"}


class ResourceService:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def list_knowledge_sources(self) -> list[dict]:
        return sorted(self.registry.list("knowledge_sources"), key=lambda item: item.get("created_at", ""), reverse=True)

    def list_prompt_assets(self) -> list[dict]:
        return sorted(self.registry.list("prompt_assets"), key=lambda item: item.get("created_at", ""), reverse=True)

    def create_knowledge_source(self, request: KnowledgeSourceRequest, upload: UploadedFile) -> dict:
        suffix = Path(upload.filename or "").suffix.lower()
        allowed_suffixes = KNOWLEDGE_SUFFIXES[request.source_type]
        if suffix not in allowed_suffixes:
            allowed_text = "、".join(sorted(allowed_suffixes))
            raise ValueError(f"{_knowledge_type_text(request.source_type)}知识源仅支持 {allowed_text} 文件。")

        source_id = uuid.uuid4().hex
        source_dir = (KNOWLEDGE_DIR / source_id).resolve()
        source_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{safe_name(Path(upload.filename or request.name).stem)}{suffix}"
        stored_path = source_dir / stored_name

        try:
            self._save_upload(upload, stored_path)
            preview, stats = self._summarize_knowledge(stored_path, request.source_type, suffix)
            item = {
                "id": source_id,
                "name": request.name,
                "source_type": request.source_type,
                "type_label": _knowledge_type_text(request.source_type),
                "description": request.description,
                "original_name": upload.filename,
                "stored_name": stored_name,
                "path": str(stored_path),
                "size_bytes": stored_path.stat().st_size,
                "preview": preview,
                "stats": stats,
                "status": "ready",
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            self.registry.add("knowledge_sources", item)
            return item
        except Exception:
            if source_dir.exists():
                shutil.rmtree(source_dir)
            raise

    def create_prompt_asset(self, request: PromptAssetRequest, upload: UploadedFile | None = None) -> dict:
        prompt_id = uuid.uuid4().hex
        prompt_dir = (PROMPTS_DIR / prompt_id).resolve()
        prompt_dir.mkdir(parents=True, exist_ok=True)

        try:
            if upload and upload.filename:
                suffix = Path(upload.filename).suffix.lower()
                if suffix not in PROMPT_SUFFIXES:
                    raise ValueError("提示词文件仅支持 .txt。")
                stored_name = f"{safe_name(Path(upload.filename).stem)}{suffix}"
                stored_path = prompt_dir / stored_name
                self._save_upload(upload, stored_path)
                content = self._read_prompt_content(stored_path, suffix)
                original_name = upload.filename
            else:
                content = request.content.strip()
                if not content:
                    raise ValueError("请填写提示词内容或上传提示词文件。")
                stored_name = "prompt.txt"
                stored_path = prompt_dir / stored_name
                stored_path.write_text(content, encoding="utf-8")
                original_name = ""

            content = content.strip()
            if not content:
                raise ValueError("提示词内容不能为空。")

            item = {
                "id": prompt_id,
                "name": request.name,
                "prompt_type": request.prompt_type,
                "type_label": _prompt_type_text(request.prompt_type),
                "description": request.description,
                "original_name": original_name,
                "stored_name": stored_name,
                "path": str(stored_path),
                "size_bytes": stored_path.stat().st_size,
                "content": content,
                "preview": _preview(content),
                "status": "ready",
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            self.registry.add("prompt_assets", item)
            return item
        except Exception:
            if prompt_dir.exists():
                shutil.rmtree(prompt_dir)
            raise

    def delete_knowledge_source(self, item_id: str) -> bool:
        item = self.registry.delete("knowledge_sources", item_id)
        if not item:
            return False
        self._delete_managed_path(item.get("path", ""), KNOWLEDGE_DIR)
        return True

    def delete_prompt_asset(self, item_id: str) -> bool:
        item = self.registry.delete("prompt_assets", item_id)
        if not item:
            return False
        self._delete_managed_path(item.get("path", ""), PROMPTS_DIR)
        return True

    def batch_delete_knowledge_sources(self, item_ids: list[str]) -> dict:
        deleted = [item_id for item_id in item_ids if self.delete_knowledge_source(item_id)]
        return {"deleted": deleted, "deleted_count": len(deleted)}

    def batch_delete_prompt_assets(self, item_ids: list[str]) -> dict:
        deleted = [item_id for item_id in item_ids if self.delete_prompt_asset(item_id)]
        return {"deleted": deleted, "deleted_count": len(deleted)}

    def _save_upload(self, upload: UploadedFile, destination: Path) -> None:
        stream = getattr(upload, "stream", upload)
        with destination.open("wb") as handle:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    def _summarize_knowledge(self, path: Path, source_type: str, suffix: str) -> tuple[str, dict[str, Any]]:
        if source_type in {"text", "markdown"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            return _preview(text), {"chars": len(text), "lines": len(text.splitlines())}

        if source_type == "json":
            if suffix == ".jsonl":
                rows = []
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if stripped:
                            rows.append(json.loads(stripped))
                text = "\n".join(_json_to_text(row) for row in rows[:5])
                return _preview(text), {"rows": len(rows)}

            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            text = _json_to_text(payload)
            count = len(payload) if isinstance(payload, (list, dict)) else 1
            return _preview(text), {"items": count}

        delimiter = "\t" if suffix == ".tsv" else ","
        row_count = 0
        preview_rows = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.reader(handle, delimiter=delimiter):
                row_count += 1
                if len(preview_rows) < 5:
                    preview_rows.append(" | ".join(cell.strip() for cell in row))
        return _preview("\n".join(preview_rows)), {"rows": row_count}

    def _read_prompt_content(self, path: Path, suffix: str) -> str:
        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(payload, dict):
            for key in ("prompt", "content", "text", "system"):
                if payload.get(key):
                    return str(payload[key])
        return _json_to_text(payload)

    def _delete_managed_path(self, path_value: str, root: Path) -> None:
        path = Path(path_value).resolve()
        target_dir = path.parent if path.is_file() else path
        if is_path_inside(target_dir, root) and target_dir.exists():
            shutil.rmtree(target_dir)


def _json_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_json_to_text(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_json_to_text(item)}" for key, item in value.items())
    return str(value)


def _preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _knowledge_type_text(source_type: str) -> str:
    return {
        "text": "文本",
        "markdown": "Markdown",
        "json": "JSON",
        "table": "表格",
    }.get(source_type, source_type)


def _prompt_type_text(prompt_type: str) -> str:
    return {
        "system": "系统提示词",
        "instruction": "任务指令",
        "template": "模板",
        "other": "其他",
    }.get(prompt_type, prompt_type)
