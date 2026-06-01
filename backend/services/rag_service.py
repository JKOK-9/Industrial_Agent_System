from __future__ import annotations

import csv
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Protocol

from .safe_paths import safe_name
from ..config import OUTPUTS_DIR
from ..schemas import RagConfigRequest
from ..storage import Registry, utc_now


class UploadedFile(Protocol):
    filename: str

    def read(self, size: int = -1) -> bytes:
        ...


ALLOWED_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv"}
MAX_INDEX_CHARS = 2_000_000
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class RagService:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def create_rag_model(self, request: RagConfigRequest, knowledge_file: UploadedFile) -> dict:
        base_model = self.registry.get("base_models", request.model_id)
        if not base_model:
            raise ValueError("基座模型不存在。")
        if base_model.get("status") != "ready":
            raise ValueError("基座模型尚未 ready。")

        suffix = Path(knowledge_file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError("RAG 数据文件仅支持 .txt、.md、.json、.jsonl、.csv、.tsv。")

        model_record_id = uuid.uuid4().hex
        output_dir = (OUTPUTS_DIR / f"{safe_name(request.display_name)}-rag-{model_record_id[:8]}").resolve()
        documents_dir = output_dir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)

        original_stem = Path(knowledge_file.filename or "knowledge").stem
        stored_name = f"{safe_name(original_stem)}{suffix}"
        stored_path = documents_dir / stored_name

        try:
            self._save_upload(knowledge_file, stored_path)
            extracted_text = self._extract_text(stored_path, suffix)
            normalized_text = self._normalize_text(extracted_text)
            if not normalized_text:
                raise ValueError("RAG 数据文件内容为空，无法建立知识索引。")

            truncated = len(normalized_text) > MAX_INDEX_CHARS
            indexed_text = normalized_text[:MAX_INDEX_CHARS]
            chunks = self._chunk_text(indexed_text)
            index = [
                {
                    "id": index_id,
                    "text": chunk,
                    "tokens": sorted(_tokenize(chunk)),
                }
                for index_id, chunk in enumerate(chunks, start=1)
            ]

            manifest_path = output_dir / "rag_manifest.json"
            index_path = output_dir / "rag_index.json"
            runtime_path = output_dir / "rag_runtime.py"
            manifest = {
                "id": model_record_id,
                "display_name": request.display_name,
                "model_type": "rag",
                "base_model_id": base_model["id"],
                "base_model_name": base_model["display_name"],
                "base_model_path": base_model["path"],
                "domain": request.domain,
                "prompt": request.prompt,
                "document": {
                    "original_name": knowledge_file.filename,
                    "stored_name": stored_name,
                    "path": str(stored_path),
                    "suffix": suffix,
                    "indexed_chars": len(indexed_text),
                    "truncated": truncated,
                },
                "index_path": str(index_path),
                "chunk_count": len(index),
                "created_at": utc_now(),
            }

            _write_json(manifest_path, manifest)
            _write_json(index_path, {"chunks": index})
            runtime_path.write_text(_runtime_source(), encoding="utf-8")

            model = {
                "id": model_record_id,
                "display_name": request.display_name,
                "domain": request.domain,
                "base_model_id": base_model["id"],
                "base_model_name": base_model["display_name"],
                "training_method": "rag",
                "model_type": "rag",
                "path": str(output_dir),
                "status": "ready",
                "rag": {
                    "prompt": request.prompt,
                    "document_name": knowledge_file.filename,
                    "chunk_count": len(index),
                    "manifest_path": str(manifest_path),
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            self.registry.add("fine_tuned_models", model)
            return model
        except Exception:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            raise

    def build_prompt(self, model_id: str, question: str, top_k: int = 4) -> dict:
        model = self.registry.get("fine_tuned_models", model_id)
        if not model:
            raise ValueError("RAG 模型不存在。")
        if model.get("model_type") != "rag":
            raise ValueError("该模型不是 RAG 封装模型。")

        manifest_path_value = model.get("rag", {}).get("manifest_path")
        if not manifest_path_value:
            raise ValueError("RAG 模型清单文件不存在。")
        manifest_path = Path(manifest_path_value)
        if not manifest_path.is_file():
            raise ValueError("RAG 模型清单文件不存在。")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = Path(manifest["index_path"])
        index = json.loads(index_path.read_text(encoding="utf-8"))
        contexts = _retrieve(question, index.get("chunks", []), top_k)
        context_text = "\n\n".join(f"[片段 {item['id']}]\n{item['text']}" for item in contexts) or "未检索到相关片段。"
        prompt = (
            f"{manifest['prompt']}\n\n"
            "请优先依据以下资料片段回答用户问题；如果资料不足，请明确说明不足之处。\n\n"
            f"资料片段：\n{context_text}\n\n"
            f"用户问题：{question}\n\n"
            "回答："
        )

        return {
            "model_id": model_id,
            "model_name": model["display_name"],
            "base_model_id": manifest["base_model_id"],
            "base_model_name": manifest["base_model_name"],
            "base_model_path": manifest["base_model_path"],
            "question": question,
            "contexts": contexts,
            "prompt": prompt,
        }

    def _save_upload(self, upload: UploadedFile, destination: Path) -> None:
        stream = getattr(upload, "stream", upload)
        with destination.open("wb") as handle:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    def _extract_text(self, path: Path, suffix: str) -> str:
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".jsonl":
            rows: list[str] = []
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    rows.append(_json_to_text(json.loads(stripped)))
            return "\n\n".join(rows)
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return _json_to_text(payload)
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                return "\n".join(" | ".join(cell.strip() for cell in row) for row in csv.reader(handle, delimiter=delimiter))
        return ""

    def _normalize_text(self, text: str) -> str:
        lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        return "\n".join(line for line in lines if line).strip()

    def _chunk_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
        return chunks


def _json_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_json_to_text(item) for item in value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _json_to_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "\n".join(parts)
    return str(value)


def _tokenize(text: str) -> set[str]:
    tokens = {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}
    compact_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    for size in (2, 3, 4):
        tokens.update(compact_cjk[index : index + size] for index in range(max(len(compact_cjk) - size + 1, 0)))
    return {token for token in tokens if token.strip()}


def _retrieve(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    query_tokens = _tokenize(question)
    scored = []
    for chunk in chunks:
        tokens = set(chunk.get("tokens") or [])
        overlap = query_tokens & tokens
        if not overlap:
            continue
        score = len(overlap) / max(len(query_tokens), 1)
        scored.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "score": round(score, 4),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _runtime_source() -> str:
    return '''from __future__ import annotations

import json
import re
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\\u4e00-\\u9fff]")


def tokenize(text: str) -> set[str]:
    tokens = {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}
    compact_cjk = "".join(re.findall(r"[\\u4e00-\\u9fff]", text))
    for size in (2, 3, 4):
        tokens.update(compact_cjk[index : index + size] for index in range(max(len(compact_cjk) - size + 1, 0)))
    return {token for token in tokens if token.strip()}


def build_prompt(question: str, top_k: int = 4, package_dir: str | Path | None = None) -> dict:
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    manifest = json.loads((root / "rag_manifest.json").read_text(encoding="utf-8"))
    index = json.loads((root / "rag_index.json").read_text(encoding="utf-8"))
    query_tokens = tokenize(question)
    contexts = []
    for chunk in index.get("chunks", []):
        overlap = query_tokens & set(chunk.get("tokens") or [])
        if not overlap:
            continue
        contexts.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "score": round(len(overlap) / max(len(query_tokens), 1), 4),
            }
        )
    contexts.sort(key=lambda item: item["score"], reverse=True)
    contexts = contexts[:top_k]
    context_text = "\\n\\n".join(f"[片段 {item['id']}]\\n{item['text']}" for item in contexts) or "未检索到相关片段。"
    prompt = (
        f"{manifest['prompt']}\\n\\n"
        "请优先依据以下资料片段回答用户问题；如果资料不足，请明确说明不足之处。\\n\\n"
        f"资料片段：\\n{context_text}\\n\\n"
        f"用户问题：{question}\\n\\n"
        "回答："
    )
    return {
        "base_model_path": manifest["base_model_path"],
        "contexts": contexts,
        "prompt": prompt,
    }
'''
