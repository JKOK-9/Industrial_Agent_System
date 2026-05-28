from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal, Protocol


DatasetFormat = Literal["alpaca", "sharegpt", "openai"]


class UploadedFile(Protocol):
    filename: str

    def read(self, size: int = -1) -> bytes:
        ...


class DatasetValidationError(ValueError):
    pass


def save_and_prepare_dataset(
    upload: UploadedFile,
    dataset_dir: Path,
    dataset_name: str,
    dataset_format: DatasetFormat,
) -> dict[str, Any]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise DatasetValidationError("当前仅支持 .json 或 .jsonl 文件。")

    data_path = dataset_dir / f"{dataset_name}{suffix}"
    stream = getattr(upload, "stream", upload)
    with data_path.open("wb") as handle:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    samples = _load_samples(data_path)
    _validate_samples(samples, dataset_format)

    normalized_path = data_path
    if suffix == ".jsonl":
        normalized_path = dataset_dir / f"{dataset_name}.json"
        with normalized_path.open("w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False, indent=2)
        data_path.unlink(missing_ok=True)

    dataset_info = {
        dataset_name: _dataset_description(normalized_path.name, dataset_format),
    }
    with (dataset_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, ensure_ascii=False, indent=2)

    return {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "file_name": normalized_path.name,
        "sample_count": len(samples),
        "format": dataset_format,
    }


def remove_dataset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _load_samples(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".jsonl":
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    item = json.loads(stripped)
                    if not isinstance(item, dict):
                        raise DatasetValidationError(f"第 {line_number} 行必须是 JSON 对象。")
                    rows.append(item)
            return rows

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("data", "train", "rows", "samples"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise DatasetValidationError("JSON 文件顶层必须是数组，或包含 data/train/rows/samples 数组。")
        if not all(isinstance(item, dict) for item in payload):
            raise DatasetValidationError("数据集中的每条样本都必须是 JSON 对象。")
        return payload
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"JSON 解析失败：{exc.msg}") from exc


def _validate_samples(samples: list[dict[str, Any]], dataset_format: DatasetFormat) -> None:
    if not samples:
        raise DatasetValidationError("数据集不能为空。")

    validators = {
        "alpaca": _validate_alpaca_sample,
        "sharegpt": _validate_sharegpt_sample,
        "openai": _validate_openai_sample,
    }
    validator = validators[dataset_format]
    for index, sample in enumerate(samples[:20], start=1):
        validator(sample, index)


def _validate_alpaca_sample(sample: dict[str, Any], index: int) -> None:
    if not sample.get("instruction"):
        raise DatasetValidationError(f"第 {index} 条 Alpaca 样本缺少 instruction。")
    if not sample.get("output"):
        raise DatasetValidationError(f"第 {index} 条 Alpaca 样本缺少 output。")


def _validate_sharegpt_sample(sample: dict[str, Any], index: int) -> None:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise DatasetValidationError(f"第 {index} 条 ShareGPT 样本缺少 conversations。")
    for message in conversations:
        if not isinstance(message, dict) or "from" not in message or "value" not in message:
            raise DatasetValidationError(f"第 {index} 条 ShareGPT 样本消息必须包含 from 和 value。")


def _validate_openai_sample(sample: dict[str, Any], index: int) -> None:
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise DatasetValidationError(f"第 {index} 条 OpenAI 样本缺少 messages。")
    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise DatasetValidationError(f"第 {index} 条 OpenAI 样本消息必须包含 role 和 content。")


def _dataset_description(file_name: str, dataset_format: DatasetFormat) -> dict[str, Any]:
    if dataset_format == "alpaca":
        return {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        }

    if dataset_format == "sharegpt":
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }

    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
