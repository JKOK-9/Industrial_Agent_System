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

    uploaded_path = dataset_dir / f"{dataset_name}{suffix}"
    stream = getattr(upload, "stream", upload)
    with uploaded_path.open("wb") as handle:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    samples = _load_samples(uploaded_path)
    normalized_samples = _normalize_samples(samples)

    normalized_path = dataset_dir / f"{dataset_name}.json"
    with normalized_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized_samples, handle, ensure_ascii=False, indent=2)
    if uploaded_path != normalized_path:
        uploaded_path.unlink(missing_ok=True)

    dataset_info = {
        dataset_name: _dataset_description(normalized_path.name),
    }
    with (dataset_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, ensure_ascii=False, indent=2)

    return {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "file_name": normalized_path.name,
        "sample_count": len(normalized_samples),
        "format": "openai",
        "requested_format": dataset_format,
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
        if isinstance(payload, list):
            if not all(isinstance(item, dict) for item in payload):
                raise DatasetValidationError("数据集中的每条样本都必须是 JSON 对象。")
            return payload
        if isinstance(payload, dict):
            for key in ("data", "train", "rows", "samples"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    if not all(isinstance(item, dict) for item in rows):
                        raise DatasetValidationError(f"{key} 数组中的每条样本都必须是 JSON 对象。")
                    return rows
            if _looks_like_single_sample(payload):
                return [payload]
        raise DatasetValidationError("JSON 文件顶层必须是样本数组、单条样本对象，或包含 data/train/rows/samples 数组。")
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"JSON 解析失败：{exc.msg}") from exc


def _looks_like_single_sample(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "messages",
            "conversations",
            "instruction",
            "prompt",
            "question",
            "query",
            "input",
            "output",
            "response",
            "answer",
            "completion",
        )
    )


def _normalize_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not samples:
        raise DatasetValidationError("数据集不能为空。")

    normalized: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        normalized.append({"messages": _sample_to_messages(sample, index)})
    return normalized


def _sample_to_messages(sample: dict[str, Any], index: int) -> list[dict[str, str]]:
    if isinstance(sample.get("messages"), list):
        return _normalize_message_list(sample["messages"], index, role_keys=("role", "from"), content_keys=("content", "value", "text"))

    if isinstance(sample.get("conversations"), list):
        return _normalize_message_list(sample["conversations"], index, role_keys=("from", "role"), content_keys=("value", "content", "text"))

    system = _first_text(sample, ("system", "system_prompt"))
    instruction = _first_text(sample, ("instruction", "prompt", "question", "query", "task"))
    input_text = _first_text(sample, ("input", "user", "user_input"))
    response = _first_text(sample, ("output", "response", "answer", "completion", "assistant", "target"))

    if not instruction and input_text:
        instruction = input_text
        input_text = ""
    if not instruction:
        raise DatasetValidationError(
            f"第 {index} 条样本缺少用户输入字段；支持 messages/conversations，或 instruction/prompt/question/query/input。"
        )
    if not response:
        raise DatasetValidationError(f"第 {index} 条样本缺少助手输出字段；支持 output/response/answer/completion/assistant。")

    user_content = instruction
    if input_text and input_text != instruction:
        user_content = f"{instruction}\n{input_text}"

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})
    return messages


def _normalize_message_list(
    messages: list[Any],
    sample_index: int,
    role_keys: tuple[str, ...],
    content_keys: tuple[str, ...],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message_index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise DatasetValidationError(f"第 {sample_index} 条样本的第 {message_index} 条消息必须是 JSON 对象。")
        role = _normalize_role(_first_text(message, role_keys))
        content = _first_text(message, content_keys)
        if not role:
            raise DatasetValidationError(f"第 {sample_index} 条样本的第 {message_index} 条消息缺少可识别角色。")
        if not content:
            raise DatasetValidationError(f"第 {sample_index} 条样本的第 {message_index} 条消息缺少内容。")
        if role in {"system", "user", "assistant"}:
            normalized.append({"role": role, "content": content})

    if not any(message["role"] == "user" for message in normalized):
        raise DatasetValidationError(f"第 {sample_index} 条样本缺少 user/human 消息。")
    if not any(message["role"] == "assistant" for message in normalized):
        raise DatasetValidationError(f"第 {sample_index} 条样本缺少 assistant/gpt 消息。")
    return normalized


def _first_text(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    return ""


def _normalize_role(value: str) -> str:
    role = value.strip().lower()
    return {
        "system": "system",
        "user": "user",
        "human": "user",
        "customer": "user",
        "input": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
        "output": "assistant",
    }.get(role, "")


def _dataset_description(file_name: str) -> dict[str, Any]:
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
